"""Windowed neural inference for the Redact model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np

from ._assets import Assets, resolve_assets
from ._deterministic import OWNED
from ._deterministic import detect as deterministic_detect
from ._head import Head
from ._pipeline import (
    Span,
    attach_building_numbers,
    attach_state_codes,
    bioes_to_spans,
    bridge_name_gaps,
    clean_spans,
    extend_particle_names,
    hysteresis,
    mask_text,
    merge_priority,
    redact_secondary_address,
    redact_us_street,
    relabel_by_context,
    resolve,
    snap_spans,
)
from ._tokenizer import Token, Tokenizer
from ._utf16 import UTF16Text


class _Head(Protocol):
    def run(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray: ...


class _Tokenizer(Protocol):
    bos_id: int
    eos_id: int

    def tokenize(self, text: str) -> list[Token]: ...


SEQ = 256
MAX_CONTENT = SEQ - 2
STRIDE = 64
STEP = MAX_CONTENT - STRIDE
LOW_SCORE = 0.3


def reconstruct_offsets(
    text: UTF16Text, tokens: Sequence[Token]
) -> list[tuple[int, int]]:
    """Map token surfaces to UTF-16 source ranges by scanning forward."""
    cursor = 0
    offsets: list[tuple[int, int]] = []
    for token in tokens:
        scalars = list(token.scalars)
        if scalars and scalars[0] == "\u2581":
            scalars.pop(0)
        if not scalars:
            offsets.append((cursor, cursor))
            continue
        surface = "".join(scalars)
        found = text.find(surface, cursor)
        if found is None:
            offsets.append((cursor, cursor))
        else:
            offsets.append(found)
            cursor = found[1]
    return offsets


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=-1, keepdims=True)


def _run_window(
    head: _Head,
    token_ids: Sequence[int],
    token_offsets: Sequence[tuple[int, int]],
    bos_id: int,
    eos_id: int,
    id2label: Mapping[int, str],
) -> tuple[list[str], list[tuple[int, int]], list[float]]:
    ids = [bos_id, *token_ids, eos_id]
    real_length = len(ids)
    padded_ids = np.ones((1, SEQ), dtype=np.int32)
    attention = np.zeros((1, SEQ), dtype=np.int32)
    padded_ids[0, :real_length] = ids
    attention[0, :real_length] = 1
    logits = head.run(padded_ids, attention)[0, :real_length]
    probabilities = _softmax(logits)
    label_ids = np.argmax(probabilities, axis=-1)
    scores = probabilities[np.arange(real_length), label_ids]
    tags = [id2label.get(int(label_id), "O") for label_id in label_ids]
    offsets = [(0, 0), *token_offsets, (0, 0)]
    return tags, offsets, [float(score) for score in scores]


def ml_spans(
    text: str,
    tokenizer: _Tokenizer,
    head: _Head,
    id2label: Mapping[int, str],
    min_score: float,
) -> list[Span]:
    """Infer and post-process spans over overlapping 256-token windows."""
    view = UTF16Text(text)
    tokens = tokenizer.tokenize(text)
    offsets = reconstruct_offsets(view, tokens)
    low = min(LOW_SCORE, min_score)
    best: dict[tuple[int, int, str], int] = {}
    scored: list[tuple[Span, float]] = []
    index = 0
    while tokens:
        end = min(index + MAX_CONTENT, len(tokens))
        chunk = tokens[index:end]
        tags, tag_offsets, probabilities = _run_window(
            head,
            [token.id for token in chunk],
            offsets[index:end],
            tokenizer.bos_id,
            tokenizer.eos_id,
            id2label,
        )
        usable = [
            tag if probability >= low else "O"
            for tag, probability in zip(tags, probabilities, strict=True)
        ]
        for span in bioes_to_spans(usable, tag_offsets):
            score = max(
                (
                    probability
                    for (start, end_offset), probability in zip(
                        tag_offsets, probabilities, strict=True
                    )
                    if end_offset > start
                    and max(start, span.start) < min(end_offset, span.end)
                ),
                default=0.0,
            )
            key = (span.start, span.end, span.label)
            previous = best.get(key)
            if previous is None:
                best[key] = len(scored)
                scored.append((span, score))
            elif score > scored[previous][1]:
                scored[previous] = (scored[previous][0], score)
        if end == len(tokens):
            break
        index += STEP

    kept = hysteresis(view, scored, min_score)
    kept = merge_priority(kept)
    kept = attach_building_numbers(
        view,
        extend_particle_names(
            view,
            bridge_name_gaps(view, snap_spans(view, kept)),
        ),
    )
    kept = redact_secondary_address(
        view,
        attach_state_codes(view, redact_us_street(view, kept)),
    )
    return merge_priority(kept)


class _Detector:
    def __init__(self, assets: Assets) -> None:
        import json

        labels_data = json.loads(assets.labels.read_text(encoding="utf-8"))
        self._labels = {
            int(identifier): label
            for identifier, label in labels_data["id2label"].items()
        }
        self._tokenizer = Tokenizer(assets.tokenizer.read_bytes())
        self._head = Head(assets.tflite)

    def detect(self, text: str, minimum_confidence: float) -> list[Span]:
        deterministic = deterministic_detect(
            text,
            enabled=OWNED | {"PHONE", "GOVERNMENT_ID", "PASSPORT", "DRIVERS_LICENSE"},
        )
        deterministic_pipeline = [
            Span(span.start, span.end, span.label, span.score) for span in deterministic
        ]
        masked = mask_text(text, deterministic_pipeline)
        neural = ml_spans(
            masked,
            self._tokenizer,
            self._head,
            self._labels,
            minimum_confidence,
        )
        return clean_spans(
            text,
            relabel_by_context(text, resolve(deterministic_pipeline, neural)),
        )


class Redact:
    """Detect and redact personal information with the Redact model."""

    def __init__(self, *, directory: str | Path | None = None) -> None:
        self._directory = Path(directory) if directory is not None else None
        self._detector: _Detector | None = None

    def _load_detector(self) -> _Detector:
        if self._detector is None:
            if self._directory is None:
                assets = resolve_assets()
            else:
                assets = resolve_assets(self._directory)
            self._detector = _Detector(assets)
        return self._detector

    def _spans(self, text: str, minimum_confidence: float) -> list[Span]:
        return self._load_detector().detect(text, minimum_confidence)

    def redaction(self, text: str, options: Options | None = None) -> Redaction:
        chosen = options or Options()
        allowed = (
            chosen.labels if chosen.labels is not None else Label.default_enabled()
        )
        spans = [
            span
            for span in self._spans(text, chosen.minimum_confidence)
            if span.label in {label.value for label in allowed}
        ]
        spans.sort(key=lambda span: (span.start, span.end))
        units = list(text.encode("utf-16-le"))
        unit_count = len(units) // 2
        output = bytearray()
        items: list[Item] = []
        counts: dict[Label, int] = {}
        last = 0
        for span in spans:
            start, end = span.start, span.end
            if start < last or start >= end or end > unit_count:
                continue
            label = Label(span.label)
            number = counts.get(label, 0) + 1
            counts[label] = number
            placeholder = f"[{label.value}_{number}]"
            original = bytes(units[start * 2 : end * 2]).decode(
                "utf-16-le", errors="replace"
            )
            output.extend(units[last * 2 : start * 2])
            output.extend(placeholder.encode("utf-16-le"))
            items.append(Item(label, original, placeholder, span.score, start, end))
            last = end
        output.extend(units[last * 2 :])
        return Redaction(
            output.decode("utf-16-le", errors="replace"),
            items,
        )


class Label(str, Enum):
    """A stable Redact entity category."""

    GIVEN_NAME = "GIVEN_NAME"
    SURNAME = "SURNAME"
    STREET_NAME = "STREET_NAME"
    BUILDING_NUMBER = "BUILDING_NUMBER"
    SECONDARY_ADDRESS = "SECONDARY_ADDRESS"
    CITY = "CITY"
    STATE = "STATE"
    ZIP_CODE = "ZIP_CODE"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    CREDIT_CARD = "CREDIT_CARD"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    ROUTING_NUMBER = "ROUTING_NUMBER"
    IP_ADDRESS = "IP_ADDRESS"
    URL = "URL"
    GOVERNMENT_ID = "GOVERNMENT_ID"
    PASSPORT = "PASSPORT"
    DRIVERS_LICENSE = "DRIVERS_LICENSE"
    TAX_ID = "TAX_ID"
    SSN = "SSN"
    IMEI = "IMEI"
    ORG = "ORG"

    @property
    def display_name(self) -> str:
        return {
            "GIVEN_NAME": "Given name",
            "SURNAME": "Surname",
            "STREET_NAME": "Street",
            "BUILDING_NUMBER": "Building number",
            "SECONDARY_ADDRESS": "Unit / apartment",
            "CITY": "City",
            "STATE": "State / region",
            "ZIP_CODE": "Postal code",
            "EMAIL": "Email",
            "PHONE": "Phone",
            "CREDIT_CARD": "Credit card",
            "BANK_ACCOUNT": "Bank account",
            "ROUTING_NUMBER": "Routing number",
            "IP_ADDRESS": "IP address",
            "URL": "URL",
            "GOVERNMENT_ID": "Government ID",
            "PASSPORT": "Passport",
            "DRIVERS_LICENSE": "Driver's license",
            "TAX_ID": "Tax ID",
            "SSN": "SSN",
            "IMEI": "IMEI",
            "ORG": "Organisation",
        }[self.value]

    @classmethod
    def default_enabled(cls) -> frozenset[Label]:
        return frozenset(label for label in cls if label is not cls.ORG)


@dataclass(frozen=True)
class Item:
    label: Label
    original: str
    placeholder: str
    confidence: float
    start: int
    end: int


@dataclass(frozen=True)
class Redaction:
    redacted_text: str
    items: list[Item]

    def restore(self, processed: str) -> str:
        result = processed
        for item in self.items:
            result = result.replace(item.placeholder, item.original)
        return result


@dataclass
class Options:
    minimum_confidence: float = 0.6
    labels: frozenset[Label] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.minimum_confidence):
            self.minimum_confidence = 0.6
        else:
            self.minimum_confidence = min(1.0, max(0.0, self.minimum_confidence))


__all__ = [
    "Head",
    "Item",
    "Label",
    "Options",
    "Redact",
    "Redaction",
    "ml_spans",
    "reconstruct_offsets",
]
