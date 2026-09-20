"""Windowed neural inference for the Redact model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np

from ._head import Head
from ._pipeline import (
    Span,
    attach_building_numbers,
    attach_state_codes,
    bioes_to_spans,
    bridge_name_gaps,
    extend_particle_names,
    hysteresis,
    merge_priority,
    redact_secondary_address,
    redact_us_street,
    snap_spans,
)
from ._tokenizer import Token, Tokenizer
from ._utf16 import UTF16Text


class _Head(Protocol):
    def run(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray: ...


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
    tokenizer: Tokenizer,
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


__all__ = ["Head", "ml_spans", "reconstruct_offsets"]
