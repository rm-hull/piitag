"""Tests for Phase 6 token windowing."""

from types import SimpleNamespace

import numpy as np

from piitag._pipeline import Span
from piitag._tokenizer import Token
from piitag._utf16 import UTF16Text
from piitag.model import Item, Label, Options, Redact, ml_spans, reconstruct_offsets


class FakeHead:
    def run(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        del input_ids, attention_mask
        logits = np.zeros((1, 256, 2), dtype=np.float32)
        logits[:, :, 0] = 4
        logits[:, 2, 1] = 8
        return logits


def test_reconstruct_offsets_uses_utf16_units() -> None:
    tokens = [
        Token(1, ("\u2581", "😀")),
        Token(2, ("x",)),
    ]

    assert reconstruct_offsets(UTF16Text("😀 x"), tokens) == [(0, 2), (3, 4)]


def test_ml_spans_runs_a_window_and_decodes_bioes() -> None:
    tokenizer = SimpleNamespace(
        bos_id=10,
        eos_id=11,
        tokenize=lambda _: [
            Token(1, ("\u2581", "a")),
            Token(2, ("b",)),
        ],
    )

    assert ml_spans("a b", tokenizer, FakeHead(), {0: "O", 1: "S-EMAIL"}, 0.6) == [
        Span(2, 3, "EMAIL", 1.0)
    ]


def test_public_redaction_uses_utf16_ranges_and_restores_placeholders() -> None:
    redactor = Redact(directory="unused")
    redactor._spans = lambda text, minimum_confidence: [  # type: ignore[method-assign]
        Span(0, 2, "EMAIL", 1.0),
        Span(3, 8, "EMAIL", 0.8),
    ]

    result = redactor.redaction("😀 alice", Options(labels=frozenset({Label.EMAIL})))

    assert result.redacted_text == "[EMAIL_1] [EMAIL_2]"
    assert result.items == [
        Item(Label.EMAIL, "😀", "[EMAIL_1]", 1.0, 0, 2),
        Item(Label.EMAIL, "alice", "[EMAIL_2]", 0.8, 3, 8),
    ]
    assert result.restore("x [EMAIL_2] then [EMAIL_1]") == "x alice then 😀"


def test_label_display_names_and_options_are_stable() -> None:
    assert Label.DRIVERS_LICENSE.display_name == "Driver's license"
    assert Label.ORG not in Label.default_enabled()
    assert Options(float("nan")).minimum_confidence == 0.6
    assert Options(4).minimum_confidence == 1.0
