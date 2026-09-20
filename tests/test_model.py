"""Tests for Phase 6 token windowing."""

from types import SimpleNamespace

import numpy as np

from piitag._pipeline import Span
from piitag._tokenizer import Token
from piitag._utf16 import UTF16Text
from piitag.model import ml_spans, reconstruct_offsets


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
