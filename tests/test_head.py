"""Tests for the TFLite inference wrapper."""

from pathlib import Path

import numpy as np
import pytest

import piitag._head as head_module
from piitag._head import Head


class FakeInterpreter:
    def __init__(self, *, model_path: str) -> None:
        self.model_path = model_path
        self.ids: np.ndarray | None = None

    def allocate_tensors(self) -> None:
        return None

    def get_input_details(self) -> list[dict[str, object]]:
        return [
            {"name": "serving_default_attention_mask:0", "index": 3},
            {"name": "serving_default_input_ids:0", "index": 7},
        ]

    def get_output_details(self) -> list[dict[str, object]]:
        return [{"name": "StatefulPartitionedCall:0", "index": 9}]

    def set_tensor(self, index: object, value: np.ndarray) -> None:
        if index == 7:
            self.ids = value.copy()

    def invoke(self) -> None:
        return None

    def get_tensor(self, index: object) -> np.ndarray:
        assert index == 9
        return np.zeros((1, 256, 3), dtype=np.float32)


def test_head_discovers_tensor_names_and_returns_logits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = tmp_path / "redact.tflite"
    model.touch()
    interpreter = FakeInterpreter(model_path=str(model))
    monkeypatch.setattr(head_module, "_load_interpreter", lambda _: interpreter)

    result = Head(model).run(
        np.zeros((1, 256), dtype=np.int32),
        np.ones((1, 256), dtype=np.int32),
    )

    assert result.shape == (1, 256, 3)
    assert result.dtype == np.float32


def test_head_rejects_wrong_tensor_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = tmp_path / "redact.tflite"
    model.touch()
    monkeypatch.setattr(
        head_module,
        "_load_interpreter",
        lambda _: FakeInterpreter(model_path=str(model)),
    )
    head = Head(model)

    with pytest.raises(ValueError, match="shape"):
        head.run(np.zeros((1, 2), dtype=np.int32), np.ones((1, 256), dtype=np.int32))
