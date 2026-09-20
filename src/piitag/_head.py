"""LiteRT/TFLite inference wrapper for the Redact token classifier."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import numpy as np


def _load_interpreter(model_path: Path) -> object:
    """Create an interpreter using LiteRT, with TensorFlow as a fallback."""
    errors: list[Exception] = []
    for module_name in ("ai_edge_litert.interpreter", "tensorflow.lite"):
        try:
            module: ModuleType = importlib.import_module(module_name)
            interpreter_type = module.Interpreter
            return interpreter_type(model_path=str(model_path))
        except (ImportError, AttributeError, OSError, TypeError) as error:
            errors.append(error)
    details = "; ".join(str(error) for error in errors)
    raise RuntimeError(
        "No supported TFLite runtime is installed (tried ai-edge-litert and tensorflow)"
        + (f": {details}" if details else "")
    )


class Head:
    """Run the fixed ``(1, 256)`` Redact token-classification graph."""

    _shape = (1, 256)

    def __init__(self, tflite_path: str | Path) -> None:
        path = Path(tflite_path)
        if not path.is_file():
            raise FileNotFoundError(f"TFLite model does not exist: {path}")
        self._interpreter = _load_interpreter(path)
        self._interpreter.allocate_tensors()  # type: ignore[attr-defined]
        inputs = self._interpreter.get_input_details()  # type: ignore[attr-defined]
        outputs = self._interpreter.get_output_details()  # type: ignore[attr-defined]
        if len(inputs) < 2:
            raise ValueError(
                "Redact TFLite model must expose input_ids and attention_mask"
            )
        if not outputs:
            raise ValueError("Redact TFLite model has no output tensor")
        self._input_ids = self._find_input(inputs, "input_ids")
        self._attention_mask = self._find_input(inputs, "attention_mask")
        self._output = outputs[0]

    @staticmethod
    def _find_input(
        details: list[dict[str, object]], required: str
    ) -> dict[str, object]:
        for detail in details:
            name = str(detail.get("name", "")).lower()
            if required in name:
                return detail
        raise ValueError(f"Redact TFLite model is missing {required!r} input")

    @classmethod
    def _validate_input(cls, value: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(value)
        if array.shape != cls._shape:
            raise ValueError(f"{name} must have shape {cls._shape}, got {array.shape}")
        if array.dtype != np.int32:
            raise TypeError(f"{name} must have dtype int32, got {array.dtype}")
        return array

    def run(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Return logits with shape ``(1, 256, num_labels)``."""
        ids = self._validate_input(input_ids, "input_ids")
        mask = self._validate_input(attention_mask, "attention_mask")
        interpreter = self._interpreter
        interpreter.set_tensor(self._input_ids["index"], ids)  # type: ignore[attr-defined]
        interpreter.set_tensor(self._attention_mask["index"], mask)  # type: ignore[attr-defined]
        interpreter.invoke()  # type: ignore[attr-defined]
        logits = np.asarray(interpreter.get_tensor(self._output["index"]))  # type: ignore[attr-defined]
        if logits.ndim != 3 or logits.shape[:2] != self._shape:
            raise ValueError(
                f"model output must have shape (1, 256, labels), got {logits.shape}"
            )
        if not np.issubdtype(logits.dtype, np.floating):
            raise TypeError(f"model output must be floating point, got {logits.dtype}")
        return logits.astype(np.float32, copy=False)
