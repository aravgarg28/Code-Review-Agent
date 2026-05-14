"""ONNX Runtime inference engine for anti-pattern classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ml.data import parse_diff_to_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InferenceError(RuntimeError):
    """Raised when model inference fails."""


class ModelLoadError(RuntimeError):
    """Raised when the ONNX model or tokenizer cannot be loaded."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class EngineConfig(BaseModel):
    """Configuration for the ONNX inference engine."""

    model_dir: Path
    max_length: int = Field(default=512, gt=0)
    label_names: list[str] = Field(
        default=[
            "clean",
            "performance",
            "security",
            "error_handling",
            "style",
            "logic",
        ]
    )


# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------


class PredictionResult(BaseModel):
    """Structured output from a single inference call."""

    predicted_label: str
    predicted_index: int
    confidence: float
    probabilities: dict[str, float]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ONNXReviewEngine:
    """Loads a quantized ONNX model and runs anti-pattern classification."""

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._tokenizer: PreTrainedTokenizerBase = self._load_tokenizer(config.model_dir)
        self._session: ort.InferenceSession = self._load_session(config.model_dir)

        logger.info(
            "engine_initialized",
            extra={
                "model_dir": str(config.model_dir),
                "max_length": config.max_length,
                "num_labels": len(config.label_names),
            },
        )

    def predict(self, diff: str) -> PredictionResult:
        """Run inference on a single unified diff string."""
        text = parse_diff_to_text(diff)

        encoding: dict[str, Any] = self._tokenizer(
            text,
            truncation=True,
            max_length=self._config.max_length,
            padding="max_length",
            return_tensors="np",
        )

        input_feed: dict[str, np.ndarray] = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

        try:
            outputs = self._session.run(None, input_feed)
        except ort.capi.onnxruntime_pybind11_state.RuntimeException as exc:
            raise InferenceError(f"ONNX inference failed: {exc}") from exc

        logits: np.ndarray = outputs[0]
        probabilities = self._softmax(logits[0])

        predicted_index = int(np.argmax(probabilities))
        label_names = self._config.label_names

        prob_map = {
            label_names[i]: float(probabilities[i])
            for i in range(len(label_names))
        }

        return PredictionResult(
            predicted_label=label_names[predicted_index],
            predicted_index=predicted_index,
            confidence=float(probabilities[predicted_index]),
            probabilities=prob_map,
        )

    def predict_batch(self, diffs: list[str]) -> list[PredictionResult]:
        """Run inference on multiple diffs sequentially."""
        return [self.predict(d) for d in diffs]

    # -- private helpers ----------------------------------------------------

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp = np.exp(logits - np.max(logits))
        return exp / exp.sum()

    @staticmethod
    def _load_tokenizer(model_dir: Path) -> PreTrainedTokenizerBase:
        try:
            return AutoTokenizer.from_pretrained(str(model_dir))
        except (OSError, ValueError) as exc:
            raise ModelLoadError(f"Cannot load tokenizer from {model_dir}: {exc}") from exc

    @staticmethod
    def _load_session(model_dir: Path) -> ort.InferenceSession:
        onnx_path = model_dir / "model.onnx"
        if not onnx_path.exists():
            candidates = list(model_dir.glob("*.onnx"))
            if not candidates:
                raise ModelLoadError(f"No .onnx file found in {model_dir}")
            onnx_path = candidates[0]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            return ort.InferenceSession(str(onnx_path), sess_options=sess_options)
        except ort.capi.onnxruntime_pybind11_state.RuntimeException as exc:
            raise ModelLoadError(f"Cannot load ONNX model from {onnx_path}: {exc}") from exc
