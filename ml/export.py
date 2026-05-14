"""Export a fine-tuned PEFT checkpoint to ONNX with int8 dynamic quantization."""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import torch
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ExportError(RuntimeError):
    """Raised when ONNX export or quantization fails."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ExportConfig(BaseModel):
    """Configuration for the ONNX export pipeline."""

    checkpoint_dir: Path
    output_dir: Path = Field(default=Path("outputs/onnx"))
    opset: int = Field(default=14, ge=9)
    quantize: bool = Field(default=True)
    num_labels: int = Field(default=6, gt=1)


# ---------------------------------------------------------------------------
# Export pipeline
# ---------------------------------------------------------------------------


def merge_peft_weights(checkpoint_dir: Path, num_labels: int) -> Path:
    """Merge LoRA adapters into the base model and save a standalone checkpoint."""
    merged_dir = checkpoint_dir / "merged"
    if merged_dir.exists():
        shutil.rmtree(merged_dir)

    logger.info("merging_lora", extra={"checkpoint": str(checkpoint_dir)})

    base_model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir,
        num_labels=num_labels,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model = model.merge_and_unload()

    model.save_pretrained(merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(merged_dir)

    logger.info("lora_merged", extra={"merged_dir": str(merged_dir)})
    return merged_dir


def export_to_onnx(merged_dir: Path, output_dir: Path, opset: int) -> Path:
    """Convert a merged PyTorch model to ONNX format via Optimum."""
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "onnx_export_start",
        extra={"source": str(merged_dir), "target": str(onnx_dir), "opset": opset},
    )

    try:
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            merged_dir, export=True
        )
        ort_model.save_pretrained(onnx_dir)
        AutoTokenizer.from_pretrained(merged_dir).save_pretrained(onnx_dir)
    except Exception as exc:
        raise ExportError(f"ONNX export failed: {exc}") from exc

    logger.info("onnx_export_complete", extra={"onnx_dir": str(onnx_dir)})
    return onnx_dir


def quantize_onnx(onnx_dir: Path, output_dir: Path) -> Path:
    """Apply int8 dynamic quantization to the ONNX model."""
    quantized_dir = output_dir / "quantized"
    quantized_dir.mkdir(parents=True, exist_ok=True)

    logger.info("quantization_start", extra={"source": str(onnx_dir)})

    try:
        quantizer = ORTQuantizer.from_pretrained(onnx_dir)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        quantizer.quantize(save_dir=quantized_dir, quantization_config=qconfig)
        AutoTokenizer.from_pretrained(onnx_dir).save_pretrained(quantized_dir)
    except Exception as exc:
        raise ExportError(f"Quantization failed: {exc}") from exc

    logger.info("quantization_complete", extra={"quantized_dir": str(quantized_dir)})
    return quantized_dir


def export(config: ExportConfig) -> Path:
    """Run the full export pipeline: merge → ONNX → quantize."""
    torch.manual_seed(42)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    merged_dir = merge_peft_weights(config.checkpoint_dir, config.num_labels)
    onnx_dir = export_to_onnx(merged_dir, config.output_dir, config.opset)

    if config.quantize:
        final_dir = quantize_onnx(onnx_dir, config.output_dir)
    else:
        final_dir = onnx_dir

    logger.info("export_pipeline_complete", extra={"final_dir": str(final_dir)})
    return final_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Load a JSON config and run the export pipeline."""
    if len(sys.argv) != 2:
        print("Usage: python -m ml.export <config.json>", file=sys.stderr)
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        raise ExportError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    config = ExportConfig.model_validate(raw)
    final = export(config)
    print(f"Export complete: {final}")


if __name__ == "__main__":
    main()
