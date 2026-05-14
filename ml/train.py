"""Fine-tuning script with PEFT/LoRA and Weights & Biases tracking."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from pydantic import BaseModel, Field
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from ml.data import DatasetConfig, PRDiffDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TrainingConfigError(ValueError):
    """Raised when the training configuration is invalid."""


# ---------------------------------------------------------------------------
# Pydantic configuration
# ---------------------------------------------------------------------------


class LoRAParams(BaseModel):
    """PEFT/LoRA hyperparameters."""

    r: int = Field(default=8, gt=0)
    lora_alpha: int = Field(default=16, gt=0)
    lora_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(default=["query", "value"])


class TrainingConfig(BaseModel):
    """Full training run configuration."""

    model_name: str
    tokenizer_name: str
    train_data: Path
    eval_data: Path
    output_dir: Path = Field(default=Path("outputs/checkpoints"))
    num_labels: int = Field(default=6, gt=1)
    max_length: int = Field(default=512, gt=0)
    seed: int = Field(default=42)

    # training hyperparameters
    epochs: int = Field(default=5, gt=0)
    train_batch_size: int = Field(default=16, gt=0)
    eval_batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=2e-5, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    warmup_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    fp16: bool = Field(default=False)
    gradient_accumulation_steps: int = Field(default=1, gt=0)

    # LoRA
    lora: LoRAParams = Field(default_factory=LoRAParams)

    # W&B
    wandb_project: str = Field(default="code-review-agent")
    wandb_run_name: str | None = Field(default=None)
    wandb_enabled: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Compute per-class and macro-averaged F1, precision, and recall."""
    logits = eval_pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)
    labels = eval_pred.label_ids

    metrics: dict[str, float] = {
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "precision_macro": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
    }

    per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
    for i, score in enumerate(per_class_f1):
        metrics[f"f1_class_{i}"] = float(score)

    return metrics


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_model(config: TrainingConfig) -> torch.nn.Module:
    """Load a pretrained classifier and wrap it with LoRA adapters."""
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
    )

    model = get_peft_model(base_model, lora_config)

    trainable, total = 0, 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    logger.info(
        "model_built",
        extra={
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": f"{100 * trainable / total:.2f}%",
        },
    )

    return model


# ---------------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------------


def train(config: TrainingConfig) -> Path:
    """Run a full fine-tuning loop. Returns the path to the best checkpoint."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    dataset_kwargs: dict[str, Any] = {
        "tokenizer_name": config.tokenizer_name,
        "max_length": config.max_length,
        "seed": config.seed,
        "num_labels": config.num_labels,
    }
    train_dataset = PRDiffDataset(DatasetConfig(data_path=config.train_data, **dataset_kwargs))
    eval_dataset = PRDiffDataset(DatasetConfig(data_path=config.eval_data, **dataset_kwargs))

    model = build_model(config)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        report_to=["wandb"] if config.wandb_enabled else ["none"],
        run_name=config.wandb_run_name,
        seed=config.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("training_start", extra={"config": config.model_dump(mode="json")})

    trainer.train()

    best_ckpt = Path(training_args.output_dir) / "best"
    trainer.save_model(str(best_ckpt))
    AutoTokenizer.from_pretrained(config.tokenizer_name).save_pretrained(str(best_ckpt))

    logger.info("training_complete", extra={"best_checkpoint": str(best_ckpt)})

    return best_ckpt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Load a JSON config file and launch training."""
    if len(sys.argv) != 2:
        print(f"Usage: python -m ml.train <config.json>", file=sys.stderr)
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        raise TrainingConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    config = TrainingConfig.model_validate(raw)

    if config.wandb_enabled:
        import wandb

        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.model_dump(mode="json"),
        )

    best = train(config)
    print(f"Best checkpoint saved to: {best}")

    if config.wandb_enabled:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
