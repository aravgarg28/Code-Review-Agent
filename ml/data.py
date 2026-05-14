"""PyTorch Dataset for parsing GitHub PR diffs into tokenized sequences."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


logger = logging.getLogger(__name__)

_HUNK_HEADER_RE = re.compile(r"^@@\s")
_FILE_HEADER_OLD_RE = re.compile(r"^--- ")
_FILE_HEADER_NEW_RE = re.compile(r"^\+\+\+ ")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DatasetLoadError(OSError):
    """Raised when the JSONL data file cannot be read or parsed."""


class DiffParseError(ValueError):
    """Raised when a unified diff string is structurally invalid."""


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class DiffRecord(BaseModel):
    """A single labeled PR diff sample as stored in the JSONL training data."""

    diff: str
    label: int

    @field_validator("label")
    @classmethod
    def label_must_be_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"label must be >= 0, got {v}")
        return v

    @field_validator("diff")
    @classmethod
    def diff_must_be_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("diff must not be empty")
        return v


class DatasetConfig(BaseModel):
    """Configuration for PRDiffDataset."""

    data_path: Path
    tokenizer_name: str
    max_length: int = Field(default=512, gt=0)
    seed: int = Field(default=42)
    num_labels: int = Field(default=6, gt=1)

    @field_validator("data_path")
    @classmethod
    def path_must_be_jsonl(cls, v: Path) -> Path:
        if v.suffix not in (".jsonl", ".json"):
            raise ValueError(f"data_path must be a .jsonl or .json file, got {v.suffix}")
        return v


# ---------------------------------------------------------------------------
# Diff parsing
# ---------------------------------------------------------------------------


def parse_diff_to_text(diff: str) -> str:
    """Convert a unified diff string into a structured text representation.

    Keeps hunk headers, context lines, and changed lines with directional
    markers ([ADD] / [DEL]) so the model can distinguish additions from
    removals while retaining surrounding scope.

    Raises:
        DiffParseError: If the diff contains no parseable hunks.
    """
    lines = diff.splitlines()
    output_parts: list[str] = []

    for line in lines:
        if _FILE_HEADER_OLD_RE.match(line) or _FILE_HEADER_NEW_RE.match(line):
            continue

        if _HUNK_HEADER_RE.match(line):
            output_parts.append(line)
        elif line.startswith("+"):
            output_parts.append(f"[ADD] {line[1:]}")
        elif line.startswith("-"):
            output_parts.append(f"[DEL] {line[1:]}")
        elif line.startswith(" "):
            output_parts.append(f"[CTX] {line[1:]}")

    result = "\n".join(output_parts)

    if not result.strip():
        raise DiffParseError("Diff produced no parseable hunks or changes")

    return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PRDiffDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch Dataset that reads labeled PR diffs from a JSONL file.

    Each sample is tokenized on access via the configured HuggingFace
    tokenizer and returned as a dict of tensors ready for a Trainer.
    """

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__()

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self._config = config
        self._records = self._load_records(config.data_path)
        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            config.tokenizer_name
        )

        logger.info(
            "dataset_loaded",
            extra={
                "num_samples": len(self._records),
                "tokenizer": config.tokenizer_name,
                "max_length": config.max_length,
                "num_labels": config.num_labels,
            },
        )

    # -- public interface ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self._records[idx]

        text = parse_diff_to_text(record.diff)

        encoding: dict[str, Any] = self._tokenizer(
            text,
            truncation=True,
            max_length=self._config.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(record.label, dtype=torch.long),
        }

    # -- private helpers ----------------------------------------------------

    @staticmethod
    def _load_records(data_path: Path) -> list[DiffRecord]:
        if not data_path.exists():
            raise DatasetLoadError(f"Data file not found: {data_path}")

        records: list[DiffRecord] = []
        try:
            with data_path.open("r", encoding="utf-8") as fh:
                for line_num, line in enumerate(fh, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        raw = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise DatasetLoadError(
                            f"Invalid JSON on line {line_num} of {data_path}: {exc}"
                        ) from exc
                    records.append(DiffRecord.model_validate(raw))
        except OSError as exc:
            raise DatasetLoadError(f"Cannot read {data_path}: {exc}") from exc

        if not records:
            raise DatasetLoadError(f"No valid records found in {data_path}")

        logger.info(
            "records_loaded",
            extra={"count": len(records), "path": str(data_path)},
        )
        return records
