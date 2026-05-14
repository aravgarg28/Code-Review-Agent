"""FastAPI application scaffold with lifespan-managed ONNX engine."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI

from backend.inference.engine import EngineConfig, ONNXReviewEngine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize the ONNX engine on startup, clean up on shutdown."""
    model_dir = Path(os.environ.get("MODEL_DIR", "outputs/onnx/quantized"))

    label_names_raw = os.environ.get("LABEL_NAMES", "")
    label_names = (
        [s.strip() for s in label_names_raw.split(",") if s.strip()]
        if label_names_raw
        else ["clean", "performance", "security", "error_handling", "style", "logic"]
    )

    config = EngineConfig(
        model_dir=model_dir,
        max_length=int(os.environ.get("MAX_LENGTH", "512")),
        label_names=label_names,
    )

    engine = ONNXReviewEngine(config)
    app.state.engine = engine

    logger.info("app_startup", extra={"model_dir": str(model_dir)})
    yield
    logger.info("app_shutdown")


app = FastAPI(
    title="Code Review Agent",
    version="0.1.0",
    lifespan=lifespan,
)


from backend.api.routes import router  # noqa: E402

app.include_router(router, prefix="/api/v1")
