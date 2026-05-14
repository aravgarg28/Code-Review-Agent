"""API routes for the code review agent."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.github.client import GitHubClient, GitHubClientError
from backend.inference.engine import InferenceError, ONNXReviewEngine, PredictionResult
from ml.data import DiffParseError

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ReviewPRRequest(BaseModel):
    """Payload for POST /api/v1/review/pr."""

    repository: str
    pull_request_number: int = Field(gt=0)
    github_token: str


class ReviewComment(BaseModel):
    """A single inline review comment to post."""

    file_path: str
    line: int
    body: str
    label: str
    confidence: float


class ReviewPRResponse(BaseModel):
    """Response from POST /api/v1/review/pr."""

    status: str
    comments_posted: int
    comments: list[ReviewComment]
    inference_time_ms: float


class ReviewDiffRequest(BaseModel):
    """Payload for POST /api/v1/review/diff — direct diff review without GitHub."""

    diff: str


class ReviewDiffResponse(BaseModel):
    """Response from POST /api/v1/review/diff."""

    prediction: PredictionResult
    inference_time_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/review/pr", response_model=ReviewPRResponse)
async def review_pr(request: Request, payload: ReviewPRRequest) -> ReviewPRResponse:
    """Fetch a PR diff from GitHub, run inference, and post inline comments."""
    engine: ONNXReviewEngine = request.app.state.engine

    try:
        gh = GitHubClient(token=payload.github_token)
        files = await gh.get_pr_diff_files(payload.repository, payload.pull_request_number)
    except GitHubClientError as exc:
        raise HTTPException(status_code=502, detail=f"GitHub API error: {exc}") from exc

    comments: list[ReviewComment] = []
    start = time.perf_counter()

    for file_info in files:
        patch = file_info.get("patch", "")
        if not patch:
            continue

        try:
            result = engine.predict(patch)
        except DiffParseError:
            continue
        except InferenceError as exc:
            logger.error("inference_error", extra={"file": file_info["filename"], "error": str(exc)})
            continue

        if result.predicted_label == "clean":
            continue

        comments.append(
            ReviewComment(
                file_path=file_info["filename"],
                line=file_info.get("changes", 1),
                body=(
                    f"**[{result.predicted_label.upper()}]** "
                    f"(confidence: {result.confidence:.0%})\n\n"
                    f"Potential anti-pattern detected in this change."
                ),
                label=result.predicted_label,
                confidence=result.confidence,
            )
        )

    elapsed_ms = (time.perf_counter() - start) * 1000

    if comments:
        try:
            await gh.post_review_comments(
                repo=payload.repository,
                pr_number=payload.pull_request_number,
                comments=[
                    {
                        "path": c.file_path,
                        "line": c.line,
                        "body": c.body,
                    }
                    for c in comments
                ],
            )
        except GitHubClientError as exc:
            logger.error("comment_post_error", extra={"error": str(exc)})
            raise HTTPException(
                status_code=502, detail=f"Failed to post comments: {exc}"
            ) from exc

    logger.info(
        "review_complete",
        extra={
            "repo": payload.repository,
            "pr": payload.pull_request_number,
            "comments": len(comments),
            "inference_ms": f"{elapsed_ms:.1f}",
        },
    )

    return ReviewPRResponse(
        status="success",
        comments_posted=len(comments),
        comments=comments,
        inference_time_ms=round(elapsed_ms, 1),
    )


@router.post("/review/diff", response_model=ReviewDiffResponse)
async def review_diff(request: Request, payload: ReviewDiffRequest) -> ReviewDiffResponse:
    """Run inference on a raw unified diff string (no GitHub interaction)."""
    engine: ONNXReviewEngine = request.app.state.engine

    start = time.perf_counter()

    try:
        result = engine.predict(payload.diff)
    except DiffParseError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid diff: {exc}") from exc
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    elapsed_ms = (time.perf_counter() - start) * 1000

    return ReviewDiffResponse(
        prediction=result,
        inference_time_ms=round(elapsed_ms, 1),
    )
