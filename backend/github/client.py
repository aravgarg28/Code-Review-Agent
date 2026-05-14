"""GitHub REST API client for fetching PR diffs and posting review comments."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_GITHUB_API_BASE = "https://api.github.com"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GitHubClientError(RuntimeError):
    """Raised when a GitHub API call fails."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class GitHubClient:
    """Async wrapper around the GitHub REST API for PR operations."""

    def __init__(self, token: str) -> None:
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def get_pr_diff_files(
        self, repo: str, pr_number: int
    ) -> list[dict[str, Any]]:
        """Fetch the list of changed files (with patches) for a PR."""
        url = f"{_GITHUB_API_BASE}/repos/{repo}/pulls/{pr_number}/files"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self._headers)

        if response.status_code != 200:
            raise GitHubClientError(
                f"GET {url} returned {response.status_code}: {response.text}"
            )

        files: list[dict[str, Any]] = response.json()

        logger.info(
            "pr_files_fetched",
            extra={"repo": repo, "pr": pr_number, "file_count": len(files)},
        )
        return files

    async def post_review_comments(
        self,
        repo: str,
        pr_number: int,
        comments: list[dict[str, Any]],
    ) -> None:
        """Post inline review comments on a PR via the pulls reviews API."""
        url = f"{_GITHUB_API_BASE}/repos/{repo}/pulls/{pr_number}/reviews"

        body = {
            "event": "COMMENT",
            "comments": comments,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=self._headers, json=body)

        if response.status_code not in (200, 201):
            raise GitHubClientError(
                f"POST {url} returned {response.status_code}: {response.text}"
            )

        logger.info(
            "review_posted",
            extra={"repo": repo, "pr": pr_number, "comment_count": len(comments)},
        )
