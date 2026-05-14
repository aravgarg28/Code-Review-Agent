"""Ingest a local Git repository into pgvector for RAG retrieval."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import AsyncIterator

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sqlalchemy import delete, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import RepositoryContext
from backend.db.session import async_session_factory

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".py", ".md"})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class IngestionError(RuntimeError):
    """Raised when repository ingestion fails."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class IngestConfig(BaseModel):
    """Configuration for repository ingestion."""

    repo_path: Path
    repo_name: str
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)
    batch_size: int = Field(default=32, gt=0)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def discover_files(repo_path: Path) -> list[Path]:
    """Walk the repo and return paths to supported source files."""
    if not repo_path.is_dir():
        raise IngestionError(f"Repository path is not a directory: {repo_path}")

    files: list[Path] = []
    for path in repo_path.rglob("*"):
        if path.is_file() and path.suffix in SUPPORTED_EXTENSIONS:
            if ".git" not in path.parts and "node_modules" not in path.parts:
                files.append(path)

    logger.info("files_discovered", extra={"count": len(files), "repo": str(repo_path)})
    return files


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_file(path: Path, repo_path: Path, config: IngestConfig) -> list[dict[str, str]]:
    """Read a file and split it into overlapping text chunks."""
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("file_read_error", extra={"path": str(path), "error": str(exc)})
        return []

    if not content.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        strip_whitespace=True,
    )
    chunks = splitter.split_text(content)

    relative_path = str(path.relative_to(repo_path))
    return [
        {"file_path": relative_path, "chunk_content": chunk}
        for chunk in chunks
    ]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def embed_chunks(
    chunks: list[dict[str, str]], model: SentenceTransformer
) -> list[dict[str, str | list[float]]]:
    """Add embedding vectors to each chunk dict."""
    texts = [c["chunk_content"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    result: list[dict[str, str | list[float]]] = []
    for chunk, vec in zip(chunks, vectors):
        result.append({**chunk, "embedding": vec.tolist()})
    return result


# ---------------------------------------------------------------------------
# Database upsert
# ---------------------------------------------------------------------------


async def upsert_chunks(
    session: AsyncSession,
    repo_name: str,
    chunks: list[dict[str, str | list[float]]],
) -> int:
    """Insert or update chunks in the repository_context table."""
    if not chunks:
        return 0

    rows = [
        {
            "repo_name": repo_name,
            "file_path": c["file_path"],
            "chunk_content": c["chunk_content"],
            "embedding": c["embedding"],
        }
        for c in chunks
    ]

    stmt = insert(RepositoryContext).values(rows)
    stmt = stmt.on_conflict_do_nothing()
    await session.execute(stmt)
    await session.commit()

    return len(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def ingest(config: IngestConfig) -> int:
    """Run the full ingestion pipeline. Returns total chunks upserted."""
    files = discover_files(config.repo_path)
    if not files:
        raise IngestionError(f"No supported files found in {config.repo_path}")

    model = SentenceTransformer(config.embedding_model)

    all_chunks: list[dict[str, str]] = []
    for f in files:
        all_chunks.extend(chunk_file(f, config.repo_path, config))

    logger.info("chunking_complete", extra={"total_chunks": len(all_chunks)})

    total_upserted = 0

    async with async_session_factory() as session:
        await session.execute(
            delete(RepositoryContext).where(RepositoryContext.repo_name == config.repo_name)
        )
        await session.commit()

        for i in range(0, len(all_chunks), config.batch_size):
            batch = all_chunks[i : i + config.batch_size]
            embedded = embed_chunks(batch, model)
            count = await upsert_chunks(session, config.repo_name, embedded)
            total_upserted += count

    logger.info(
        "ingestion_complete",
        extra={"repo": config.repo_name, "total_upserted": total_upserted},
    )
    return total_upserted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def _async_main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m backend.rag.ingest <config.json>", file=sys.stderr)
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        raise IngestionError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    config = IngestConfig.model_validate(raw)
    total = await ingest(config)
    print(f"Ingested {total} chunks for {config.repo_name}")


def main() -> None:
    import asyncio
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
