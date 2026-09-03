"""FastAPI entry point for real-time Porto TrajAgg retrieval."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .trajagg_runtime import TrajAggRuntime


DEFAULT_ROOT = Path("/home/devuser/mnt/mydisk2/lixinyi/TrajAgg")


def environment_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def cors_origins() -> List[str]:
    configured = os.getenv("TRAJAGG_CORS_ORIGINS", "")
    if configured.strip():
        return [value.strip().rstrip("/") for value in configured.split(",") if value.strip()]
    return [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "https://tgf137650-max.github.io",
    ]


def build_runtime() -> TrajAggRuntime:
    source_dir = Path(os.getenv(
        "TRAJAGG_SOURCE_DIR", str(DEFAULT_ROOT / "source/TrajAgg")
    ))
    artifact_dir = Path(os.getenv(
        "TRAJAGG_ARTIFACT_DIR",
        str(DEFAULT_ROOT / "artifacts/porto_haus_hybrid_export_200"),
    ))
    return TrajAggRuntime(
        source_dir=source_dir,
        artifact_dir=artifact_dir,
        gpu_index=int(os.getenv("TRAJAGG_GPU", "0")),
        require_cuda=environment_flag("TRAJAGG_REQUIRE_CUDA", True),
        load_ground_truth=environment_flag("TRAJAGG_LOAD_GROUND_TRUTH", True),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime = build_runtime()
    yield
    del app.state.runtime


app = FastAPI(
    title="TrajAgg Explorer API",
    version="1.0.0",
    description=(
        "Real-time inference over the saved Porto–Hausdorff TrajAgg checkpoint "
        "and its 7,000-trajectory test embedding library."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class RetrievalRequest(BaseModel):
    queryId: str = Field(
        ...,
        examples=["Q-04887"],
        description="Porto test trajectory ID from Q-03000 through Q-09999",
    )
    topK: int = Field(3, description="The academic demo supports Top-1 or Top-3")
    includeGroundTruth: bool = Field(
        True,
        description="Attach precomputed Hausdorff distance and rank for explanation",
    )


def runtime_from(request: Request) -> TrajAggRuntime:
    return request.app.state.runtime


@app.get("/")
def api_root():
    return {
        "name": "TrajAgg Explorer API",
        "status": "ready",
        "documentation": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health(request: Request):
    return runtime_from(request).health_payload()


@app.get("/api/config")
def config(request: Request):
    return runtime_from(request).config_payload()


@app.get("/api/queries")
def queries(
    request: Request,
    offset: int = Query(0, ge=0, le=6999),
    limit: int = Query(20, ge=1, le=100),
    search: str = Query("", max_length=32),
):
    return runtime_from(request).list_queries(offset=offset, limit=limit, search=search)


@app.get("/api/queries/{query_id}")
def query_preview(query_id: str, request: Request):
    try:
        return runtime_from(request).query_preview(query_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/api/retrieve")
def retrieve(payload: RetrievalRequest, request: Request):
    try:
        return runtime_from(request).retrieve(
            query_id=payload.queryId,
            top_k=payload.topK,
            include_ground_truth=payload.includeGroundTruth,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
