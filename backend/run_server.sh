#!/usr/bin/env bash
set -euo pipefail

TRAJAGG_PROJECT_ROOT="${TRAJAGG_ROOT:-/home/devuser/mnt/mydisk2/lixinyi/TrajAgg}"
TRAJAGG_API_HOST="${TRAJAGG_API_HOST:-127.0.0.1}"
TRAJAGG_API_PORT="${TRAJAGG_API_PORT:-8000}"
TRAJAGG_VISIBLE_GPU="${TRAJAGG_VISIBLE_GPU:-0}"

export TRAJAGG_SOURCE_DIR="${TRAJAGG_SOURCE_DIR:-$TRAJAGG_PROJECT_ROOT/source/TrajAgg}"
export TRAJAGG_ARTIFACT_DIR="${TRAJAGG_ARTIFACT_DIR:-$TRAJAGG_PROJECT_ROOT/artifacts/porto_haus_hybrid_export_200}"
# CUDA_VISIBLE_DEVICES remaps the selected physical GPU to cuda:0 in this process.
export TRAJAGG_GPU=0
export TRAJAGG_REQUIRE_CUDA="${TRAJAGG_REQUIRE_CUDA:-true}"
export TRAJAGG_LOAD_GROUND_TRUTH="${TRAJAGG_LOAD_GROUND_TRUTH:-true}"
export TRAJAGG_CORS_ORIGINS="${TRAJAGG_CORS_ORIGINS:-http://127.0.0.1:5173,http://localhost:5173,https://tgf137650-max.github.io}"

cd "$TRAJAGG_PROJECT_ROOT/demo_api"
exec env CUDA_VISIBLE_DEVICES="$TRAJAGG_VISIBLE_GPU" \
  "$TRAJAGG_PROJECT_ROOT/envs/trajagg-py39/bin/python" -m uvicorn \
  backend.app:app --host "$TRAJAGG_API_HOST" --port "$TRAJAGG_API_PORT"
