# TrajAgg Explorer real-time API

This FastAPI service performs real TrajAgg inference for any trajectory in the
official Porto test split (`Q-03000` through `Q-09999`). It loads the saved best
checkpoint and the `7000 × 128` embedding library once at startup. Each request:

1. applies the author-compatible Mercator normalization and 100 m grid mapping;
2. encodes the selected query with the Epoch 145 checkpoint;
3. scans the 7,000 saved embeddings using Chebyshev distance;
4. returns Top-1 or Top-3, plus optional Hausdorff ground-truth ranks.

It does not retrain the model. An ID-based query is the second-stage interface;
arbitrary uploaded GPS files remain a later feature because they require input
validation and confirmation that every point lies inside the Porto cell space.

## Install on the laboratory server

Use the existing Python 3.9 TrajAgg environment:

```bash
ROOT=/home/devuser/mnt/mydisk2/lixinyi/TrajAgg
PY="$ROOT/envs/trajagg-py39/bin/python"

"$PY" -m pip install -r "$ROOT/demo_api/backend/requirements.txt"
```

## Run privately on the server

The repository includes a launcher with the fixed experiment paths:

```bash
TRAJAGG_ROOT=/home/devuser/mnt/mydisk2/lixinyi/TrajAgg \
  bash /home/devuser/mnt/mydisk2/lixinyi/TrajAgg/demo_api/backend/run_server.sh
```

The equivalent explicit configuration is:

```bash
ROOT=/home/devuser/mnt/mydisk2/lixinyi/TrajAgg
cd "$ROOT/demo_api"

export TRAJAGG_SOURCE_DIR="$ROOT/source/TrajAgg"
export TRAJAGG_ARTIFACT_DIR="$ROOT/artifacts/porto_haus_hybrid_export_200"
export TRAJAGG_GPU=0
export TRAJAGG_REQUIRE_CUDA=true
export TRAJAGG_LOAD_GROUND_TRUTH=true
export TRAJAGG_CORS_ORIGINS="http://127.0.0.1:5173,http://localhost:5173,https://tgf137650-max.github.io"

CUDA_VISIBLE_DEVICES=0 "$ROOT/envs/trajagg-py39/bin/python" -m uvicorn \
  backend.app:app --host 127.0.0.1 --port 8000
```

Binding to `127.0.0.1` keeps the research server private. Test through SSH:

```bash
curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queryId":"Q-04887","topK":3,"includeGroundTruth":true}'
```

## Local browser tunnel

Keep this command running in a separate macOS Terminal:

```bash
ssh -N -L 8000:127.0.0.1:8000 devuser@10.140.34.37
```

Then the API is available on the Mac at `http://127.0.0.1:8000`. A public
GitHub Pages site cannot call this private HTTP endpoint directly. Public live
mode therefore requires a laboratory-approved public HTTPS reverse proxy or
tunnel; the static 100-query dataset remains the safe public fallback.
