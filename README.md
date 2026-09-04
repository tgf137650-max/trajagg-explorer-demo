# TrajAgg Explorer

TrajAgg Explorer is an interactive academic demo of trajectory-similarity retrieval. The website has two paths: **Dataset Query**, with 100 validated exported cases, and **Custom Query**, with new map-drawn queries encoded directly in the browser. File upload and laboratory API connections have been removed from the website. The saved Epoch 145 model is reused; neither path trains a model. No lab connection, SSH tunnel or API configuration is required.

## Draw and retrieve (browser-only)

1. Open the local page and choose **Custom Query** in the left panel.
2. Click inside the outlined Porto study area in travel order. Drag to pan; use zoom controls as needed. With the map focused, arrow keys pan and Enter adds a point at its centre.
3. Draw 1–25 km using 2–64 control points. Undo, Clear and Edit let you revise the route. Edits invalidate old results.
4. Choose Top-1 or Top-3 and click **Run browser Top-k**. First use downloads/prepares the model, 7,000 saved vectors, candidate geometry and WASM engine (about 24 MB uncompressed). Subsequent queries reuse the in-memory model.
5. Inspect real candidate routes, distances, the query's actual embedding trace and measured device timings. Cancel stops the worker and preserves the drawing.

The polyline connects control points with straight segments: it is not a routing service or observed taxi GPS. Resampling preserves vertices/order and subdivides to 20–200 points; the author-compatible grid preprocessing must leave at least two cells. The 1 km minimum is a **demo input policy**, informed by a test-library first length percentile of about 1.08 km, not an author training requirement. The 25 km upper bound is an input/coverage safeguard, near the observed library maximum of 24.75 km, not a claim of generalization.

The new query is processed in a dedicated Web Worker using ONNX Runtime Web 1.29.0, float32 WASM CPU, one thread. Query coordinates are not sent to an inference server or added to the library. Model/data files and map tiles still need network access; the map tile provider can observe the viewed map area. No persistent query storage is introduced.

New queries have no precomputed Hausdorff row. **GT not computed** is intentional, and the fixed reproduction HR values are not accuracy estimates for drawings. Similarity `exp(-d)` is a display transform, not a probability. No RTX 3090 timing or 213.88× benchmark speedup is reused as a browser result. Browser query-compute timing includes preprocessing, encoding and exact ranking, but excludes drawing/resampling, initial download/setup, result geometry and rendering; additional timing fields are labelled separately.

The worker verifies asset sizes and SHA256 hashes before use. Its vectors and all 7,000 candidate geometries are exported from the trusted saved artifacts. No training checkpoint, pickle, credentials, private endpoint, or 525 MB distance matrix is needed by this path.

### Browser-only assets and validation

- `public/models/porto-epoch145-v1/`: ONNX, little-endian embedding/geometry/offset binaries and manifest.
- `scripts/prepare-browser-runtime.mjs`: copies the pinned runtime's matching MJS/WASM files into `public/ort/` before dev/build. These generated runtime files are ignored by Git but included in the built site.
- `src/browser/`: drawing rules, preprocessing, inference worker, cancellation/progress client and timing types.
- `tests/static-site.test.mjs`: confirms the two-mode UI has no upload/API path, validates all 100 local cases, checks asset-loading errors and scans the built app for old connection controls/endpoints.
- `tests/browser.test.mjs`: input guards, shape-preserving sampling, all 100 real-case model checks, a fresh synthetic drawing, Top-1/3, production-worker errors/retry and no-API asset loading.
- `tests/route-preview.test.mjs`: display-only Mercator fitting, grid scaling, degenerate routes, a long drawing and all 100 real query previews. Preview fitting does not modify model inputs.

Local model checks: all 100 identity queries matched the saved vectors within 1e-5 (maximum observed error 4.77e-7). Prior independent conversion checks also matched the ordered, self-excluded Top-3/10/50 in all 100 cases. These tests run the actual Web WASM engine in Node, including the built worker, not a fake model. The user reviewed and approved the local interface before publication; automated cross-browser UI testing is not implied by a build or Node test.

```bash
npm run build
npm test
npm run lint
```

Node 24 was used for direct TypeScript test loading; normal dev/build supports the declared engine range. Run the build before tests so the production-worker test uses the current bundle. A dependency audit endpoint timed out during this change; no forced dependency upgrades were applied.

## Current data status

The published cases are real Porto test-split retrieval artifacts, not synthetic interaction values.

- Source subset: 10,000 preprocessed Porto trajectories
- Retrieval library: 7,000 test-split trajectories
- Ground-truth supervision/evaluation: Hausdorff distance
- Online embedding ranking: Chebyshev distance
- Grid cell: 100 m
- Fusion coefficient: μ = 0.5
- Training mode: Hybrid (`embedding + pairwise`)
- Selected checkpoint: Epoch 145, chosen by maximum validation HR@1

The strict reproduction Test results shown in the interface are:

| Metric | Result |
| --- | ---: |
| HR@1 | 0.601143 |
| HR@5 | 0.716514 |
| HR@10 | 0.768971 |
| HR@20 | 0.809986 |
| HR@50 | 0.861586 |
| R10@50 | 0.991329 |

The interface also shows a measured per-query efficiency comparison for 100 exported cases. On an RTX 3090, the median across queries was 4.001 ms for query preprocessing + TrajAgg encoding + exhaustive Chebyshev Top-k, versus 855.068 ms for direct author `traj-dist` Hausdorff computation and sorting on one CPU process: a median 213.88× speedup. Each query used 20 warmups and 100 measured runs with CUDA synchronization; browser rendering and ground-truth matrix lookup were excluded.

## Run locally

Node.js 20.19 or newer is required.

```bash
cd ~/Desktop/demo页面
npm install
npm run dev
```

The development server normally prints `http://localhost:5173`. Run the production check with:

```bash
npm run build
```

This opens the 100-case **Dataset Query** view with **Custom Query** available. Both paths work with static hosting; no `.env.local`, laboratory service or SSH tunnel is needed. Any old `VITE_API_BASE_URL` setting is ignored by the frontend.

## What the interface shows

1. Search and page through 100 real Porto test trajectories selected with a fixed-seed, coverage-oriented strategy.
2. Compare the raw GPS/WGS84 route with its 100 m grid representation.
3. Inspect the real Chebyshev Top-1 or Top-3 from the saved 7,000-trajectory embedding library.
4. Compare measured TrajAgg retrieval time with direct Hausdorff search for the selected query.
5. View each candidate's Chebyshev distance, Hausdorff ground-truth distance, and ground-truth rank.
6. Follow the query → dual-scale encoder → embedding library → Top-k trace.

Choose **Custom Query** to create a new query and run the saved model directly on your device. **Dataset Query** replays already exported results; its measured RTX 3090 benchmark remains clearly separate from fresh browser timings.

The central route canvas places the exported WGS84 coordinates on an interactive Porto street map. It uses OpenStreetMap cartography with visible attribution, and supports pan, zoom, route tooltips, candidate highlighting, and GPS/grid visibility controls. Timestamp and duration are marked unavailable because the author-compatible preprocessed 10,000-trajectory subset retains `trajlen`, `wgs_seq`, and `merc_seq`, but not those metadata fields.

Hausdorff values are not labelled as metres. The author preprocessing calculates them from WGS84 coordinate sequences, so the demo reports them in WGS84 coordinate space.

## Static data layout

```text
public/data/
├── index.json
├── query_manifest_100.json
├── retrieval_benchmark.json
└── cases/
    ├── Q-03057.json
    ├── ...
    └── Q-09943.json  (100 real case files)
```

`index.json` records configuration, checkpoint selection, strict Test metrics, benchmark protocol, and lightweight query summaries. The browser loads this index first and fetches an individual case only when selected. `query_manifest_100.json` preserves the fixed-seed query selection and `retrieval_benchmark.json` preserves the complete per-query benchmark artifact. Each case contains the real query route, grid sequence, saved embedding preview, Top-3 candidates, Chebyshev scores, Hausdorff ground-truth ranks, and measured query timing.

## Offline artifact pipeline

The non-invasive export workflow keeps the author repository unchanged:

1. Run the official Porto / Hausdorff / Chebyshev / Hybrid training components.
2. At each author-defined best-validation event, persist `best_checkpoint.pt`, `test_embeddings.pt`, and `best_metrics.json`.
3. Use `scripts/export_trajagg_demo_data.py` to rank candidates and write the static JSON cases.

The Dataset Query path reads the exported JSON. The Custom Query path additionally loads the ONNX model and full embedding/geometry library on demand, but never PyTorch or the 525 MB distance matrix.

Earlier server scripts under `backend/` and standalone upload/API helpers/tests are retained only as historical local research material. The frontend does not import them, expose upload controls, read endpoint configuration or call their endpoints. They are not required for this site or included as executable code in the frontend build.

## Publishing

The repository includes `.github/workflows/deploy.yml`. GitHub Pages should use **GitHub Actions** as its source; pushes to `main` build and publish the Vite site.

Public demo: https://tgf137650-max.github.io/trajagg-explorer-demo/

The user approved publication of this browser-only version. Static browser inference necessarily makes the ONNX model, embedding library and candidate geometries downloadable by visitors. Original PyTorch checkpoints, local environment files and the full Hausdorff matrix are not needed in the public bundle. Map attribution remains visible.
