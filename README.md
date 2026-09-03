# TrajAgg Explorer

TrajAgg Explorer is an interactive academic demo of trajectory-similarity retrieval. It supports two clearly labelled data paths: the public site reads 100 validated cases exported from a saved checkpoint, while an optional private FastAPI service performs on-demand inference for any of the 7,000 Porto test trajectories. Neither path trains a model in the browser.

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

By default this opens the public-compatible 100-query static mode. To use live
RTX 3090 inference, first start the server API and an SSH tunnel as documented in
[`backend/README.md`](backend/README.md), then create `.env.local`:

```bash
cp .env.example .env.local
npm run dev
```

The interface will then offer all 7,000 IDs from `Q-03000` through `Q-09999`.
If the private API cannot be reached, it explicitly falls back to the validated
100-query static mode.

## What the interface shows

1. Search and page through 100 real Porto test trajectories selected with a fixed-seed, coverage-oriented strategy.
2. Compare the raw GPS/WGS84 route with its 100 m grid representation.
3. Inspect the real Chebyshev Top-1 or Top-3 from the saved 7,000-trajectory embedding library.
4. Compare measured TrajAgg retrieval time with direct Hausdorff search for the selected query.
5. View each candidate's Chebyshev distance, Hausdorff ground-truth distance, and ground-truth rank.
6. Follow the query → dual-scale encoder → embedding library → Top-k trace.

In live mode, selecting a query loads only its real geometry. Clicking **Run live
Top-k** performs author-compatible preprocessing, fresh query encoding, and
Chebyshev retrieval on the server, then reports the synchronized request timing.

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

The public front end only reads these exported files; it does not load PyTorch,
the 525 MB distance matrix, or the full embedding library. The optional private
backend loads those artifacts once and exposes read-only retrieval endpoints:

- `GET /api/health`
- `GET /api/config`
- `GET /api/queries`
- `GET /api/queries/{query_id}`
- `POST /api/retrieve`

The API is bound to server localhost by default. A public GitHub Pages build
continues to use static mode until a laboratory-approved public HTTPS reverse
proxy is available.

## Publishing

The repository includes `.github/workflows/deploy.yml`. GitHub Pages should use **GitHub Actions** as its source; pushes to `main` build and publish the Vite site.
