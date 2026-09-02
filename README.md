# TrajAgg Explorer

TrajAgg Explorer is an interactive academic demo of trajectory-similarity retrieval. The browser does not train a model: it reads static cases exported offline from a saved TrajAgg checkpoint.

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

## What the interface shows

1. Select one of five real Porto test trajectories.
2. Compare the raw GPS/WGS84 route with its 100 m grid representation.
3. Inspect the real Chebyshev Top-1 or Top-3 from the saved 7,000-trajectory embedding library.
4. View exported Chebyshev distance, `exp(-distance)` similarity, and the corresponding Hausdorff ground-truth value.
5. Follow the query → dual-scale encoder → embedding library → Top-k trace.

The central route canvas places the exported WGS84 coordinates on an interactive Porto street map. It uses OpenStreetMap cartography with visible attribution, and supports pan, zoom, route tooltips, candidate highlighting, and GPS/grid visibility controls. Timestamp and duration are marked unavailable because the author-compatible preprocessed 10,000-trajectory subset retains `trajlen`, `wgs_seq`, and `merc_seq`, but not those metadata fields.

Hausdorff values are not labelled as metres. The author preprocessing calculates them from WGS84 coordinate sequences, so the demo reports them in WGS84 coordinate space.

## Static data layout

```text
public/data/
├── index.json
└── cases/
    ├── Q-03007.json
    ├── Q-03527.json
    ├── Q-04493.json
    ├── Q-06271.json
    └── Q-08988.json
```

`index.json` records configuration, checkpoint selection, strict Test metrics, and query summaries. Each case contains the real query route, grid sequence, saved embedding preview, Top-3 candidates, Chebyshev scores, and Hausdorff ground truth.

## Offline artifact pipeline

The non-invasive export workflow keeps the author repository unchanged:

1. Run the official Porto / Hausdorff / Chebyshev / Hybrid training components.
2. At each author-defined best-validation event, persist `best_checkpoint.pt`, `test_embeddings.pt`, and `best_metrics.json`.
3. Use `scripts/export_trajagg_demo_data.py` to rank candidates and write the static JSON cases.

The front end only reads these exported files; it does not load PyTorch, the 525 MB distance matrix, or the full embedding library.

## Publishing

The repository includes `.github/workflows/deploy.yml`. GitHub Pages should use **GitHub Actions** as its source; pushes to `main` build and publish the Vite site.
