#!/usr/bin/env python3
"""Export real TrajAgg Porto retrieval cases for the static demo website.

The input checkpoint/embedding artifacts come from ``run_porto_export.py``.
This script does not train a model and never fabricates trajectories or scores.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch


COLORS = ["#ff7b16", "#46b54a", "#8150c7"]
TEST_OFFSET = 3000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export real Porto cases to the TrajAgg Explorer JSON schema.")
    parser.add_argument("--source-dir", required=True, help="Official TrajAgg repository root")
    parser.add_argument("--artifact-dir", required=True, help="Output directory of run_porto_export.py")
    parser.add_argument("--output-dir", required=True, help="Target public/data directory")
    parser.add_argument("--top-k", type=int, default=3, choices=[1, 3])
    parser.add_argument(
        "--query-local-indices", type=int, nargs="+", default=[7, 527, 1493, 3271, 5988],
        help="Indices in the 7,000-trajectory test library, not global Porto row indices",
    )
    return parser.parse_args()


def as_points(values: Iterable[Sequence[float]], digits: int | None = None) -> List[List[float]]:
    result: List[List[float]] = []
    for point in values:
        if digits is None:
            result.append([float(point[0]), float(point[1])])
        else:
            result.append([round(float(point[0]), digits), round(float(point[1]), digits)])
    return result


def route_length_km(mercator: Sequence[Sequence[float]]) -> float:
    if len(mercator) < 2:
        return 0.0
    points = np.asarray(mercator, dtype=np.float64)
    return round(float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum() / 1000.0), 3)


def json_save(value: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def upper_triangle_distance(matrix: Sequence[Sequence[float]], first: int, second: int) -> float:
    if first == second:
        return 0.0
    low, high = sorted((first, second))
    return float(matrix[low][high])


def trajectory_payload(row: Any, cellspace: Any) -> Dict[str, Any]:
    from utils.tools import merc2cell2  # pylint: disable=import-outside-toplevel

    mercator = row.merc_seq
    grid_indices, _ = merc2cell2(torch.tensor(mercator, dtype=torch.float32), cellspace)
    return {
        "gps": as_points(row.wgs_seq, digits=7),
        "mercator": as_points(mercator, digits=3),
        "grid": [[int(cell[0]), int(cell[1])] for cell in grid_indices.tolist()],
    }


def metadata_for(global_index: int, row: Any) -> Dict[str, Any]:
    return {
        "id": f"Q-{global_index:05d}",
        "title": f"Porto test trajectory {global_index}",
        "startTime": "Not available in the preprocessed subset",
        "distanceKm": route_length_km(row.merc_seq),
        "durationMin": None,
        "pointCount": int(row.trajlen),
        "area": "Porto study area",
        "description": (
            "Real Porto test-split case exported from the saved TrajAgg model. "
            "The preprocessed 10,000-trajectory subset retains coordinates and length, "
            "but not timestamp or duration metadata."
        ),
    }


def embedding_preview(vector: torch.Tensor) -> List[float]:
    # The front end uses these values only as a compact, scaled vector preview.
    values = vector.detach().cpu().float().numpy()
    scale = float(np.max(np.abs(values))) or 1.0
    return [round(float(abs(item) / scale), 6) for item in values[:16]]


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).expanduser().resolve()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not (source_dir / "main.py").is_file():
        raise FileNotFoundError(f"Official TrajAgg main.py not found: {source_dir}")

    metadata_path = artifact_dir / "best_metrics.json"
    embeddings_path = artifact_dir / "test_embeddings.pt"
    if not metadata_path.is_file() or not embeddings_path.is_file():
        raise FileNotFoundError("Missing best_metrics.json or test_embeddings.pt in artifact directory")

    os.chdir(source_dir)
    sys.path.insert(0, str(source_dir))
    from utils.cellspace import CellSpace  # pylint: disable=import-outside-toplevel

    with metadata_path.open("r", encoding="utf-8") as handle:
        artifact_metadata = json.load(handle)
    embedding_payload = torch.load(embeddings_path, map_location="cpu")
    embeddings = embedding_payload["embeddings"].float()
    if embeddings.ndim != 2 or embeddings.shape[0] != 7000:
        raise ValueError(f"Expected 7,000 test embeddings; found {tuple(embeddings.shape)}")

    porto_dir = source_dir / "dataset" / "porto"
    with (porto_dir / "porto_1w.pkl").open("rb") as handle:
        all_trajectories = pickle.load(handle)
    with (porto_dir / "traj_simi_dict_hausdorff.pkl").open("rb") as handle:
        haus_upper_triangle = pickle.load(handle)
    if len(all_trajectories) != 10000 or len(haus_upper_triangle) != 10000:
        raise ValueError("Expected the complete 10,000-trajectory Porto data and Hausdorff matrix")

    cellspace = CellSpace(100, 100, -8.7005, 41.1001, -8.5192, 41.2086)
    local_queries = args.query_local_indices
    if len(set(local_queries)) != len(local_queries) or any(index < 0 or index >= len(embeddings) for index in local_queries):
        raise ValueError("Every query local index must be unique and fall in [0, 6999]")

    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    index_queries: List[Dict[str, Any]] = []

    for local_query in local_queries:
        global_query = TEST_OFFSET + local_query
        query_row = all_trajectories.iloc[global_query]
        query_vector = embeddings[local_query]
        distances = torch.amax(torch.abs(embeddings - query_vector.unsqueeze(0)), dim=1)
        distances[local_query] = float("inf")  # Explicitly exclude self, as top_k.py does.
        candidate_locals = torch.argsort(distances)[: args.top_k].tolist()

        candidates = []
        for rank, local_candidate in enumerate(candidate_locals, start=1):
            global_candidate = TEST_OFFSET + local_candidate
            candidate_row = all_trajectories.iloc[global_candidate]
            distance = float(distances[local_candidate])
            candidate_payload = trajectory_payload(candidate_row, cellspace)
            candidates.append({
                "rank": rank,
                "id": f"TRJ-{global_candidate:05d}",
                "sourceIndex": global_candidate,
                "color": COLORS[rank - 1],
                **candidate_payload,
                "chebyshevDistance": round(distance, 8),
                "predictedSimilarity": round(math.exp(-distance), 8),
                "hausdorffDistance": round(
                    upper_triangle_distance(haus_upper_triangle, global_query, global_candidate), 10
                ),
                "hausdorffUnit": "WGS84 coordinate units",
                "note": (
                    "Real candidate from the 7,000-trajectory test library. "
                    "It is ranked by ascending Chebyshev distance between saved TrajAgg embeddings."
                ),
            })

        query_metadata = metadata_for(global_query, query_row)
        case_data = {
            "id": query_metadata["id"],
            "metadata": query_metadata,
            "query": {
                "sourceIndex": global_query,
                **trajectory_payload(query_row, cellspace),
            },
            "candidates": candidates,
            "modelTrace": {
                "gpsEncoder": "GPS/Mercator sequence processed by the learned dual-scale encoder",
                "gridEncoder": "100 m grid sequence processed by the learned dual-scale encoder",
                "mergedEmbedding": embedding_preview(query_vector),
                "queryEmbedding": embedding_preview(query_vector),
                "librarySize": int(embeddings.shape[0]),
                "embeddingDimension": int(embeddings.shape[1]),
            },
            "provenance": {
                "kind": "real exported retrieval case",
                "dataset": "Porto",
                "librarySplit": "test",
                "retrievalLibrarySize": int(embeddings.shape[0]),
                "bestEpoch": artifact_metadata["best_epoch"],
                "selectionRule": artifact_metadata["selection_rule"],
                "groundTruth": "Hausdorff distance from the author-compatible 10,000x10,000 matrix",
                "retrievalDistance": "Chebyshev distance between saved embeddings",
            },
        }
        json_save(case_data, cases_dir / f"{query_metadata['id']}.json")
        index_queries.append({key: query_metadata[key] for key in ("id", "title", "startTime", "distanceKm", "durationMin", "pointCount")})

    index_data = {
        "status": "real-exported-data",
        "dataStatement": (
            "Real Porto test-split retrieval cases exported offline from a saved TrajAgg checkpoint; "
            "these are not synthetic interaction values."
        ),
        "dataOrigin": (
            "Source: offline export from the Porto–Hausdorff TrajAgg artifact. "
            "The model ranks by Chebyshev embedding distance; Hausdorff remains supervision/evaluation ground truth."
        ),
        "config": {
            "dataset": "Porto (10,000-trajectory subset; 7,000 test-library trajectories)",
            "supervisionMetric": "Hausdorff",
            "retrievalDistance": "Chebyshev",
            "gridCellSize": "100 m",
            "mu": "0.5",
            "trainMode": "Hybrid",
        },
        "reproduction": {
            "bestEpoch": artifact_metadata["best_epoch"],
            "selectionRule": artifact_metadata["selection_rule"],
            "testMetrics": artifact_metadata["test_metrics"],
            "artifactStatement": artifact_metadata["statement"],
        },
        "queries": index_queries,
    }
    json_save(index_data, output_dir / "index.json")
    print(f"Exported {len(index_queries)} real retrieval cases to {output_dir}")


if __name__ == "__main__":
    main()
