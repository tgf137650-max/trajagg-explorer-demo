"""Real-time TrajAgg inference runtime for the Porto academic demo."""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


TEST_OFFSET = 3000
TEST_SIZE = 7000
COLORS = ["#ff7b16", "#46b54a", "#8150c7"]


def as_points(values: Iterable[Sequence[float]], digits: int | None = None) -> List[List[float]]:
    points: List[List[float]] = []
    for value in values:
        if digits is None:
            points.append([float(value[0]), float(value[1])])
        else:
            points.append([round(float(value[0]), digits), round(float(value[1]), digits)])
    return points


def preview_points(values: Sequence[Sequence[float]], max_points: int = 18) -> List[List[float]]:
    if len(values) <= max_points:
        return as_points(values, digits=7)
    indices = np.linspace(0, len(values) - 1, max_points, dtype=np.int64)
    return as_points([values[int(index)] for index in indices], digits=7)


def route_length_km(mercator: Sequence[Sequence[float]]) -> float:
    if len(mercator) < 2:
        return 0.0
    points = np.asarray(mercator, dtype=np.float64)
    return round(float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum() / 1000.0), 3)


def embedding_preview(vector: torch.Tensor) -> List[float]:
    values = vector.detach().cpu().float().numpy()
    scale = float(np.max(np.abs(values))) or 1.0
    return [round(float(abs(item) / scale), 6) for item in values[:16]]


def upper_triangle_distance(matrix: Sequence[Sequence[float]], first: int, second: int) -> float:
    if first == second:
        return 0.0
    low, high = sorted((first, second))
    return float(matrix[low][high])


def parse_query_id(query_id: str) -> Tuple[int, int]:
    normalized = query_id.strip().upper()
    if normalized.startswith("Q-") or normalized.startswith("TRJ-"):
        normalized = normalized.split("-", 1)[1]
    try:
        global_index = int(normalized)
    except ValueError as error:
        raise ValueError("queryId must look like Q-04887 or contain a Porto test index") from error
    if not TEST_OFFSET <= global_index < TEST_OFFSET + TEST_SIZE:
        raise ValueError("queryId must refer to a Porto test trajectory in [Q-03000, Q-09999]")
    return global_index - TEST_OFFSET, global_index


class TrajAggRuntime:
    """Load one model/library and serve deterministic ID-based retrieval requests."""

    def __init__(
        self,
        source_dir: Path,
        artifact_dir: Path,
        gpu_index: int = 0,
        require_cuda: bool = True,
        load_ground_truth: bool = True,
    ) -> None:
        self.source_dir = source_dir.expanduser().resolve()
        self.artifact_dir = artifact_dir.expanduser().resolve()
        self.gpu_index = gpu_index
        self.lock = threading.Lock()

        required = [
            self.source_dir / "main.py",
            self.source_dir / "dataset/porto/porto_1w.pkl",
            self.artifact_dir / "best_checkpoint.pt",
            self.artifact_dir / "test_embeddings.pt",
            self.artifact_dir / "best_metrics.json",
        ]
        if load_ground_truth:
            required.append(self.source_dir / "dataset/porto/traj_simi_dict_hausdorff.pkl")
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing TrajAgg runtime input(s): " + ", ".join(missing))

        os.chdir(self.source_dir)
        sys.path.insert(0, str(self.source_dir))
        from Model.AggTransformer import AggAttnEncoder  # pylint: disable=import-outside-toplevel
        from utils.cellspace import CellSpace  # pylint: disable=import-outside-toplevel
        from utils.tools import merc2cell2  # pylint: disable=import-outside-toplevel

        self.merc2cell2 = merc2cell2
        self.cellspace = CellSpace(100, 100, -8.7005, 41.1001, -8.5192, 41.2086)

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_index}")
        elif require_cuda:
            raise RuntimeError("CUDA is required by configuration but is unavailable")
        else:
            self.device = torch.device("cpu")

        torch.manual_seed(108)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(108)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        checkpoint = torch.load(
            self.artifact_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False
        )
        embedding_payload = torch.load(
            self.artifact_dir / "test_embeddings.pt", map_location="cpu", weights_only=False
        )
        with (self.artifact_dir / "best_metrics.json").open("r", encoding="utf-8") as handle:
            self.metrics = json.load(handle)
        config = checkpoint["config"]

        self.model = AggAttnEncoder(
            int(config["emb_dim"]),
            int(config["nhead"]),
            int(config["nlayer"]),
            float(config["dropout"]),
            float(config["mu"]),
            str(config["metric"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.library = embedding_payload["embeddings"].float().to(self.device)
        if tuple(self.library.shape) != (TEST_SIZE, 128):
            raise ValueError(f"Expected a (7000, 128) embedding library, found {tuple(self.library.shape)}")

        with (self.source_dir / "dataset/porto/porto_1w.pkl").open("rb") as handle:
            trajectories = pickle.load(handle)
        if len(trajectories) != TEST_OFFSET + TEST_SIZE:
            raise ValueError(f"Expected 10,000 Porto trajectories, found {len(trajectories)}")
        self.test_trajectories = trajectories.iloc[TEST_OFFSET:].reset_index(drop=True)

        all_points = np.concatenate([
            np.asarray(value, dtype=np.float64)
            for value in self.test_trajectories.merc_seq
        ])
        self.mean = torch.tensor(all_points.mean(axis=0), dtype=torch.float32)
        self.std = torch.tensor(all_points.std(axis=0), dtype=torch.float32)

        self.ground_truth = None
        if load_ground_truth:
            with (self.source_dir / "dataset/porto/traj_simi_dict_hausdorff.pkl").open("rb") as handle:
                self.ground_truth = pickle.load(handle)
            if len(self.ground_truth) != TEST_OFFSET + TEST_SIZE:
                raise ValueError("Hausdorff matrix does not cover all 10,000 Porto trajectories")

        self.best_epoch = int(checkpoint["best_epoch"])
        self.embedding_dimension = int(self.library.shape[1])
        self._warm_up()

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _preprocess(self, local_index: int):
        row = self.test_trajectories.iloc[local_index]
        grid_xy, retained_points = self.merc2cell2(torch.tensor(row.merc_seq), self.cellspace)
        continuous = ((retained_points - self.mean) / self.std).float().unsqueeze(0).to(self.device)
        discrete = grid_xy.float().unsqueeze(0).to(self.device)
        mask = torch.ones((1, continuous.shape[1]), dtype=torch.bool, device=self.device)
        return continuous, discrete, mask, grid_xy

    def _encode(self, local_index: int) -> torch.Tensor:
        continuous, discrete, mask, _ = self._preprocess(local_index)
        with torch.inference_mode():
            return self.model(continuous, discrete, None, mask)

    def _warm_up(self) -> None:
        with self.lock:
            for _ in range(3):
                self._encode(0)
            self._synchronize()

    def trajectory_summary(self, local_index: int) -> Dict[str, Any]:
        row = self.test_trajectories.iloc[local_index]
        global_index = TEST_OFFSET + local_index
        return {
            "id": f"Q-{global_index:05d}",
            "title": f"Porto test trajectory {global_index}",
            "startTime": "Not available in the preprocessed subset",
            "distanceKm": route_length_km(row.merc_seq),
            "durationMin": None,
            "pointCount": int(row.trajlen),
            "sourceIndex": global_index,
            "previewGps": preview_points(row.wgs_seq),
        }

    def list_queries(self, offset: int, limit: int, search: str = "") -> Dict[str, Any]:
        keyword = search.strip().upper()
        if keyword:
            matching = [
                local_index
                for local_index in range(TEST_SIZE)
                if keyword in f"Q-{TEST_OFFSET + local_index:05d}"
            ]
        else:
            matching = list(range(TEST_SIZE))
        page = matching[offset : offset + limit]
        return {
            "schemaVersion": "2.0",
            "mode": "live-api",
            "total": len(matching),
            "offset": offset,
            "limit": limit,
            "items": [self.trajectory_summary(local_index) for local_index in page],
        }

    def _trajectory_payload(self, local_index: int) -> Dict[str, Any]:
        row = self.test_trajectories.iloc[local_index]
        grid_xy, _ = self.merc2cell2(torch.tensor(row.merc_seq), self.cellspace)
        return {
            "gps": as_points(row.wgs_seq, digits=7),
            "mercator": as_points(row.merc_seq, digits=3),
            "grid": [[int(cell[0]), int(cell[1])] for cell in grid_xy.tolist()],
        }

    def query_preview(self, query_id: str) -> Dict[str, Any]:
        """Return real query geometry without running model inference."""
        local_query, global_query = parse_query_id(query_id)
        metadata = {
            **self.trajectory_summary(local_query),
            "caseFile": None,
            "area": "Porto study area",
            "description": (
                "Real Porto test-split query. Click Retrieve to encode it with "
                "the saved best TrajAgg checkpoint."
            ),
        }
        return {
            "schemaVersion": "3.0",
            "mode": "live-api-query-preview",
            "id": f"Q-{global_query:05d}",
            "metadata": metadata,
            "query": {
                "sourceIndex": global_query,
                **self._trajectory_payload(local_query),
            },
            "candidates": [],
            "modelTrace": {
                "gpsEncoder": "Awaiting on-demand TrajAgg inference",
                "gridEncoder": "Author-compatible 100 m grid preprocessing is ready",
                "mergedEmbedding": [],
                "queryEmbedding": [],
                "librarySize": TEST_SIZE,
                "embeddingDimension": self.embedding_dimension,
            },
            "provenance": {
                "kind": "real query preview; inference not run yet",
                "bestEpoch": self.best_epoch,
                "selectionRule": self.metrics["selection_rule"],
            },
            "timing": None,
        }

    def retrieve(self, query_id: str, top_k: int, include_ground_truth: bool = True) -> Dict[str, Any]:
        if top_k not in (1, 3):
            raise ValueError("topK must be 1 or 3 for this academic demo")
        local_query, global_query = parse_query_id(query_id)
        request_start = time.perf_counter_ns()

        # The lock prevents concurrent requests from interleaving CUDA timing and
        # inference on the single model instance. It does not change model output.
        with self.lock:
            self._synchronize()
            preprocess_start = time.perf_counter_ns()
            continuous, discrete, mask, _ = self._preprocess(local_query)
            self._synchronize()
            preprocess_end = time.perf_counter_ns()

            with torch.inference_mode():
                query_embedding = self.model(continuous, discrete, None, mask)
            self._synchronize()
            encode_end = time.perf_counter_ns()

            with torch.inference_mode():
                distances = torch.amax(torch.abs(self.library - query_embedding), dim=1)
                distances[local_query] = float("inf")
                top_values, top_indices = torch.topk(
                    distances, k=top_k, largest=False, sorted=True
                )
            self._synchronize()
            retrieval_end = time.perf_counter_ns()

            saved_query = self.library[local_query].unsqueeze(0)
            embedding_error = float(
                torch.max(torch.abs(query_embedding - saved_query)).item()
            )
            candidate_locals = [int(value) for value in top_indices.detach().cpu().tolist()]
            candidate_distances = [float(value) for value in top_values.detach().cpu().tolist()]
            query_vector = query_embedding.detach().cpu().squeeze(0)

        ground_truth_started = time.perf_counter_ns()
        ground_truth_distances: np.ndarray | None = None
        ground_truth_ranks: np.ndarray | None = None
        if include_ground_truth:
            if self.ground_truth is None:
                raise ValueError("Ground-truth output was requested but the matrix is not loaded")
            ground_truth_distances = np.asarray([
                float("inf") if candidate == local_query else upper_triangle_distance(
                    self.ground_truth, global_query, TEST_OFFSET + candidate
                )
                for candidate in range(TEST_SIZE)
            ], dtype=np.float64)
            sorted_locals = np.argsort(ground_truth_distances)
            ground_truth_ranks = np.empty(TEST_SIZE, dtype=np.int64)
            ground_truth_ranks[sorted_locals] = np.arange(1, TEST_SIZE + 1, dtype=np.int64)
        ground_truth_ended = time.perf_counter_ns()

        candidates: List[Dict[str, Any]] = []
        for rank, (local_candidate, distance) in enumerate(
            zip(candidate_locals, candidate_distances), start=1
        ):
            global_candidate = TEST_OFFSET + local_candidate
            candidate: Dict[str, Any] = {
                "rank": rank,
                "id": f"TRJ-{global_candidate:05d}",
                "sourceIndex": global_candidate,
                "color": COLORS[rank - 1],
                **self._trajectory_payload(local_candidate),
                "chebyshevDistance": round(distance, 8),
                "predictedSimilarity": round(math.exp(-distance), 8),
                "note": (
                    "Real-time candidate from the 7,000-trajectory test library. "
                    "It is ranked by ascending Chebyshev distance between the newly "
                    "encoded query and the saved TrajAgg embeddings."
                ),
            }
            if ground_truth_distances is not None and ground_truth_ranks is not None:
                ground_truth_rank = int(ground_truth_ranks[local_candidate])
                candidate.update({
                    "hausdorffDistance": round(
                        float(ground_truth_distances[local_candidate]), 10
                    ),
                    "hausdorffUnit": "WGS84 coordinate units",
                    "groundTruthRank": ground_truth_rank,
                    "inGroundTruthTop10": ground_truth_rank <= 10,
                    "inGroundTruthTop50": ground_truth_rank <= 50,
                })
            candidates.append(candidate)

        metadata = {
            **self.trajectory_summary(local_query),
            "caseFile": None,
            "area": "Porto study area",
            "description": (
                "Real Porto test-split query encoded on demand by the saved best "
                "TrajAgg checkpoint."
            ),
        }
        preprocess_ms = (preprocess_end - preprocess_start) / 1e6
        encode_ms = (encode_end - preprocess_end) / 1e6
        retrieval_ms = (retrieval_end - encode_end) / 1e6
        learned_total_ms = (retrieval_end - preprocess_start) / 1e6
        ground_truth_ms = (ground_truth_ended - ground_truth_started) / 1e6
        response_finished = time.perf_counter_ns()

        return {
            "schemaVersion": "3.0",
            "mode": "live-api",
            "id": f"Q-{global_query:05d}",
            "metadata": metadata,
            "query": {
                "sourceIndex": global_query,
                **self._trajectory_payload(local_query),
            },
            "candidates": candidates,
            "modelTrace": {
                "gpsEncoder": "GPS/Mercator sequence encoded on demand by the saved author model",
                "gridEncoder": "100 m grid sequence encoded on demand by the saved author model",
                "mergedEmbedding": embedding_preview(query_vector),
                "queryEmbedding": embedding_preview(query_vector),
                "librarySize": TEST_SIZE,
                "embeddingDimension": self.embedding_dimension,
            },
            "provenance": {
                "kind": "real-time model inference",
                "dataset": "Porto",
                "librarySplit": "test",
                "retrievalLibrarySize": TEST_SIZE,
                "bestEpoch": self.best_epoch,
                "selectionRule": self.metrics["selection_rule"],
                "groundTruth": (
                    "Hausdorff distance from the author-compatible 10,000x10,000 matrix"
                    if include_ground_truth else "not requested"
                ),
                "retrievalDistance": "Chebyshev distance between TrajAgg embeddings",
                "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
            },
            "timing": {
                "status": "measured-live-request",
                "protocol": {
                    "statement": (
                        "Single synchronized request timing. Ground-truth matrix lookup "
                        "and response serialization are excluded from learned retrieval time."
                    ),
                    "cudaSynchronization": self.device.type == "cuda",
                    "browserRenderingTimed": False,
                },
                "learned": {
                    "preprocessMs": round(preprocess_ms, 6),
                    "encodeMs": round(encode_ms, 6),
                    "chebyshevDistanceAndTopKMs": round(retrieval_ms, 6),
                    "totalMs": round(learned_total_ms, 6),
                    "maxAbsErrorVsSavedEmbedding": embedding_error,
                },
                "groundTruthLookupMs": round(ground_truth_ms, 6),
                "serverProcessingMs": round((response_finished - request_start) / 1e6, 6),
            },
        }

    def config_payload(self) -> Dict[str, Any]:
        return {
            "schemaVersion": "3.0",
            "mode": "live-api",
            "config": {
                "dataset": "Porto (10,000-trajectory subset; 7,000 test-library trajectories)",
                "supervisionMetric": "Hausdorff",
                "retrievalDistance": "Chebyshev",
                "gridCellSize": "100 m",
                "mu": "0.5",
                "trainMode": "Hybrid",
            },
            "reproduction": {
                "bestEpoch": self.best_epoch,
                "selectionRule": self.metrics["selection_rule"],
                "testMetrics": self.metrics["test_metrics"],
                "sourceCommit": self.metrics.get("source_commit", "unavailable"),
            },
            "capabilities": {
                "queryIdRange": ["Q-03000", "Q-09999"],
                "queryCount": TEST_SIZE,
                "supportedTopK": [1, 3],
                "groundTruthAvailable": self.ground_truth is not None,
            },
        }

    def health_payload(self) -> Dict[str, Any]:
        return {
            "status": "ready",
            "mode": "live-api",
            "device": str(self.device),
            "gpu": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda" else None
            ),
            "bestEpoch": self.best_epoch,
            "librarySize": TEST_SIZE,
            "embeddingDimension": self.embedding_dimension,
            "groundTruthLoaded": self.ground_truth is not None,
        }
