#!/usr/bin/env python3
"""
TrajAgg Explorer 的离线数据导出接口（预留）。

此脚本刻意不在前端或这里“伪造”模型推理。真实导出时应复用作者的
Mercator / grid / padding 预处理和 checkpoint 推理，再将结果写成
public/data/index.json 与 public/data/cases/<query-id>.json。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent


def existing_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"文件不存在：{path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为 TrajAgg Explorer 导出离线检索案例 JSON（接口预留）。"
    )
    parser.add_argument("--checkpoint", type=existing_path, required=True, help="训练好的 TrajAgg .pth checkpoint")
    parser.add_argument("--trajectories", type=existing_path, required=True, help="Porto 库轨迹及 ID 的预处理文件")
    parser.add_argument("--embeddings", type=existing_path, required=True, help="每条库轨迹的离线 embedding 文件")
    parser.add_argument("--hausdorff-matrix", type=existing_path, help="可选的 Hausdorff 真实距离矩阵")
    parser.add_argument("--query-ids", nargs="+", required=True, help="要导出的查询轨迹 ID")
    parser.add_argument("--output", type=Path, default=Path("public/data"), help="JSON 输出目录，默认 public/data")
    parser.add_argument("--top-k", type=int, default=3, choices=range(1, 51), metavar="1..50")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("TrajAgg Explorer · offline export interface")
    print(f"checkpoint: {args.checkpoint}")
    print(f"trajectories: {args.trajectories}")
    print(f"embeddings: {args.embeddings}")
    print(f"hausdorff matrix: {args.hausdorff_matrix or 'not provided'}")
    print(f"query ids: {', '.join(args.query_ids)}; Top-{args.top_k}")
    print(f"output: {args.output.resolve()}")
    raise NotImplementedError(
        dedent(
            """
            这是一个诚实的预留接口，尚未连接作者代码，因此不会生成伪造的“真实结果”。

            实现时请按以下顺序接入：
            1. 使用作者 dataloader 中完全相同的 Mercator、100 m grid、padding mask 预处理；
            2. 载入 AggTransformer checkpoint，对查询和库轨迹执行 eval() 前向推理；
            3. 使用论文配置的 Chebyshev embedding distance 排序，导出 Top-k；
            4. 若提供 Hausdorff 矩阵，仅把对应真实距离写进案例 JSON 的 hausdorffDistance；
            5. 同时写入 index.json 与 cases/<query-id>.json，前端无需训练或访问 PyTorch。
            """
        ).strip()
    )


if __name__ == "__main__":
    main()
