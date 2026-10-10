#!/usr/bin/env python
"""用 TM-score 比较 noise00 与 noise05 的骨架多样性。

只读 RFdiffusion 写出的复合物，取出结合剂链（默认 B），两两计算
TM-score 后按 0.6 贪心聚类。这一步不参与过滤，不改 scores.csv、
passing.csv、topn.csv 或 funnel.json。

TM-score 来自 tmtools（Zhang 的 TM-align）。比较时取按较短链归一化的分数。
骨架按文件名排序后贪心：与任一已有代表的 TM-score >= 0.6 就并入该簇。

在 Python 3.12 虚拟环境里运行：

    .venv-tmtools\\Scripts\\python.exe analysis\\tmscore_cluster.py

中断后重跑会接着算未完成的配对。配对结果留在 outputs/tmscore/*_scores.f32。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

try:
    from tmtools import tm_align
except ImportError as exc:
    sys.exit(
        "需要 Python 3.12 虚拟环境 .venv-tmtools 里的 tmtools。\n"
        "创建：py -3.12 -m venv .venv-tmtools\n"
        "安装：.venv-tmtools\\Scripts\\python.exe -m pip install tmtools\n"
        f"当前导入失败：{exc}"
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = REPO_ROOT / "outputs"
DIFFUSION = OUTPUTS / "diffusion"
OUT_DIR = OUTPUTS / "tmscore"

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# 每个 worker 进程里的骨架。由 Pool initializer 填入，避免每个批次重复传输。
_COORDS: list[np.ndarray] = []
_SEQS: list[str] = []


def read_chain_ca(path: Path, chain: str) -> tuple[np.ndarray, str]:
    """读取一条链的 Cα 坐标和序列。同一残基只保留第一个 Cα。"""
    coords: list[list[float]] = []
    seq: list[str] = []
    seen: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        if len(line) < 54 or line[21] != chain:
            continue
        if line[12:16].strip() != "CA":
            continue
        if line[16] not in (" ", "A"):
            continue
        key = (line[22:26], line[26])
        if key in seen:
            continue
        seen.add(key)
        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        seq.append(THREE_TO_ONE.get(line[17:20].strip(), "X"))
    if not coords:
        raise ValueError(f"{path.name} 没有链 {chain} 的 Cα")
    return np.asarray(coords, dtype=np.float64), "".join(seq)


def load_group(group: str, chain: str) -> tuple[list[str], list[np.ndarray], list[str]]:
    folder = DIFFUSION / group
    paths = sorted(folder.glob("*.pdb"))
    if not paths:
        sys.exit(f"没有找到 {folder} 下的骨架 PDB")
    names: list[str] = []
    coords: list[np.ndarray] = []
    seqs: list[str] = []
    for path in paths:
        xyz, seq = read_chain_ca(path, chain)
        names.append(path.stem)
        coords.append(xyz)
        seqs.append(seq)
    return names, coords, seqs


def n_pairs(n: int) -> int:
    return n * (n - 1) // 2


def pair_at(k: int, n: int) -> tuple[int, int]:
    """把上三角压缩下标还原成 i < j。"""
    i = 0
    remaining = k
    while True:
        row = n - i - 1
        if remaining < row:
            return i, i + 1 + remaining
        remaining -= row
        i += 1


def shorter_tm(i: int, j: int) -> float:
    coords_i, seq_i = _COORDS[i], _SEQS[i]
    coords_j, seq_j = _COORDS[j], _SEQS[j]
    result = tm_align(coords_i, coords_j, seq_i, seq_j)
    if len(seq_i) <= len(seq_j):
        return float(result.tm_norm_chain1)
    return float(result.tm_norm_chain2)


def _init_worker(coords: list[np.ndarray], seqs: list[str]) -> None:
    global _COORDS, _SEQS
    _COORDS = coords
    _SEQS = seqs


def _score_chunk(pairs: list[tuple[int, int, int]]) -> list[tuple[int, float]]:
    return [(k, shorter_tm(i, j)) for k, i, j in pairs]


def open_scores(path: Path, names: list[str], n: int) -> np.memmap:
    """打开或续写配对分数。名单不一致时停止，避免把旧矩阵套到新骨架上。"""
    manifest = path.with_name(path.stem.replace("_scores", "") + "_names.json")
    # path is <group>_scores.f32; manifest is <group>_names.json
    manifest = path.with_name(path.name.replace("_scores.f32", "_names.json"))
    expected = n_pairs(n)
    if path.exists() and manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved != names:
            sys.exit(f"{manifest.name} 与当前 PDB 名单不一致，请删掉该组的 f32 和 names 后重跑")
        mm = np.memmap(path, dtype=np.float32, mode="r+", shape=(expected,))
        if mm.shape[0] != expected:
            sys.exit(f"{path.name} 的长度与骨架数不符")
        return mm
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(names), encoding="utf-8")
    mm = np.memmap(path, dtype=np.float32, mode="w+", shape=(expected,))
    mm[:] = np.nan
    mm.flush()
    return mm


def fill_scores(
    group: str,
    names: list[str],
    coords: list[np.ndarray],
    seqs: list[str],
    workers: int,
    chunk_size: int,
) -> np.ndarray:
    n = len(names)
    score_path = OUT_DIR / f"{group}_scores.f32"
    mm = open_scores(score_path, names, n)
    pending = np.flatnonzero(np.isnan(mm))
    total = n_pairs(n)
    done = total - int(pending.size)
    print(f"[info] {group}: {n} 条骨架，配对 {total}，已完成 {done}")
    if pending.size == 0:
        return np.array(mm, dtype=np.float32)

    tasks = [(int(k), *pair_at(int(k), n)) for k in pending]
    chunks = [tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)]
    finished = done
    with Pool(processes=workers, initializer=_init_worker, initargs=(coords, seqs)) as pool:
        for scored in pool.imap_unordered(_score_chunk, chunks):
            for k, value in scored:
                mm[k] = value
            mm.flush()
            finished += len(scored)
            print(f"[info] {group}: {finished}/{total}")
    if np.isnan(mm).any():
        sys.exit(f"{group} 仍有未完成的配对")
    return np.array(mm, dtype=np.float32)


def greedy_cluster(scores: np.ndarray, n: int, cutoff: float) -> tuple[list[int], list[bool]]:
    """按文件名顺序贪心。返回每条骨架的簇号，以及它是不是该簇代表。"""
    reps: list[int] = []
    cluster_ids = [-1] * n
    is_rep = [False] * n

    def score_of(i: int, j: int) -> float:
        if i > j:
            i, j = j, i
        k = i * (2 * n - i - 1) // 2 + (j - i - 1)
        return float(scores[k])

    for i in range(n):
        assigned = None
        for rep in reps:
            if score_of(i, rep) >= cutoff:
                assigned = cluster_ids[rep]
                break
        if assigned is None:
            reps.append(i)
            cluster_ids[i] = len(reps) - 1
            is_rep[i] = True
        else:
            cluster_ids[i] = assigned
    return cluster_ids, is_rep


def write_clusters(
    group: str,
    names: list[str],
    seqs: list[str],
    cluster_ids: list[int],
    is_rep: list[bool],
) -> Path:
    path = OUT_DIR / f"{group}_clusters.csv"
    lines = ["name,length,cluster,representative"]
    for name, seq, cid, rep in zip(names, seqs, cluster_ids, is_rep):
        lines.append(f"{name},{len(seq)},{cid},{int(rep)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[write] {path}")
    return path


def cluster_group(
    group: str,
    chain: str,
    cutoff: float,
    workers: int,
    chunk_size: int,
) -> dict:
    names, coords, seqs = load_group(group, chain)
    scores = fill_scores(group, names, coords, seqs, workers, chunk_size)
    cluster_ids, is_rep = greedy_cluster(scores, len(names), cutoff)
    write_clusters(group, names, seqs, cluster_ids, is_rep)
    n_clusters = len({c for c in cluster_ids})
    summary = {
        "structures": len(names),
        "clusters": n_clusters,
        "pairs": n_pairs(len(names)),
    }
    print(f"[info] {group}: {summary['structures']} 条骨架，{n_clusters} 个结构簇")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="按 TM-score 聚类比较推理噪声两组的骨架多样性")
    parser.add_argument("--groups", nargs="+", default=["noise00", "noise05"])
    parser.add_argument("--chain", default="B", help="结合剂链。RFdiffusion 输出里扩散链是 B")
    parser.add_argument("--cutoff", type=float, default=0.6)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--chunk-size", type=int, default=200)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_group = {}
    for group in args.groups:
        by_group[group] = cluster_group(
            group, args.chain, args.cutoff, args.workers, args.chunk_size
        )
    summary = {
        "cutoff": args.cutoff,
        "chain": args.chain,
        "normalization": "shorter",
        "method": "greedy",
        "order": "filename",
        "groups": by_group,
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[write] {summary_path}")


if __name__ == "__main__":
    main()
