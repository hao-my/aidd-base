#!/usr/bin/env python
"""把 outputs/throughput.json 汇总成「这台卡上跑 N 个设计要多久」的表。

阶段 0 试点跑完就应该执行这个，用输出的数字反推正式 campaign 的规模，
不要凭别人的 README 估算自己的卡。

用法:
    python analysis/throughput_report.py --plan 1000 2000 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import THROUGHPUT_FILE  # noqa: E402

# 每条骨架产生几个 AF2 任务，需与 campaign 的 seqs_per_struct 一致
DEFAULT_SEQS_PER_STRUCT = 2


def aggregate(data: dict) -> dict:
    """按阶段大类（diffusion / sequence / predict）算加权平均单位耗时。"""
    buckets: dict[str, list[tuple[int, float]]] = {}
    for stage, entries in data.items():
        kind = stage.split(".")[0]
        for e in entries:
            if e.get("seconds_per_item"):
                buckets.setdefault(kind, []).append((e["n_items"], e["seconds"]))
    out = {}
    for kind, pairs in buckets.items():
        n = sum(p[0] for p in pairs)
        s = sum(p[1] for p in pairs)
        out[kind] = {"n_items": n, "seconds": round(s, 1), "sec_per_item": round(s / n, 2)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=int,
        nargs="+",
        default=[500, 1000, 2000],
        help="想评估的骨架总数（所有组加起来）",
    )
    parser.add_argument("--seqs-per-struct", type=int, default=DEFAULT_SEQS_PER_STRUCT)
    args = parser.parse_args()

    if not THROUGHPUT_FILE.exists():
        raise SystemExit(f"没有 {THROUGHPUT_FILE}，先跑试点 campaign")
    data = json.loads(THROUGHPUT_FILE.read_text(encoding="utf-8"))
    agg = aggregate(data)

    gpu = next(
        (e.get("gpu") for entries in data.values() for e in entries if e.get("gpu")),
        "unknown",
    )
    print(f"GPU: {gpu}\n")
    print("实测单位耗时:")
    for kind, v in agg.items():
        print(f"  {kind:10s} {v['sec_per_item']:8.2f} s/item   (n={v['n_items']})")

    diff = agg.get("diffusion", {}).get("sec_per_item")
    seq = agg.get("sequence", {}).get("sec_per_item")
    pred = agg.get("predict", {}).get("sec_per_item")
    if not (diff and pred):
        print("\n缺 diffusion 或 predict 的实测数据，无法外推")
        return

    print("\n外推（GPU 小时；sequence 若跑在 CPU 上不计入 GPU 时间）:")
    print(f"{'骨架数':>8} {'扩散':>8} {'AF2':>8} {'合计':>8}")
    for n in args.plan:
        h_diff = n * diff / 3600
        h_pred = n * args.seqs_per_struct * pred / 3600
        print(f"{n:>8} {h_diff:>8.1f} {h_pred:>8.1f} {h_diff + h_pred:>8.1f}")
    if seq:
        print(f"\n（序列设计另需约 {seq:.2f} s/design，跑在 CPU 上不占 GPU）")


if __name__ == "__main__":
    main()
