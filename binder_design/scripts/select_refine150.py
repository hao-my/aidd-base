#!/usr/bin/env python
"""从全量通过设计里挑 150 个骨架，接到 outputs/diffusion/refine150。

每个骨架只留界面 pAE 最低的那条序列，再按这个 pAE 取前 150。
扩散 PDB 用符号链接，不复制、不改全量目录。
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORES = ROOT / "outputs" / "scores.csv"
DEST = ROOT / "outputs" / "diffusion" / "refine150"
MANIFEST = ROOT / "outputs" / "refine150_selection.csv"
N = 150


def main() -> None:
    rows = list(csv.DictReader(SCORES.open(encoding="utf-8")))
    best: dict[str, dict] = {}
    for row in rows:
        if row.get("pass_primary") not in {"True", "true", "1"}:
            continue
        key = row["backbone"]
        pae = float(row["pae_interaction"])
        if key not in best or pae < float(best[key]["pae_interaction"]):
            best[key] = row
    ranked = sorted(best.values(), key=lambda r: float(r["pae_interaction"]))[:N]
    if len(ranked) < N:
        raise SystemExit(f"通过的独立骨架只有 {len(ranked)} 个，不足 {N}")

    DEST.mkdir(parents=True, exist_ok=True)
    for old in DEST.glob("*.pdb"):
        old.unlink()

    with MANIFEST.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["rank", "group", "backbone", "description", "pae_interaction", "src"],
        )
        writer.writeheader()
        for i, row in enumerate(ranked, start=1):
            src = ROOT / "outputs" / "diffusion" / row["group"] / f"{row['backbone']}.pdb"
            if not src.exists():
                raise SystemExit(f"找不到骨架 {src}")
            link = DEST / src.name
            link.symlink_to(src)
            writer.writerow(
                {
                    "rank": i,
                    "group": row["group"],
                    "backbone": row["backbone"],
                    "description": row["description"],
                    "pae_interaction": row["pae_interaction"],
                    "src": str(src),
                }
            )
    print(f"[select] {len(ranked)} 个骨架 -> {DEST}")
    print(f"[select] 最好 pAE {ranked[0]['pae_interaction']}  第150 {ranked[-1]['pae_interaction']}")


if __name__ == "__main__":
    main()
