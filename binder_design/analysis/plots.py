#!/usr/bin/env python
"""出图：消融对比、正负对照分离、漏斗、指标相关性。

读 outputs/scores.csv（由 05_filter_rank.py 生成），不需要 GPU，
在本机跑即可。四张图分别回答四个问题：

1. noise_ablation.png   —— 推理噪声 0 vs 0.5，interaction pAE 分布差多少
2. positive_vs_negative —— 界面 hotspot 和背面 hotspot 的分布是否分开
3. funnel.png           —— 每条过滤各卡掉多少
4. metric_scatter.png   —— pae_interaction 与 plddt_binder 的关系，看两者是否冗余

用法:
    python analysis/plots.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import OUTPUTS, load_yaml  # noqa: E402

FIGDIR = OUTPUTS / "figures"
PAE_LINE = 10.0  # Bennett et al. 2023 的经验阈值


def _bins(n: int) -> int:
    """按样本量定 bin 数。试点阶段样本少，固定 40 个 bin 会碎成噪声。"""
    return max(10, min(40, n // 10))


def _save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[write] {path}")


def noise_ablation(df: pd.DataFrame) -> None:
    groups = [g for g in ("noise00", "noise05") if g in set(df["group"])]
    if len(groups) < 2:
        print("[skip] 消融图需要 noise00 和 noise05 两组都有数据")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    series = {name: df[df["group"] == name]["pae_interaction"].dropna() for name in groups}
    nbins = _bins(min(len(s) for s in series.values()))
    for name, sub in series.items():
        median = sub.median()
        ax.hist(
            sub,
            bins=nbins,
            alpha=0.55,
            density=True,
            label=f"{name} (n={len(sub)}, median={median:.1f})",
        )
    ax.axvline(PAE_LINE, ls="--", c="k", lw=1)
    ax.annotate(
        "pAE=10",
        xy=(PAE_LINE, ax.get_ylim()[1]),
        xytext=(3, -3),
        textcoords="offset points",
        va="top",
        fontsize=8,
    )
    ax.set_xlabel("pae_interaction")
    ax.set_ylabel("density")
    ax.set_title("Inference noise ablation")
    ax.legend()
    _save(fig, "noise_ablation.png")

    # 顺手把数字打出来，写报告要用
    for name in groups:
        sub = df[df["group"] == name]["pae_interaction"].dropna()
        rate = (sub < PAE_LINE).mean() if len(sub) else float("nan")
        print(
            f"[stat] {name}: median pAE={sub.median():.2f}, "
            f"frac(pAE<10)={rate:.3f}, n={len(sub)}"
        )


def positive_vs_negative(df: pd.DataFrame) -> None:
    if "negctrl" not in set(df["group"]):
        print("[skip] 没有阴性对照组")
        return
    pos = df[df["group"] != "negctrl"]["pae_interaction"].dropna()
    neg = df[df["group"] == "negctrl"]["pae_interaction"].dropna()
    nbins = _bins(min(len(pos), len(neg)))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        pos, bins=nbins, alpha=0.55, density=True,
        label=f"interface hotspots (n={len(pos)}, median={pos.median():.1f})",
    )
    ax.hist(
        neg, bins=nbins, alpha=0.55, density=True,
        label=f"back-surface control (n={len(neg)}, median={neg.median():.1f})",
    )
    ax.axvline(PAE_LINE, ls="--", c="k", lw=1)
    ax.set_xlabel("pae_interaction")
    ax.set_ylabel("density")
    ax.set_title("Positive vs negative hotspot control")
    ax.legend()
    _save(fig, "positive_vs_negative.png")
    print(
        f"[stat] frac(pAE<10): 界面={((pos < PAE_LINE).mean()):.3f} "
        f"背面对照={((neg < PAE_LINE).mean()):.3f}"
    )


def funnel(df: pd.DataFrame, primary: dict) -> None:
    labels = ["AF2 predicted"]
    values = [len(df)]
    mask = pd.Series(True, index=df.index)
    for col in primary:
        flag = f"pass_{col}"
        if flag not in df.columns:
            continue
        mask &= df[flag].fillna(False)
        labels.append(f"+ {col}")
        values.append(int(mask.sum()))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(range(len(values)), values, color="steelblue")
    ax.set_yticks(range(len(values)), labels)
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v, i, f" {v}", va="center", fontsize=9)
    ax.set_xlabel("designs remaining")
    ax.set_title("Cumulative filter funnel")
    _save(fig, "funnel.png")


def metric_scatter(df: pd.DataFrame) -> None:
    sub = df.dropna(subset=["pae_interaction", "plddt_binder"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    passed = sub[sub["pass_primary"]] if "pass_primary" in sub else sub.iloc[0:0]
    failed = sub.drop(passed.index)
    ax.scatter(failed["pae_interaction"], failed["plddt_binder"], s=6, alpha=0.3, label="filtered out")
    ax.scatter(passed["pae_interaction"], passed["plddt_binder"], s=10, alpha=0.8, label="passed")
    ax.axvline(PAE_LINE, ls="--", c="k", lw=1)
    ax.axhline(80.0, ls="--", c="k", lw=1)
    ax.set_xlabel("pae_interaction")
    ax.set_ylabel("plddt_binder")
    ax.set_title("Interface vs monomer confidence")
    ax.legend()
    _save(fig, "metric_scatter.png")
    print(f"[stat] Pearson r = {sub['pae_interaction'].corr(sub['plddt_binder']):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filters", default="config/filters.yaml")
    args = parser.parse_args()

    csv = OUTPUTS / "scores.csv"
    if not csv.exists():
        raise SystemExit(f"没有 {csv}，先跑 scripts/05_filter_rank.py")
    df = pd.read_csv(csv)
    primary = load_yaml(args.filters)["primary"]

    noise_ablation(df)
    positive_vs_negative(df)
    funnel(df, primary)
    metric_scatter(df)


if __name__ == "__main__":
    main()
