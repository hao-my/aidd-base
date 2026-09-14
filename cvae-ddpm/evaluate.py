"""
evaluate.py - 生成结果的系统评测

回答四个问题：

  1. 生成质量：validity / uniqueness / novelty / 内部多样性 / 骨架新颖性。
     这一组是分子生成的通用体检指标，任何生成模型都得先过这一关。

  2. 条件是否真的起作用（本项目的核心命题）：
     在不同目标 LogP 下各生成一批分子，用 RDKit Crippen MolLogP 计算真实 logP，
     看"目标 LogP"与"生成分子 LogP"是否单调相关。
     不用 CVAE 自己的属性头，避免自评自夸。
     同时做一个空对照：随机从 ZINC 抽同样多的分子算 LogP，
     如果生成分子的 LogP 和随机分子差不多且不随目标变化，那说明条件没生效。

  3. 生成分子像不像训练分布：QED / logP / 分子量与 ZINC 训练集对比。

  4. 化学空间在哪：Morgan 指纹 + t-SNE，把 ZINC 训练分子与不同条件下生成的分子
     画在同一张图上。理想情况是生成分子落在 ZINC 流形上，而不是漂到未知区域。

运行：
    python evaluate.py                  # 有 latent_ddpm.pt 时主实验走扩散先验
    python evaluate.py --prior gaussian --ablation
    python evaluate.py --num 500        # 每个条件生成 500 个，跑得快些
"""

import argparse
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from rdkit import DataStructs  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

import config  # noqa: E402
from datasets import load_zinc, mol_logp, murcko_scaffold  # noqa: E402
from fingerprints import smiles_to_array, smiles_to_fps  # noqa: E402
from generate import (ckpt_cond_bounds, choose_prior, generate_table,  # noqa: E402
                      load_cvae, molecule_properties, resolve_generation_knobs)
from latent_diffusion import load_latent_ddpm  # noqa: E402
from utils import get_device, set_seed  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TARGET_COL = "target_logp"


# ==================== 相似度指标 ====================

def internal_diversity(fps, max_n=500, seed=42):
    """
    内部多样性 = 1 - 两两 Tanimoto 相似度均值。越接近 1 说明生成集合越分散。

    输入：
        fps  : list[ExplicitBitVect]
        max_n: int，最多取多少个做两两比较（两两比较是 O(n^2)，全算太慢）
        seed : int
    输出：
        float，取值 0~1；样本不足 2 个时返回 0.0
    """
    if len(fps) < 2:
        return 0.0
    if len(fps) > max_n:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(fps), size=max_n, replace=False)
        fps = [fps[i] for i in idx]

    sims = []
    for i in range(len(fps) - 1):
        # BulkTanimotoSimilarity 一次算一个 vs 一批，比双层 Python 循环快得多
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
    return float(1.0 - np.mean(sims))


def nearest_neighbor_similarity(gen_fps, ref_fps, max_ref=2000, seed=42):
    """
    每个生成分子到参考集合的最大 Tanimoto 相似度（衡量"抄得有多像"）。

    输入：
        gen_fps: list[ExplicitBitVect]，生成分子指纹
        ref_fps: list[ExplicitBitVect]，参考（如 ZINC 训练集）指纹
        max_ref: int，参考集抽样上限
        seed   : int
    输出：
        np.ndarray[float]，长度等于 gen_fps
    """
    if not gen_fps or not ref_fps:
        return np.array([])
    if len(ref_fps) > max_ref:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(ref_fps), size=max_ref, replace=False)
        ref_fps = [ref_fps[i] for i in idx]
    return np.array([max(DataStructs.BulkTanimotoSimilarity(fp, ref_fps)) for fp in gen_fps])


# ==================== 各项评测 ====================

def basic_metrics(df, stats_list, train_smiles):
    """
    生成质量的通用指标。

    输入：
        df          : pd.DataFrame，生成结果（含 smiles 列）
        stats_list  : list[dict]，各目标的采样统计（validity/uniqueness 来自采样过程）
        train_smiles: list[str]，阶段2 训练集，用于 novelty
    输出：
        dict[str, float]
    """
    smiles = df["smiles"].tolist()
    train_set = set(train_smiles)

    novel_vs_train = sum(1 for s in smiles if s not in train_set) / max(len(smiles), 1)

    train_scaffolds = {murcko_scaffold(s) for s in train_smiles}
    gen_scaffolds = [murcko_scaffold(s) for s in smiles]
    novel_scaffold = sum(
        1 for sc in gen_scaffolds if sc and sc not in train_scaffolds
    ) / max(len(gen_scaffolds), 1)

    gen_fps, _ = smiles_to_fps(smiles)
    ref_fps, _ = smiles_to_fps(train_smiles)
    nn_sim = nearest_neighbor_similarity(gen_fps, ref_fps)

    return {
        "validity": float(
            sum(s["num_valid"] for s in stats_list)
            / max(sum(s["total_sampled"] for s in stats_list), 1)
        ),
        "uniqueness": float(np.mean([s["uniqueness"] for s in stats_list])),
        "novelty_vs_train": float(novel_vs_train),
        "novel_scaffold_ratio": float(novel_scaffold),
        "unique_scaffolds": int(len(set(sc for sc in gen_scaffolds if sc))),
        "internal_diversity": internal_diversity(gen_fps),
        "mean_nn_similarity_to_train": float(np.mean(nn_sim)) if len(nn_sim) else 0.0,
        "n_molecules": int(len(smiles)),
    }


def _alignment_stats(df, pred_col, baseline_preds, target_col=TARGET_COL):
    """
    给定一列实测性质，算出条件对齐的各项统计量。

    输入：
        df            : pd.DataFrame，含 target_col 与 pred_col
        pred_col      : str，实测列名（当前为 RDKit logp）
        baseline_preds: np.ndarray，ZINC 随机分子的实测值（空对照）
        target_col    : str
    输出：
        dict，含逐目标均值/标准差、相关系数、MAE、空对照统计
    """
    sub = df[[target_col, pred_col]].dropna()
    # groupby(target_col)：按目标值分堆。[pred_col]：每组里只看实测 logp 这一列。
    # agg(["mean", "std"])：对每组计算均值和标准差
    # 结果 per_target 是一张小表：索引是各个目标 LogP，两列是 mean 和 std
    per_target = sub.groupby(target_col)[pred_col].agg(["mean", "std"])

    targets = sub[target_col].to_numpy(dtype=float)
    preds = sub[pred_col].to_numpy(dtype=float)
    has_variation = len(set(targets)) > 1

    return {
        # 生成分子中，每个目标logP的均值和标准差，形成字典
        "per_target_mean": {float(k): float(v) for k, v in per_target["mean"].items()},
        "per_target_std": {float(k): float(v) for k, v in per_target["std"].items()},
        # 所有分子（不分组）上，目标向量 targets 与实测向量 preds 的 Pearson 相关系数
        "pearson_r": float(pearsonr(targets, preds)[0]) if has_variation else float("nan"),
        # 同样两列的 Spearman 秩相关：只看排序是否一致（目标 1<2<3<4<5 时，实测是不是也大致这个顺序）
        "spearman_r": float(spearmanr(targets, preds)[0]) if has_variation else float("nan"),
        # 逐分子 |实测 LogP - 目标 LogP| 再平均，即 MAE。相关只看「会不会一起变」，MAE 看「差多远」
        "mae_target_vs_pred": float(np.mean(np.abs(preds - targets))),
        # 500 个随机 ZINC 分子的实测 LogP 均值和标准差。空对照：条件若没生效，生成分子的实测应堆在这附近。nanmean / nanstd 会跳过算不出 LogP 的分子。
        "zinc_baseline_mean_pred": float(np.nanmean(baseline_preds)),
        "zinc_baseline_std_pred": float(np.nanstd(baseline_preds)),
    }


def condition_alignment(df, zinc_baseline_smiles):
    """
    核心评测：目标 LogP 变化时，生成分子的 RDKit MolLogP 是否跟着变。

    输入：
        df                  : pd.DataFrame，含 target_logp 与 smiles / logp
        zinc_baseline_smiles: list[str]，随机 ZINC 分子，作为"条件没生效"的空对照
    输出：
        (df_with_logp, summary)
    """
    df = df.copy()
    if "logp" not in df.columns:
        logps = []
        for smi in df["smiles"].tolist():
            val = mol_logp(smi)
            logps.append(np.nan if val is None else val)
        df["logp"] = logps

    # 计算ZINC随机分子的RDKit MolLogP
    base = np.array(
        [v for v in (mol_logp(s) for s in zinc_baseline_smiles) if v is not None],
        dtype=float,
    )

    summary = _alignment_stats(df, "logp", base, target_col=TARGET_COL)
    return df, summary


# ==================== 画图 ====================

def plot_condition_alignment(df, summary, train_logp, save_path):
    """
    条件对齐图：箱线图 + 组均值响应曲线。对照是 ZINC 训练集的 RDKit LogP。

    输入：
        df        : pd.DataFrame，含 target_logp 与 logp
        summary   : dict，condition_alignment 的输出
        train_logp: np.ndarray，训练集 RDKit LogP
        save_path : str | Path
    输出：
        无（副作用：保存 png）
    """
    targets = sorted(df[TARGET_COL].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    data = [df.loc[df[TARGET_COL] == t, "logp"].dropna().to_numpy() for t in targets]
    axes[0].boxplot(data, tick_labels=[f"{t:.1f}" for t in targets], showfliers=False)
    axes[0].axhline(summary["zinc_baseline_mean_pred"], color="gray", ls="--",
                    label=f"ZINC 随机分子空对照 {summary['zinc_baseline_mean_pred']:.2f}")
    if len(train_logp):
        axes[0].axhline(float(np.mean(train_logp)), color="green", ls=":",
                        label=f"ZINC 训练均值 {np.mean(train_logp):.2f}")
    axes[0].set_xlabel("目标 LogP（生成条件）")
    axes[0].set_ylabel("RDKit MolLogP")
    axes[0].set_title(f"生成分子 LogP 分布\nPearson r={summary['pearson_r']:.3f} "
                      f"Spearman={summary['spearman_r']:.3f}")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    means = [summary["per_target_mean"].get(float(t), np.nan) for t in targets]
    stds = [summary["per_target_std"].get(float(t), np.nan) for t in targets]
    axes[1].errorbar(targets, means, yerr=stds, marker="o", capsize=4,
                     color="tab:blue", label="生成分子 LogP 均值")
    lo, hi = min(targets) - 0.5, max(targets) + 0.5
    axes[1].plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="理想 y=x")
    axes[1].set_xlabel("目标 LogP")
    axes[1].set_ylabel("RDKit MolLogP 均值")
    axes[1].set_title("条件响应")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[eval] 条件对齐图已保存: {save_path}")


def plot_property_distributions(df, train_smiles, save_path):
    """
    生成分子与 ZINC 训练集的 QED / logP / 分子量分布对比。
    """
    rng = np.random.RandomState(config.RANDOM_SEED)
    sample = list(train_smiles)
    if len(sample) > config.NUM_REAL_FOR_TSNE:
        sample = list(rng.choice(sample, size=config.NUM_REAL_FOR_TSNE, replace=False))
    real_props = [p for p in (molecule_properties(s) for s in sample) if p]
    real_df = pd.DataFrame(real_props)

    items = [("qed", "QED（类药性）"), ("logp", "logP（亲脂性）"), ("mw", "分子量")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, name) in zip(axes, items):
        ax.hist(real_df[key], bins=40, alpha=0.55, density=True, label="ZINC 训练集")
        ax.hist(df[key], bins=40, alpha=0.55, density=True, label="CVAE 生成")
        ax.set_xlabel(name)
        ax.set_ylabel("密度")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("生成分子 vs ZINC 训练集：理化性质分布")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[eval] 性质分布图已保存: {save_path}")


def plot_chemical_space(df, train_smiles, zinc_smiles, save_path, main_target):
    """
    Morgan 指纹 + t-SNE 化学空间图：ZINC 背景、训练集、生成分子。
    """
    n_train = min(config.NUM_REAL_FOR_TSNE, len(train_smiles))
    n_zinc = min(config.NUM_REAL_FOR_TSNE, len(zinc_smiles))
    rng = np.random.RandomState(config.RANDOM_SEED)
    train_sample = list(rng.choice(train_smiles, size=n_train, replace=False))
    zinc_sample = list(rng.choice(zinc_smiles, size=n_zinc, replace=False))

    gen_main = df.loc[df[TARGET_COL] == main_target, "smiles"].tolist()
    if not gen_main:
        top = df[TARGET_COL].max()
        gen_main = df.loc[df[TARGET_COL] == top, "smiles"].tolist()
        main_target = top
    gen_main = gen_main[: config.NUM_REAL_FOR_TSNE]

    groups = [
        ("ZINC 随机分子", zinc_sample, "lightgray", 8, 0.5),
        ("ZINC 训练集", train_sample, "tab:green", 14, 0.7),
        (f"CVAE 生成 (目标 LogP={main_target:.1f})", gen_main, "tab:red", 14, 0.7),
    ]

    blocks, labels = [], []
    for name, smis, _, _, _ in groups:
        X, kept = smiles_to_array(smis)
        blocks.append(X)
        labels.extend([name] * len(kept))
    arr = np.vstack(blocks)

    perplexity = min(30, max(5, len(arr) // 4))
    coords = TSNE(n_components=2, perplexity=perplexity, init="pca",
                  random_state=config.RANDOM_SEED).fit_transform(arr)

    fig, ax = plt.subplots(figsize=(8, 7))
    labels = np.array(labels)
    for name, _, color, size, alpha in groups:
        mask = labels == name
        ax.scatter(coords[mask, 0], coords[mask, 1], s=size, c=color,
                   alpha=alpha, label=f"{name} (n={int(mask.sum())})",
                   edgecolors="none")
    ax.set_xlabel("t-SNE 维度 1")
    ax.set_ylabel("t-SNE 维度 2")
    ax.set_title("化学空间分布（Morgan ECFP4 + t-SNE）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[eval] 化学空间图已保存: {save_path}")


# ==================== 消融 ====================

def run_ablation(model, tokenizer, device, cond_min, cond_max,
                 zinc_baseline, num=250, temperature=1.0,
                 latent_diffuser=None, diffusion_steps=None):
    """
    组件消融：高斯先验下分别开关 guidance / 潜空间导向；
    若已训阶段3 DDPM，再加一行「潜空间扩散（guidance=1，无 steering）」。
    """
    settings = [
        ("baseline 仅条件", 1.0, False, None),
        ("+ guidance", config.GUIDANCE_SCALE, False, None),
        ("+ 潜空间导向", 1.0, True, None),
        ("+ 两者", config.GUIDANCE_SCALE, True, None),
    ]
    if latent_diffuser is not None:
        settings.append(("潜空间扩散", 1.0, False, latent_diffuser))

    print("\n" + "=" * 70)
    print(f"组件消融（每个目标 {num} 个分子，指标为 RDKit MolLogP）")
    print("=" * 70)

    result = {}
    # 目标logp范围：1-5
    lo_t, hi_t = min(config.TARGET_LOGP_LIST), max(config.TARGET_LOGP_LIST)

    # 遍历四种消融组合
    for name, guidance, steering, diffuser in settings:
        # 每种消融组合中，5种目标logP都生成>=250个分子，结果汇总成一张表df_ab；每个目标logP的生成统计信息字典列表stats_ab
        df_ab, stats_ab = generate_table(
            model, tokenizer, device, config.TARGET_LOGP_LIST, num,
            cond_min, cond_max, temperature=temperature,
            guidance_scale=guidance, use_steering=steering,
            latent_diffuser=diffuser, diffusion_steps=diffusion_steps,
        )

        # 空结果兜底：如果汇总表为空，则跳过不算Spearman和有效率
        if df_ab.empty:
            result[name] = {"spearman_rdkit_logp": float("nan"), "validity": 0.0}
            continue
        
        # 计算生成分子与目标logP的Spearman和Pearson相关系数、MAE、空对照统计
        df_ab, summary = condition_alignment(df_ab, zinc_baseline)

        # 计算一个消融条件下，所有目标logP汇总的生成有效率
        validity = (sum(s["num_valid"] for s in stats_ab)
                    / max(sum(s["total_sampled"] for s in stats_ab), 1))
        result[name] = {
            "spearman_rdkit_logp": summary["spearman_r"],
            "pearson_rdkit_logp": summary["pearson_r"],
            "mae": summary["mae_target_vs_pred"],
            "mean_logp_at_lowest_target": summary["per_target_mean"].get(lo_t, float("nan")),
            "mean_logp_at_highest_target": summary["per_target_mean"].get(hi_t, float("nan")),
            "validity": validity,
        }
        r = result[name]
        print(f"  {name:18s} | Spearman={r['spearman_rdkit_logp']:+.3f} "
              f" MAE={r['mae']:.3f} "
              f" LogP {r['mean_logp_at_lowest_target']:.2f}->"
              f"{r['mean_logp_at_highest_target']:.2f} "
              f" validity={r['validity']:.3f}")

    return result


# ==================== 主流程 ====================

def main():
    """
    完整评测流程：生成 -> 通用指标 -> RDKit LogP 条件对齐 -> 性质分布 -> 化学空间 -> 汇总存盘。
    """
    parser = argparse.ArgumentParser(description="CVAE 生成结果系统评测")
    parser.add_argument("--num", type=int, default=config.NUM_GENERATE,
                        help="每个目标 LogP 生成多少个分子")
    parser.add_argument("--temperature", type=float, default=config.SAMPLE_TEMPERATURE)
    parser.add_argument("--guidance", type=float, default=None,
                        help="CFG 强度；扩散先验默认 1.0，高斯先验默认 config.GUIDANCE_SCALE")
    parser.add_argument("--no-steering", action="store_true", help="关闭潜空间导向")
    parser.add_argument("--prior", type=str, default="auto",
                        choices=["auto", "gaussian", "diffusion"],
                        help="z 的来源：auto=有 DDPM 权重就用扩散")
    parser.add_argument("--ablation", action="store_true",
                        help="做消融对比：高斯先验下 仅条件 / guidance / 导向 / 两者，"
                             "以及（若有权重）潜空间扩散")
    parser.add_argument("--reuse-csv", type=str, default=None,
                        help="复用已有的生成 CSV，跳过生成步骤")
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    device = get_device()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"CVAE 生成结果评测（ZINC LogP / RDKit）| device={device}")
    print("=" * 70)

    model, tokenizer, ckpt = load_cvae(config.CVAE_CKPT, device)
    zinc_smiles = load_zinc(config.ZINC_FILE, max_molecules=5000, seed=config.RANDOM_SEED)
    train_smiles = ckpt.get("train_smiles", zinc_smiles)
    train_logp = np.array(
        [v for v in (mol_logp(s) for s in train_smiles[: min(2000, len(train_smiles))])
         if v is not None],
        dtype=float,
    )

    # 取出checkpoint中保存的cond_min和cond_max，-2到8
    cond_min, cond_max = ckpt_cond_bounds(ckpt)

    # 先验选择：auto：有 latent_ddpm.pt 就用扩散先验，否则高斯。
    prior = choose_prior(args.prior)

    # 按先验决定CFG强度和潜空间导向。
    # 扩散先验guidance=1，关闭潜空间导向；高斯先验guidance=config.GUIDANCE_SCALE，开启潜空间导向。
    # CFG无分类器引导：解码时放大  条件带来的差异。不改z，改解码器每一步的 token logits   
    # 潜空间导向：不改解码logits，而是调整送进解码器之前的潜向量z，使其更符合条件。
    guidance, use_steering = resolve_generation_knobs(
        prior, args.guidance, args.no_steering
    )

    latent_diffuser, diffusion_steps = None, None

    # 如果prior是扩散先验，则加载潜空间扩散模型
    if prior == "diffusion" or (args.ablation and config.LATENT_DIFFUSION_CKPT.exists()):
        latent_diffuser, ld_ckpt = load_latent_ddpm(
            config.LATENT_DIFFUSION_CKPT, device
        )
        # 取出checkpoint中保存的sample_steps，去噪步数
        diffusion_steps = ld_ckpt.get("sample_steps", config.LD_SAMPLE_STEPS)
    main_diffuser = latent_diffuser if prior == "diffusion" else None

    # 随机选择500个ZINC分子作为空对照
    rng = np.random.RandomState(config.RANDOM_SEED)
    zinc_baseline = list(rng.choice(zinc_smiles, size=min(500, len(zinc_smiles)),
                                    replace=False))

    ablation_result = None
    # 进行消融对比，每种消融状态，每个目标logP生成>=250个分子
    # 计算生成分子logP与目标logP的Spearman和Pearson相关系数、MAE、不同logP的均值和方差
    if args.ablation:
        ablation_result = run_ablation(
            model, tokenizer, device, cond_min, cond_max,
            zinc_baseline, num=max(args.num // 4, 100), temperature=args.temperature,
            latent_diffuser=latent_diffuser, diffusion_steps=diffusion_steps,
        )

    if args.reuse_csv:
        df = pd.read_csv(args.reuse_csv)
        stats_list = [{"num_valid": len(df), "total_sampled": len(df), "uniqueness": 1.0}]
        print(f"[eval] 复用已有生成结果: {args.reuse_csv} ({len(df)} 个分子)")
    else:
        print(f"\n[eval] 主实验生成 | prior={prior} | guidance={guidance} "
              f"| 潜空间导向={'开' if use_steering else '关'}")
        df, stats_list = generate_table(
            model, tokenizer, device, config.TARGET_LOGP_LIST, args.num,
            cond_min, cond_max, temperature=args.temperature,
            guidance_scale=guidance, use_steering=use_steering,
            latent_diffuser=main_diffuser, diffusion_steps=diffusion_steps,
        )
        df.to_csv(config.OUTPUT_DIR / "generated_molecules.csv", index=False)

    if df.empty:
        print("[eval] 没有生成任何有效分子，评测终止。请检查阶段1/阶段2 的训练是否收敛。")
        return

    metrics = basic_metrics(df, stats_list, train_smiles)
    print("\n---------- 生成质量 ----------")
    for k, v in metrics.items():
        print(f"  {k:32s}: {v:.4f}" if isinstance(v, float) else f"  {k:32s}: {v}")

    df, cond_summary = condition_alignment(df, zinc_baseline)

    print("\n---------- 条件对齐（RDKit MolLogP）----------")
    for t in sorted(cond_summary["per_target_mean"]):
        print(f"  目标 LogP={t:.1f} -> 实测均值 {cond_summary['per_target_mean'][t]:.3f} "
              f"± {cond_summary['per_target_std'][t]:.3f}")
    print(f"  Pearson r  : {cond_summary['pearson_r']:.4f}")
    print(f"  Spearman   : {cond_summary['spearman_r']:.4f}")
    print(f"  MAE        : {cond_summary['mae_target_vs_pred']:.4f}")
    print(f"  ZINC 空对照: {cond_summary['zinc_baseline_mean_pred']:.3f} "
          f"± {cond_summary['zinc_baseline_std_pred']:.3f}")

    df.to_csv(config.OUTPUT_DIR / "generated_molecules_scored.csv", index=False)

    plot_condition_alignment(df, cond_summary, train_logp,
                             config.OUTPUT_DIR / "condition_alignment.png")
    plot_property_distributions(df, train_smiles,
                                config.OUTPUT_DIR / "property_distributions.png")
    plot_chemical_space(df, train_smiles, zinc_smiles,
                        config.OUTPUT_DIR / "chemical_space_tsne.png",
                        config.MAIN_TARGET_LOGP)

    summary = {
        "condition": "zinc_rdkit_logp",
        "generation_quality": metrics,
        "condition_alignment": {"rdkit_logp": cond_summary},
        "generation_config": {
            "prior": prior if not args.reuse_csv else "reused_csv",
            "num_per_target": args.num,
            "temperature": args.temperature,
            "guidance_scale": guidance,
            "latent_steering": use_steering,
            "targets": config.TARGET_LOGP_LIST,
            "cond_min": cond_min,
            "cond_max": cond_max,
        },
    }
    if ablation_result is not None:
        summary["ablation"] = ablation_result
    out_json = config.OUTPUT_DIR / "evaluation_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[eval] 评测汇总已保存: {out_json}")

    print("\n---------- 结论速读 ----------")
    rho = cond_summary["spearman_r"]
    mae = cond_summary["mae_target_vs_pred"]
    if rho > 0.5 and mae < 1.5:
        print(f"  条件生效：RDKit LogP 与目标明显对齐（Spearman {rho:.3f}，MAE {mae:.3f}）。")
    elif rho > 0.3:
        print(f"  条件弱生效：趋势对但绝对值有偏差（Spearman {rho:.3f}，MAE {mae:.3f}）。")
    else:
        print("  条件基本未生效：检查潜空间是否 collapse（KL 是否贴着 free bits 下限）。")
    if metrics["validity"] < 0.5:
        print("  有效率偏低：建议增大阶段1 的数据量/轮数，或把采样温度调到 1.0 以下。")


if __name__ == "__main__":
    main()
