"""
diagnose.py - CVAE 条件通路体检

条件生成失效时，光看训练 loss 是查不出原因的。这个脚本回答两个具体问题：

  1. 【z 有没有编码 LogP】计算属性头 z -> 归一化 LogP 在验证集上的 R^2。
     如果 R^2 接近 0 且预测值的标准差远小于真值标准差，说明属性头一直在输出均值，
     潜空间根本没携带亲脂性信息。这时无论条件怎么给，生成都不可能有方向性。

  2. 【解码器理不理会条件】固定同一批 z，只改条件值，看生成结果有多少完全不变。
     不变比例越高，说明条件通路越是形同虚设。

运行：
    python diagnose.py
    python diagnose.py --guidance 8.0     # 顺便看 guidance 能把条件影响放大多少
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from datasets import (SmilesDataset, load_zinc_with_logp, make_collate_fn,
                      mol_logp, normalize_cond, scaffold_split)
from generate import ckpt_cond_bounds, load_cvae
from utils import get_device, set_seed

LOGP_SENSITIVITY_TARGETS = (0.0, 3.0, 6.0)


def check_property_head(model, tokenizer, device, smiles, values, cond_min, cond_max):
    """
    检查属性头 z -> 归一化 LogP 的拟合质量。

    输入：
        model    : SmilesVAE
        tokenizer: SMILESTokenizer
        device   : torch.device
        smiles   : list[str]，验证集分子
        values   : np.ndarray，对应的 LogP 原值
        cond_min/max: float，与训练一致的归一化端点
    输出：
        dict，含 r2 / pred_std / true_std
    """
    cond = normalize_cond(values, cond_min, cond_max)
    dataset = SmilesDataset(smiles, tokenizer, conditions=cond)
    loader = DataLoader(dataset, batch_size=64,
                        collate_fn=make_collate_fn(tokenizer.pad_id))

    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"], batch["lengths"], batch["dec_input"],
                        cond=batch["cond"])
            preds.append(out["prop_pred"].cpu().numpy().ravel())
            trues.append(batch["cond"].cpu().numpy().ravel())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    ss_res = float(((trues - preds) ** 2).sum())
    ss_tot = float(((trues - trues.mean()) ** 2).sum())
    return {
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        "pred_std": float(preds.std()),
        "true_std": float(trues.std()),
    }


def check_condition_sensitivity(model, tokenizer, device, cond_min, cond_max,
                                guidance_scale=1.0, num=200,
                                targets=LOGP_SENSITIVITY_TARGETS):
    """
    固定 z、只改条件，统计生成结果完全相同的比例。

    用 greedy 解码消除采样随机性，这样任何差异都只可能来自条件。
    """
    z = torch.randn(num, model.latent_dim, device=device)
    decoded = {}
    for target in targets:
        cond = float(normalize_cond(target, cond_min, cond_max))
        ids = model.sample(num, config.MAX_LEN, device, cond_value=cond, z=z,
                           greedy=True, guidance_scale=guidance_scale)
        decoded[target] = [tokenizer.decode(row.tolist()) for row in ids.cpu()]

    base_key = targets[len(targets) // 2]
    base = decoded[base_key]
    return {t: sum(1 for a, b in zip(decoded[t], base) if a == b) / num
            for t in targets if t != base_key}, base_key


def _valid_split_from_ckpt_or_data(ckpt):
    """优先用 checkpoint 里存的验证集，否则按与训练相同的规则重划。"""
    valid_s = ckpt.get("valid_smiles")
    if valid_s:
        values = []
        kept = []
        for smi in valid_s:
            val = mol_logp(smi)
            if val is None:
                continue
            kept.append(smi)
            values.append(val)
        return kept, np.asarray(values, dtype=np.float32)

    smiles, values = load_zinc_with_logp(
        config.ZINC_FILE, max_molecules=config.FINETUNE_MAX_MOLECULES,
        seed=config.RANDOM_SEED,
    )
    _, valid_idx = scaffold_split(smiles, config.COND_TEST_RATIO, config.RANDOM_SEED)
    return [smiles[i] for i in valid_idx], values[valid_idx]


def main():
    """
    跑完两项体检并打印诊断结论。
    """
    parser = argparse.ArgumentParser(description="CVAE 条件通路体检")
    parser.add_argument("--guidance", type=float, default=1.0,
                        help="条件敏感度检查时使用的 guidance 强度")
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    device = get_device()
    model, tokenizer, ckpt = load_cvae(config.CVAE_CKPT, device)
    cond_min, cond_max = ckpt_cond_bounds(ckpt)

    # 用验证集数据做测试，检查属性头 z -> 归一化 LogP 的拟合质量
    smiles, values = _valid_split_from_ckpt_or_data(ckpt)

    print("\n" + "=" * 70)
    print("检查 1：潜空间有没有编码 LogP")
    print("=" * 70)
    res = check_property_head(model, tokenizer, device, smiles, values,
                              cond_min, cond_max)
    print(f"  属性头 z -> LogP 的 R^2 = {res['r2']:.3f}")
    print(f"  预测值标准差 {res['pred_std']:.4f} vs 真值标准差 {res['true_std']:.4f}")
    if res["r2"] < 0.15:
        print("  [异常] R^2 过低，z 基本没编码 LogP。"
              "优先检查 PROPERTY_WEIGHT 是否相对重构损失太小。")
    else:
        print("  [正常] 潜空间携带了可用的 LogP 信息。")

    print("\n" + "=" * 70)
    print(f"检查 2：解码器理不理会条件（guidance={args.guidance}）")
    print("=" * 70)
    same_ratio, base_key = check_condition_sensitivity(
        model, tokenizer, device, cond_min, cond_max,
        guidance_scale=args.guidance,
    )
    for target, ratio in same_ratio.items():
        print(f"  固定 z，条件 LogP {base_key} -> {target}：生成结果完全相同的比例 = {ratio:.3f}")
    worst = max(same_ratio.values())
    if worst > 0.7:
        print("  [异常] 改条件几乎不改变输出，条件通路形同虚设。"
              "可提高 COND_DROPOUT，或在生成时调大 guidance。")
    else:
        print("  [正常] 条件确实在影响解码结果。")


if __name__ == "__main__":
    main()
