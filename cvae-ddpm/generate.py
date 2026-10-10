"""
generate.py - 用训练好的 CVAE 做条件分子生成

做什么：
    给定目标 LogP（可以给多个），采潜向量 z，连同条件一起送进解码器自回归生成 SMILES。
    z 默认来自 N(0, I)；若存在阶段3 的潜空间 DDPM，则按目标 LogP 去噪出 z
    （可用 --prior 强制选择）。扩散先验下默认关闭 CFG 与潜空间导向。
    去重后计算基础理化性质并存成 CSV。条件对齐评测直接用本表里的 RDKit logp 列。

两个实现细节：
    1. 采样是"批量试、按需补"：一次采一大批，过滤后如果还不够目标数量就再采一轮，
       直到够数或达到最大轮次。直接采 N 条然后过滤，最后往往只剩一小半。
    2. 去重按 RDKit 规范 SMILES 做，而不是按模型吐出的原始字符串。
       同一个分子可以有多种写法，按原始串去重会把重复分子当成不同分子，虚高唯一率。

运行：
    python generate.py                          # 有 latent_ddpm.pt 则用扩散先验
    python generate.py --prior gaussian         # 强制 N(0, I) + CFG / steering
    python generate.py --target 3.0 --num 500
"""

import argparse

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED
from tqdm import tqdm

import config
from datasets import normalize_cond
from latent_diffusion import load_latent_ddpm
from model import SmilesVAE
from tokenizer import SMILESTokenizer
from utils import canonical_or_none, get_device, set_seed


def choose_prior(name="auto"):
    """auto：有 latent_ddpm.pt 就用扩散先验，否则高斯。"""
    if name == "gaussian":
        return "gaussian"
    if name == "diffusion":
        if not config.LATENT_DIFFUSION_CKPT.exists():
            raise FileNotFoundError(
                f"未找到潜空间 DDPM: {config.LATENT_DIFFUSION_CKPT}，请先运行 train_latent_diffusion.py"
            )
        return "diffusion"
    return "diffusion" if config.LATENT_DIFFUSION_CKPT.exists() else "gaussian"


def resolve_generation_knobs(prior, guidance_arg=None, no_steering=False):
    """
    按先验决定 CFG / steering 默认值。

    扩散先验已经按目标 LogP 采出 z，再叠加 CFG / steering 会把条件拧两次。
    高斯先验仍用阶段2 的默认：GUIDANCE_SCALE + 潜空间导向。
    """
    if prior == "diffusion":
        guidance = 1.0 if guidance_arg is None else float(guidance_arg)
        return guidance, False
    guidance = config.GUIDANCE_SCALE if guidance_arg is None else float(guidance_arg)
    use_steering = config.USE_LATENT_STEERING and not no_steering
    return guidance, use_steering


def ckpt_cond_bounds(ckpt):
    """从 checkpoint 取出 LogP 归一化端点，缺省回退到 config。"""
    vmin = ckpt.get("logp_min", ckpt.get("cond_min", config.LOGP_MIN))
    vmax = ckpt.get("logp_max", ckpt.get("cond_max", config.LOGP_MAX))
    return float(vmin), float(vmax)


def load_cvae(ckpt_path, device):
    """
    从 checkpoint 还原 CVAE 与配套信息。

    输入：
        ckpt_path: str | Path
        device   : torch.device
    输出：
        (model, tokenizer, ckpt)
            model    : SmilesVAE，已 eval()
            tokenizer: SMILESTokenizer
            ckpt     : dict，原始 checkpoint，含 cond_min/max、train_smiles 等
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SmilesVAE(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tokenizer = SMILESTokenizer.load(ckpt.get("vocab_file", config.VOCAB_FILE))
    return model, tokenizer, ckpt


def molecule_properties(smiles):
    """
    计算一个分子的基础理化性质。

    输入：
        smiles: str，规范 SMILES
    输出：
        dict | None，含 qed / logp / mw / num_atoms / num_rings；解析失败返回 None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "qed": QED.qed(mol),
        "logp": Crippen.MolLogP(mol),
        "mw": Descriptors.MolWt(mol),
        "num_atoms": mol.GetNumHeavyAtoms(),
        "num_rings": Chem.rdMolDescriptors.CalcNumRings(mol),
    }


def generate_for_target(model, tokenizer, device, target_logp, num_molecules,
                        cond_min, cond_max, temperature=1.0, batch_size=512,
                        max_rounds=20, show_progress=True,
                        guidance_scale=1.0, use_steering=False,
                        latent_diffuser=None, diffusion_steps=None):
    """
    在给定目标 LogP 下生成一批有效且互不相同的分子。

    z 的来源：
        latent_diffuser 为 None：从 N(0, I) 采样（可再加 steering / CFG）
        否则：用潜空间 DDPM 按目标 LogP 去噪出 z
    """
    # 归一化每次的目标logp  目标-最小值 / 最大值-最小值
    cond = float(normalize_cond(target_logp, cond_min, cond_max))

    unique = {}
    total_sampled = 0
    total_valid = 0
    sample_steps = diffusion_steps or config.LD_SAMPLE_STEPS

    pbar = tqdm(total=num_molecules, desc=f"LogP={target_logp:.1f}",
                disable=not show_progress)

    for _ in range(max_rounds):
        if len(unique) >= num_molecules:
            break

        if latent_diffuser is not None:
            # 从标准正态分布采样，经去噪得到服从当前条件的潜向量采样
            z = latent_diffuser.sample(batch_size, cond, device, steps=sample_steps)
        else:
            z = torch.randn(batch_size, model.latent_dim, device=device)
            if use_steering:
                z = model.steer_latent(z, cond, steps=config.STEER_STEPS,
                                       lr=config.STEER_LR,
                                       prior_weight=config.STEER_PRIOR_WEIGHT)

        with torch.no_grad():
            # 生成分子的token id
            ids = model.sample(batch_size, max_len=config.MAX_LEN, device=device,
                               cond_value=cond, z=z, temperature=temperature,
                               guidance_scale=guidance_scale)
        total_sampled += batch_size

        for row in ids.cpu():
            # 验证是否合法
            can = canonical_or_none(tokenizer.decode(row.tolist()))
            if can is None:
                continue
            total_valid += 1
            if can not in unique:
                unique[can] = True
                pbar.update(1)
                if len(unique) >= num_molecules:
                    break
    pbar.close()

    smiles_list = list(unique.keys())[:num_molecules]
    stats = {
        "target_logp": target_logp,
        "total_sampled": total_sampled,
        "num_valid": total_valid,
        "validity": total_valid / max(total_sampled, 1),
        "uniqueness": len(unique) / max(total_valid, 1),
        "num_returned": len(smiles_list),
        "prior": "diffusion" if latent_diffuser is not None else "gaussian",
    }
    return smiles_list, stats


def generate_table(model, tokenizer, device, targets, num_molecules, cond_min,
                   cond_max, temperature=1.0, guidance_scale=1.0, use_steering=False,
                   latent_diffuser=None, diffusion_steps=None):
    """对一组目标 LogP 批量生成，汇总成一张表。"""
    rows, stats_list = [], []

    # 遍历目标logp列表12345，每种目标logP  生成>=250个分子列表和生成统计信息字典
    for target in targets:
        smiles_list, stats = generate_for_target(
            model, tokenizer, device, target, num_molecules,
            cond_min, cond_max, temperature=temperature,
            guidance_scale=guidance_scale, use_steering=use_steering,
            latent_diffuser=latent_diffuser, diffusion_steps=diffusion_steps,
        )
        stats_list.append(stats)
        print(
            f"[generate] LogP={target:.1f} -> {stats['num_returned']} 个分子 "
            f"| validity={stats['validity']:.3f} uniqueness={stats['uniqueness']:.3f}"
            f" | prior={stats['prior']}"
        )
        for smi in smiles_list:
            # 计算每个分子的基础理化性质
            props = molecule_properties(smi)
            if props is None:
                continue
            rows.append({"target_logp": target, "smiles": smi, **props})

    # pd.DataFrame(rows) 把 列表里的每个字典当成一行，字典的键当成列名。
    return pd.DataFrame(rows), stats_list


def main():
    """
    命令行入口：加载模型 -> 按目标列表生成 -> 存 CSV。

    输入：
        无
    输出：
        无（副作用：写 outputs/generated_molecules.csv）
    """
    parser = argparse.ArgumentParser(description="CVAE 条件分子生成")
    parser.add_argument("--target", type=float, default=None,
                        help="单个目标 LogP；不指定则使用 config.TARGET_LOGP_LIST")
    parser.add_argument("--num", type=int, default=config.NUM_GENERATE)
    parser.add_argument("--temperature", type=float, default=config.SAMPLE_TEMPERATURE)
    parser.add_argument("--guidance", type=float, default=None,
                        help="CFG 强度；扩散先验默认 1.0，高斯先验默认 config.GUIDANCE_SCALE")
    parser.add_argument("--no-steering", action="store_true",
                        help="关闭潜空间导向（高斯先验默认开；扩散先验默认关）")
    parser.add_argument("--prior", type=str, default="auto",
                        choices=["auto", "gaussian", "diffusion"],
                        help="z 的来源：auto=有 DDPM 权重就用扩散")
    parser.add_argument("--out", type=str,
                        default=str(config.OUTPUT_DIR / "generated_molecules.csv"))
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    device = get_device()

    if not config.CVAE_CKPT.exists():
        raise FileNotFoundError(
            f"未找到 CVAE 权重: {config.CVAE_CKPT}，请先运行 train_conditional.py"
        )
    model, tokenizer, ckpt = load_cvae(config.CVAE_CKPT, device)
    cond_min, cond_max = ckpt_cond_bounds(ckpt)

    prior = choose_prior(args.prior)
    guidance, use_steering = resolve_generation_knobs(
        prior, args.guidance, args.no_steering
    )
    latent_diffuser, diffusion_steps = None, None
    if prior == "diffusion":
        latent_diffuser, ld_ckpt = load_latent_ddpm(config.LATENT_DIFFUSION_CKPT, device)
        diffusion_steps = ld_ckpt.get("sample_steps", config.LD_SAMPLE_STEPS)

    targets = [args.target] if args.target is not None else config.TARGET_LOGP_LIST
    print("=" * 70)
    print(f"条件生成 | prior={prior} | 目标 LogP={targets} | 每个 {args.num} 个 "
          f"| T={args.temperature} | guidance={guidance} | 潜空间导向={'开' if use_steering else '关'}")
    print("=" * 70)

    df, _ = generate_table(model, tokenizer, device, targets, args.num,
                           cond_min, cond_max, temperature=args.temperature,
                           guidance_scale=guidance, use_steering=use_steering,
                           latent_diffuser=latent_diffuser,
                           diffusion_steps=diffusion_steps)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n[generate] 共 {len(df)} 个分子已保存: {args.out}")
    if len(df):
        print(df.groupby("target_logp")[["qed", "logp", "mw"]].mean().round(3))


if __name__ == "__main__":
    main()
