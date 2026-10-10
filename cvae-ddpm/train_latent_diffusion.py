"""
train_latent_diffusion.py - 阶段3：在冻结的 VAE encoder 上训条件潜空间 DDPM

做什么：
    用阶段1/阶段2 的 encoder 把 ZINC 分子编成 mu，再在这份连续潜空间上
    学 p(z | LogP)。生成时不再从 N(0, I) 瞎抽 z，而是按目标 LogP 去噪出 z，
    然后交给原来的 SMILES 解码器。

VAE 权重一律冻结。有 cvae_zinc_logp.pt 就用它（编码时带 LogP）；
否则退回 vae_pretrained.pt（编码不带条件，扩散器仍然看 LogP）。

运行：
    python train_latent_diffusion.py
    python train_latent_diffusion.py --smoke
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import config
from datasets import (SmilesDataset, load_zinc_with_logp, make_collate_fn,
                      normalize_cond, scaffold_split)
from generate import load_cvae
from latent_diffusion import LatentDDPM
from utils import count_parameters, get_device, plot_training_curves, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="阶段3：潜空间条件 DDPM")
    p.add_argument("--epochs", type=int, default=config.LD_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.LD_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LD_LR)
    p.add_argument("--hidden-dim", type=int, default=config.LD_HIDDEN_DIM)
    p.add_argument("--timesteps", type=int, default=config.LD_TIMESTEPS)
    p.add_argument("--cond-dropout", type=float, default=config.LD_COND_DROPOUT)
    p.add_argument("--max-molecules", type=int, default=config.FINETUNE_MAX_MOLECULES)
    p.add_argument("--encode-batch-size", type=int, default=256,
                   help="用 VAE encoder 抽 mu 时的 batch 大小")
    p.add_argument("--smoke", action="store_true", help="冒烟：少量分子、3 个 epoch")
    return p.parse_args()


def resolve_vae_ckpt():
    """优先阶段2 CVAE；没有则用阶段1 预训练权重。"""
    if config.CVAE_CKPT.exists():
        return config.CVAE_CKPT, True
    if config.PRETRAIN_CKPT.exists():
        return config.PRETRAIN_CKPT, False
    raise FileNotFoundError(
        f"未找到 {config.CVAE_CKPT} 或 {config.PRETRAIN_CKPT}，请先跑阶段1/阶段2"
    )


def encode_split(vae, smiles, values, tokenizer, device, encode_with_cond, batch_size):
    """把 SMILES 编成 (mu, 归一化 LogP)。"""
    cond = normalize_cond(values, config.LOGP_MIN, config.LOGP_MAX)
    # dataset 每个样本的 ids和cond
    dataset = SmilesDataset(smiles, tokenizer, conditions=cond)
    # 每个batch返回一个字典，五个键：input_ids, dec_input, dec_target, lengths, cond（归一化之后的条件值）
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=make_collate_fn(tokenizer.pad_id),
    )
    mus, cs = [], []
    vae.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="encode", leave=False):
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            cond_t = batch["cond"].to(device) if encode_with_cond else None
            # 只获取样本映射分布的均值mu，形状为[B, latent_dim]
            mu = vae.encode_mu(input_ids, lengths, cond_t)
            mus.append(mu.cpu())
            cs.append(batch["cond"].cpu())
    # 拼接所有batch的mu和cond，形状分别为[N, latent_dim]和[N, 1]，N为总样本数
    return torch.cat(mus, dim=0), torch.cat(cs, dim=0)


def main():
    args = parse_args()
    set_seed(config.RANDOM_SEED)
    device = get_device()
    max_molecules = 2000 if args.smoke else args.max_molecules
    epochs = 3 if args.smoke else args.epochs

    vae_path, encode_with_cond = resolve_vae_ckpt()
    print("=" * 70)
    print(f"阶段3：潜空间条件 DDPM | device={device}")
    print(f"VAE: {vae_path} | 编码时带条件={encode_with_cond}")
    print("=" * 70)

    vae, tokenizer, _ = load_cvae(vae_path, device)
    vae.eval()
    for p in vae.parameters():     # VAE里所有可学习张量
        p.requires_grad_(False)    # 关掉这些参数的梯度，不建计算图，不再更新

    smiles, values = load_zinc_with_logp(
        config.ZINC_FILE, max_molecules=max_molecules, seed=config.RANDOM_SEED
    )
    train_idx, valid_idx = scaffold_split(
        smiles, test_ratio=config.COND_TEST_RATIO, seed=config.RANDOM_SEED
    )
    train_s = [smiles[i] for i in train_idx]
    train_v = values[train_idx]
    valid_s = [smiles[i] for i in valid_idx]
    valid_v = values[valid_idx]

    print("[latent-ddpm] 用冻结 encoder 抽取 mu 和 cond")
    # mu[N, latent_dim] 和 cond[N, 1]已归一化，N为总样本数
    train_mu, train_c = encode_split(
        vae, train_s, train_v, tokenizer, device, encode_with_cond, args.encode_batch_size
    )
    valid_mu, valid_c = encode_split(
        vae, valid_s, valid_v, tokenizer, device, encode_with_cond, args.encode_batch_size
    )
    print(f"[latent-ddpm] train z {tuple(train_mu.shape)} | valid z {tuple(valid_mu.shape)}")

    ddpm = LatentDDPM(
        latent_dim=vae.latent_dim,
        hidden_dim=args.hidden_dim,
        num_timesteps=args.timesteps,
    ).to(device)

    # 用全部样本mu的 均值 和 标准差 来设置潜空间的尺度，避免扩散直接打在未标定的后验上。
    ddpm.set_latent_stats(train_mu.mean(dim=0), train_mu.std(dim=0))
    print(f"[latent-ddpm] 参数量 {count_parameters(ddpm):,}")

    train_loader = DataLoader(
        TensorDataset(train_mu, train_c), batch_size=args.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        TensorDataset(valid_mu, valid_c), batch_size=args.batch_size, shuffle=False
    )

    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=args.lr)
    history = {"mse": []}
    best = float("inf")
    start = time.time()
    config.CKPT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        ddpm.train()
        train_sum, n_tr = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
        for z_mu, cond in pbar:
            z_mu = z_mu.to(device)
            cond = cond.to(device)
            # 预测噪声，返回MSE损失
            loss = ddpm(z_mu, cond, cond_dropout=args.cond_dropout)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            optimizer.step()
            train_sum += loss.item()
            n_tr += 1
            pbar.set_postfix({"mse": f"{loss.item():.4f}"})

        ddpm.eval()
        valid_sum, n_va = 0.0, 0
        with torch.no_grad():
            for z_mu, cond in valid_loader:
                loss = ddpm(z_mu.to(device), cond.to(device), cond_dropout=0.0)
                valid_sum += loss.item()
                n_va += 1
        train_mse = train_sum / max(n_tr, 1)
        valid_mse = valid_sum / max(n_va, 1)
        history["mse"].append(valid_mse)
        print(f"Epoch {epoch + 1}/{epochs} | train mse={train_mse:.4f} | valid mse={valid_mse:.4f}")

        if valid_mse < best:
            best = valid_mse
            torch.save(
                {
                    "model_state": ddpm.state_dict(),
                    "latent_dim": vae.latent_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_timesteps": args.timesteps,
                    "valid_mse": valid_mse,
                    "epoch": epoch + 1,
                    "vae_ckpt": str(vae_path),
                    "encode_with_cond": encode_with_cond,
                    "logp_min": config.LOGP_MIN,
                    "logp_max": config.LOGP_MAX,
                    "cond_min": config.LOGP_MIN,
                    "cond_max": config.LOGP_MAX,
                    "sample_steps": config.LD_SAMPLE_STEPS,
                    "vocab_file": str(config.VOCAB_FILE),
                },
                config.LATENT_DIFFUSION_CKPT,
            )

    elapsed = (time.time() - start) / 60
    print(f"\n[latent-ddpm] 完成，用时 {elapsed:.1f} 分钟，最佳权重: {config.LATENT_DIFFUSION_CKPT}")
    plot_training_curves(history, config.OUTPUT_DIR / "latent_ddpm_curves.png",
                         title="阶段3 潜空间条件 DDPM")


if __name__ == "__main__":
    main()
