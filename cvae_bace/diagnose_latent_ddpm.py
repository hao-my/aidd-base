"""
diagnose_latent_ddpm.py - 阶段3 潜空间 DDPM 体检

对照四层，定位「扩散 z 解码全是废串」断在哪：
  1. checkpoint / 标准化统计是否正常
  2. 去噪器在真实 mu 上噪声回归得怎么样
  3. 采出的 z 是否还在 encoder 后验附近
  4. 同样解码器下：真实 mu / 高斯 z / 扩散 z 的有效率
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from datasets import (SmilesDataset, load_zinc_with_logp, make_collate_fn,
                      mol_logp, normalize_cond)
from generate import load_cvae
from latent_diffusion import load_latent_ddpm
from utils import canonical_or_none, get_device, set_seed


def _stats(name, z):
    z = z.detach().float().cpu()
    nrm = z.norm(dim=-1)
    return {
        "name": name,
        "shape": tuple(z.shape),
        "mean": float(z.mean()),
        "std": float(z.std()),
        "abs_max": float(z.abs().max()),
        "norm_mean": float(nrm.mean()),
        "norm_std": float(nrm.std()),
        "norm_max": float(nrm.max()),
    }


def _print_stats(s):
    print(
        f"  {s['name']:22s}  mean={s['mean']:+.3f} std={s['std']:.3f} "
        f"|z|_mean={s['norm_mean']:.2f} |z|_max={s['norm_max']:.2f} "
        f"absmax={s['abs_max']:.2f}"
    )


@torch.no_grad()
def encode_mu_batch(vae, smiles, values, tokenizer, device, n=256):
    cond = normalize_cond(values[:n], config.LOGP_MIN, config.LOGP_MAX)
    dataset = SmilesDataset(smiles[:n], tokenizer, conditions=cond)
    loader = DataLoader(
        dataset, batch_size=64, shuffle=False,
        collate_fn=make_collate_fn(tokenizer.pad_id),
    )
    mus, cs = [], []
    for batch in loader:
        mu = vae.encode_mu(
            batch["input_ids"].to(device),
            batch["lengths"].to(device),
            batch["cond"].to(device),
        )
        mus.append(mu)
        cs.append(batch["cond"].to(device))
    return torch.cat(mus, dim=0), torch.cat(cs, dim=0)


@torch.no_grad()
def noise_mse_by_t(ddpm, mu, cond, ts=(50, 200, 500, 800, 999)):
    x0 = ddpm.normalize(mu)
    out = {}
    for t_val in ts:
        t = torch.full((mu.size(0),), t_val, device=mu.device, dtype=torch.long)
        x_t, noise = ddpm.q_sample(x0, t)
        pred = ddpm.denoiser(x_t, t, cond)
        out[t_val] = float(torch.mean((pred - noise) ** 2))
    return out


@torch.no_grad()
def sample_trace(ddpm, cond_value, device, n=32, steps=None):
    """走正式 sample()，记录每步 |x| / |x0_pred|。"""
    steps = int(steps or config.LD_SAMPLE_STEPS)
    z, trace = ddpm.sample(n, cond_value, device, steps=steps, return_trace=True)
    times = [row["t"] for row in trace]
    return z, times, trace


@torch.no_grad()
def decode_validity(vae, tokenizer, device, z, cond_value, n=64):
    z = z[:n]
    ids = vae.sample(
        z.size(0), config.MAX_LEN, device,
        cond_value=cond_value, z=z, greedy=True, guidance_scale=1.0,
    )
    valid = 0
    lengths = []
    samples = []
    for row in ids.cpu():
        raw = tokenizer.decode(row.tolist())
        can = canonical_or_none(raw)
        lengths.append(len(raw) if raw else 0)
        if can is not None:
            valid += 1
            samples.append(can)
        elif len(samples) < 3:
            samples.append(raw[:80] if raw else "")
    return valid / max(z.size(0), 1), float(np.mean(lengths)), samples[:3]


def denorm_logp(pred_norm):
    return pred_norm * (config.LOGP_MAX - config.LOGP_MIN) + config.LOGP_MIN


def main():
    set_seed(config.RANDOM_SEED)
    device = get_device()
    print("=" * 70)
    print(f"阶段3 DDPM 诊断 | device={device}")
    print("=" * 70)

    vae, tokenizer, vae_ckpt = load_cvae(config.CVAE_CKPT, device)
    ddpm, ld_ckpt = load_latent_ddpm(config.LATENT_DIFFUSION_CKPT, device)

    print("\n---------- 1. checkpoint ----------")
    print(f"  VAE  epoch={vae_ckpt.get('epoch')}  validity={vae_ckpt.get('validity')}")
    print(f"  DDPM epoch={ld_ckpt.get('epoch')}  valid_mse={ld_ckpt.get('valid_mse')}")
    print(f"  encode_with_cond={ld_ckpt.get('encode_with_cond')}  "
          f"vae_ckpt={ld_ckpt.get('vae_ckpt')}")
    print(f"  timesteps={ld_ckpt.get('num_timesteps')}  "
          f"sample_steps={ld_ckpt.get('sample_steps')}  "
          f"hidden={ld_ckpt.get('hidden_dim')}")

    zm, zs = ddpm.z_mean, ddpm.z_std
    print(f"  z_mean: mean={float(zm.mean()):+.4f} std={float(zm.std()):.4f} "
          f"absmax={float(zm.abs().max()):.4f}")
    print(f"  z_std : mean={float(zs.mean()):.4f} min={float(zs.min()):.4f} "
          f"max={float(zs.max()):.4f}")

    times = torch.linspace(ddpm.num_timesteps - 1, 0, config.LD_SAMPLE_STEPS).long()
    n_unique = len(set(times.tolist()))
    print(f"  DDIM 时间轴: {config.LD_SAMPLE_STEPS} 步 -> {n_unique} 个不重复 t；"
          f"首尾 {int(times[0])} -> {int(times[-1])}")
    print(f"  alpha_bar[0]={float(ddpm.alphas_cumprod[0]):.6f}  "
          f"alpha_bar[-1]={float(ddpm.alphas_cumprod[-1]):.6e}")

    print("\n---------- 2. 真实分子 encode_mu ----------")
    smiles, values = load_zinc_with_logp(
        config.ZINC_FILE, max_molecules=800, seed=config.RANDOM_SEED
    )
    mu, cond = encode_mu_batch(vae, smiles, values, tokenizer, device, n=256)
    mu_n = ddpm.normalize(mu)
    _print_stats(_stats("real mu", mu))
    _print_stats(_stats("real mu (normed)", mu_n))

    print("\n---------- 3. 去噪器：在真实 mu 上回归噪声 ----------")
    mse_t = noise_mse_by_t(ddpm, mu, cond)
    for t, mse in mse_t.items():
        print(f"  t={t:4d}  noise MSE={mse:.4f}   (随机猜噪声约 1.0)")

    print("\n---------- 4. 三种 z 的尺度 / 属性头 ----------")
    n = 128
    z_gauss = torch.randn(n, vae.latent_dim, device=device)
    targets = [1.0, 3.0, 5.0]
    ddpm_zs = {}
    for logp in targets:
        c = float(normalize_cond(logp, config.LOGP_MIN, config.LOGP_MAX))
        z, _, trace = sample_trace(ddpm, c, device, n=n, steps=config.LD_SAMPLE_STEPS)
        ddpm_zs[logp] = (z, c, trace)

    _print_stats(_stats("gaussian N(0,I)", z_gauss))
    _print_stats(_stats("real mu", mu[:n]))
    for logp, (z, _, _) in ddpm_zs.items():
        _print_stats(_stats(f"ddpm z | LogP={logp:.0f}", z))

    def head_logp(z):
        pred = denorm_logp(vae.property_head(z).squeeze(-1))
        return float(pred.mean()), float(pred.std())

    print("\n  属性头预测的 LogP（原始尺度）：")
    m, s = head_logp(mu[:n])
    print(f"    real mu              mean={m:+.2f}  std={s:.2f}   真值均值={float(values[:n].mean()):+.2f}")
    m, s = head_logp(z_gauss)
    print(f"    gaussian             mean={m:+.2f}  std={s:.2f}")
    for logp, (z, _, _) in ddpm_zs.items():
        m, s = head_logp(z)
        print(f"    ddpm 目标 {logp:.0f}        mean={m:+.2f}  std={s:.2f}")

    print("\n---------- 5. DDIM 轨迹（目标 LogP=3）----------")
    _, _, trace = ddpm_zs[3.0]
    for row in trace[:: max(len(trace) // 8, 1)] + [trace[-1]]:
        print(
            f"  step {row['i']:02d} t={row['t']:4d}  "
            f"|x|={row['x_norm']:.2f}  |x0|={row['x0_norm']:.2f}  "
            f"|eps|={row['eps_norm']:.2f}  x0_absmax={row['x0_absmax']:.2f}"
        )

    print("\n---------- 6. 同一解码器、三种 z 的 greedy 有效率 ----------")
    probe = float(normalize_cond(3.0, config.LOGP_MIN, config.LOGP_MAX))
    cases = [
        ("real mu + 真条件", mu[:64], None),
        ("gaussian + c=3", z_gauss[:64], probe),
        ("ddpm LogP=1", ddpm_zs[1.0][0][:64], ddpm_zs[1.0][1]),
        ("ddpm LogP=3", ddpm_zs[3.0][0][:64], ddpm_zs[3.0][1]),
        ("ddpm LogP=5", ddpm_zs[5.0][0][:64], ddpm_zs[5.0][1]),
    ]
    # real mu 用对应的真实条件
    real_c = cond[:64]
    val, slen, ex = decode_validity(vae, tokenizer, device, mu[:64], real_c, n=64)
    print(f"  real mu + 真条件     validity={val:.3f}  平均串长={slen:.1f}  例: {ex[0][:70] if ex else ''}")
    val, slen, ex = decode_validity(vae, tokenizer, device, z_gauss[:64], probe, n=64)
    print(f"  gaussian + c=3       validity={val:.3f}  平均串长={slen:.1f}  例: {ex[0][:70] if ex else ''}")
    for logp in targets:
        z, c, _ = ddpm_zs[logp]
        val, slen, ex = decode_validity(vae, tokenizer, device, z[:64], c, n=64)
        print(f"  ddpm LogP={logp:.0f}         validity={val:.3f}  平均串长={slen:.1f}  例: {ex[0][:70] if ex else ''}")

    print("\n---------- 7. 对照：扩散 z 若换成「只标准化的高斯」再 denormalize ----------")
    z_calibrated = ddpm.denormalize(torch.randn(64, ddpm.latent_dim, device=device))
    _print_stats(_stats("calibrated gauss", z_calibrated))
    val, slen, ex = decode_validity(vae, tokenizer, device, z_calibrated, probe, n=64)
    print(f"  denorm(N(0,I)) + c=3 validity={val:.3f}  平均串长={slen:.1f}  例: {ex[0][:70] if ex else ''}")
    m, s = head_logp(z_calibrated)
    print(f"  属性头               mean={m:+.2f}  std={s:.2f}")

    print("\n---------- 结论线索 ----------")
    print("  噪声 MSE 接近 1 且各 t 差不多 → 去噪器几乎没学到，DDIM 在乱走。")
    print("  扩散 |z| >> 真实 mu，或轨迹里 |x0| 爆炸 → 采样公式/标准化把 z 推离后验。")
    print("  真实 mu 有效率高、扩散 z 低 → 解码器没坏，坏的是 z 的来源。")
    print("  denorm(N(0,I)) 有效率也低 → 后验尺度和 N(0,I) 差很远，不能当先验。")


if __name__ == "__main__":
    main()
