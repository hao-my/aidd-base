"""采样出的 z 必须落在 encoder 后验附近，不能出现 |z|~1e4 的爆炸。"""

import torch

import config
from latent_diffusion import load_latent_ddpm
from utils import get_device, set_seed


def test_ddim_z_stays_on_posterior_scale():
    set_seed(0)
    device = get_device()
    ddpm, _ = load_latent_ddpm(config.LATENT_DIFFUSION_CKPT, device)
    # 返回从标准正态分布采样，再经条件去噪后的潜向量
    z = ddpm.sample(64, cond_value=0.5, device=device, steps=config.LD_SAMPLE_STEPS)
    # 对张量z做合法性断言校验：判断张量全部元素都是有限数值（不是NaN、无穷大、无穷小）
    assert torch.isfinite(z).all()
    # 对每个样本的潜向量算 L2 长度（欧氏范数）
    # 这个测试要抓的是：采出来的 z 还在 encoder 后验附近，还是 DDIM 末端把 打飞了。
    # 真实 μ 大约每维标准差 0.5，128 维时期望长度大约是 6
    norm = z.norm(dim=-1)
    # 真实 mu 的 |z| 约 6；爆炸时是 1e4+
    assert float(norm.mean()) < 20.0, f"|z|_mean={float(norm.mean()):.1f} 仍然爆炸"
    assert float(z.abs().max()) < 30.0, f"absmax={float(z.abs().max()):.1f} 仍然爆炸"


if __name__ == "__main__":
    test_ddim_z_stays_on_posterior_scale()
    print("ok")
