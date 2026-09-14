"""
latent_diffusion.py - 在 CVAE 潜空间 z 上的条件 DDPM

VAE 负责把 SMILES 写成合法分子；这里只学 p(z | LogP)，用来换掉生成时的 N(0, I) 先验。
z 只有 128 维，去噪器是浅 MLP，训练和采样都比 SMILES Transformer 便宜得多。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def cosine_betas(num_timesteps, s=0.008):
    """cosine 噪声调度：事先定好的一张β表：从t=0 到 t=T，每一步加多少噪声、还留多少信号"""
    steps = num_timesteps + 1
    # np.linspace(start, stop, num) ：在[start, stop]内按元素数量创建张量
    x = np.linspace(0, num_timesteps, steps)
    # np.cos() 是 numpy余弦函数，返回数组逐元素计算余弦值。
    alphas_cumprod = np.cos(((x / num_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    # 归一化，使第一个元素为1
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-5, 0.999)


def sinusoidal_time_embedding(timesteps, dim):
    """标准 Transformer 正弦时间编码，timesteps: [B] long。"""
    half = dim // 2
    freqs = torch.exp(
        -np.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class LatentDenoiser(nn.Module):
    """ε_θ(z_t, t, c)：预测加在潜向量上的噪声。"""

    def __init__(self, latent_dim, hidden_dim=256, time_dim=64, cond_hidden=32):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(1, cond_hidden),
            nn.SiLU(),
            nn.Linear(cond_hidden, hidden_dim),
        )
        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t, t, cond):
        """
        z_t : [B, latent_dim]
        t   : [B] long
        cond: [B, 1]，归一化 LogP；全零表示空条件
        """
        t_emb = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim))
        c_emb = self.cond_mlp(cond)
        h = self.in_proj(z_t) + t_emb + c_emb
        return self.net(h)


class LatentDDPM(nn.Module):
    """条件潜空间 DDPM。对 encoder 给出的 mu 做标准化后再扩散。"""

    def __init__(self, latent_dim, hidden_dim=256, num_timesteps=1000):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps
        self.denoiser = LatentDenoiser(latent_dim, hidden_dim=hidden_dim)

        betas = cosine_betas(num_timesteps)
        alphas = 1.0 - betas
        # np.cumprod() 是 numpy累积乘积函数，返回数组逐元素累乘的结果。
        alphas_cumprod = np.cumprod(alphas)
        # self.register_buffer(name, tensor) ：把一个不需要训练、但属于模型状态的张量挂到模块上。
        self.register_buffer("betas", torch.from_numpy(betas).float())
        self.register_buffer("alphas", torch.from_numpy(alphas).float())
        self.register_buffer("alphas_cumprod", torch.from_numpy(alphas_cumprod).float())
        self.register_buffer("sqrt_alphas_cumprod", torch.from_numpy(np.sqrt(alphas_cumprod)).float())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.from_numpy(np.sqrt(1.0 - alphas_cumprod)).float(),
        )
        self.register_buffer("z_mean", torch.zeros(latent_dim))
        self.register_buffer("z_std", torch.ones(latent_dim))

    def set_latent_stats(self, mean, std):
        """用训练集 mu 的逐维均值/标准差，避免扩散直接打在未标定的后验上。"""
        # 构造时 z_mean 全 0、z_std 全 1，相当于假定 z ~ N(0,I)
        # 真实 mu 更瘦（诊断里 std≈0.5），不改尺子的话扩散是在未标定的后验上加噪
        mean = mean.detach().to(device=self.z_mean.device, dtype=self.z_mean.dtype)
        # 每个元素小于 0.001 的抬到 0.001
        std = torch.clamp(std.detach().to(device=self.z_std.device, dtype=self.z_std.dtype), min=1e-3)
        # copy_ 是原地覆盖：不换掉 z_mean 这个 buffer 对象，只把数值写进去。
        self.z_mean.copy_(mean)
        self.z_std.copy_(std)

    def normalize(self, z):
        return (z - self.z_mean) / self.z_std

    def denormalize(self, z_norm):
        return z_norm * self.z_std + self.z_mean

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            # 创建一个与x_0形状相同的标准正态分布随机张量
            noise = torch.randn_like(x_0)
        # 计算每个样本的加噪系数a和b，形状为[B, 1]
        a = self.sqrt_alphas_cumprod[t][:, None]
        b = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        # 返回加噪后的样本x_t和噪声noise，形状分别为[B, latent_dim]和[B, latent_dim]
        return a * x_0 + b * noise, noise

    def forward(self, z_mu, cond, cond_dropout=0.0):
        """
        训练：对标准化后的 mu 加噪并回归噪声。

        z_mu: [B, latent_dim]，encoder 的后验均值（未标准化）
        cond: [B, 1]，归一化 LogP
        """
        # ！！！！！！！！标准化后的 mu作为加噪起点，形状为[B, latent_dim]
        x_0 = self.normalize(z_mu)
        batch = x_0.size(0)
        cond = cond.view(batch, 1)
        # 在[0, num_timesteps) 之间概率均匀抽整数，作为每个样本的随机扩散时刻t
        # 不按 t=0→T 逐步走，而是每个样本随机抽一个 t，一步加上对应的噪声，让网络预测 ε。这样每个 epoch 都能覆盖整条时间轴。
        t = torch.randint(0, self.num_timesteps, (batch,), device=x_0.device)
        
        # 一步加噪，得到x_t和噪声noise，形状都为[B, latent_dim]
        x_t, noise = self.q_sample(x_0, t)

        # 如果 cond_dropout > 0，则以 cond_dropout 的概率将 cond 设置为全零
        if cond_dropout > 0:
            # rand 均匀分布随机数
            drop = torch.rand(batch, 1, device=x_0.device) < cond_dropout
            cond = torch.where(drop, torch.zeros_like(cond), cond)
        
        # 预测噪声，将时间步t 和加噪后的样本x_t 及条件cond 分别经MLP都变为[B, latent_dim]，相加后返回MSE损失
        pred = self.denoiser(x_t, t, cond)
        return F.mse_loss(pred, noise)

    def ddim_times(self, steps, device, alpha_bar_min=None):
        """
        从「alpha_bar 仍够大」的最大 t 递降到 0。

        Cosine 调度最后几十步 alpha_bar 会掉到 1e-9，DDIM 的 x0 公式
        除以 sqrt(alpha_bar) 会把 ε 的一点误差放大成 |x0|~1e4。
        起点仍满足 x_t ≈ 噪声，从 N(0,I) 初始化仍然成立。
        """
        # 规定alpha_bar下限，采样从大于该下限的最大t步开始（大约907步），避免爆炸后的X0再写回潜空间时打飞。
        alpha_bar_min = float(
            config.LD_ALPHA_BAR_MIN if alpha_bar_min is None else alpha_bar_min
        )
        # nonzero(as_tuple=False).flatten() ：返回非False元素的索引，并展平为一维张量
        usable = (self.alphas_cumprod >= alpha_bar_min).nonzero(as_tuple=False).flatten()
        # 找到最大的可用步数
        t_hi = int(usable[-1].item()) if len(usable) else self.num_timesteps - 1
        steps = max(2, int(steps))

        # 从安全起点t_hi 均匀降到 0，共 steps 个点（默认 50），并收成整数下标
        times = torch.linspace(t_hi, 0, steps, device=device).long()
        uniq, seen = [], set()
        # .long() 截断之后可能出现相邻重复，不是防乱序重复。
        for t in times.tolist():
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return torch.tensor(uniq, device=device, dtype=torch.long)

    @torch.no_grad()
    def sample(self, num_samples, cond_value, device, steps=None, return_trace=False):
        """
        DDIM 从噪声生成 z（已还原到 encoder mu 的尺度）。

        cond_value: float 或 [B, 1]，归一化 LogP
        steps     : 采样步数，默认 config / 训练步数
        """
        self.eval()
        steps = int(steps or self.num_timesteps)
        # 把预测的x_0 夹在[-8,8]，防止 Cosine 末端除法再把 z 打飞。
        x0_clip = float(config.LD_X0_CLIP)

        # 如果 cond_value 不是张量，转成张量
        if not isinstance(cond_value, torch.Tensor):
            cond = torch.full((num_samples, 1), float(cond_value), device=device)
        else:
            cond = cond_value.to(device).view(-1, 1).float()
            num_samples = cond.size(0)

        # 造 DDIM 要走的时刻表，形状为[steps]
        times = self.ddim_times(steps, device)

        # 随机初始化潜向量x，形状为[num_samples, latent_dim]
        x = torch.randn(num_samples, self.latent_dim, device=device)
        trace = []

        # 遍历时刻表，逐步去噪
        for i, t_val in enumerate(times):
            t = torch.full((num_samples,), int(t_val.item()), device=device, dtype=torch.long)
            # 预测噪声，形状为[B, latent_dim]
            eps = self.denoiser(x, t, cond)

            # 计算每个样本的加噪系数a和b，形状为[B, 1]
            alpha_bar = self.alphas_cumprod[t].unsqueeze(-1).clamp(min=1e-5)

            # 反推去噪后的x0，并截断
            # 这里要除以alpha_bar，如果直接从最后一步开始去噪，此时alpha_bar极小会把 ε 的一点误差放大成 |x0|~1e4。
            x0_pred = (x - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
            x0_pred = x0_pred.clamp(-x0_clip, x0_clip)

            # 如果return_trace为True，则记录每一步的x_norm, x0_norm, x0_absmax, eps_norm
            if return_trace:
                trace.append({
                    "i": i,
                    "t": int(t_val.item()),
                    "x_norm": float(x.norm(dim=-1).mean()),
                    "x0_norm": float(x0_pred.norm(dim=-1).mean()),
                    "x0_absmax": float(x0_pred.abs().max()),
                    "eps_norm": float(eps.norm(dim=-1).mean()),
                })

            # 如果i是最后一步，则直接返回x0_pred
            # times是从起点降到0，50-0只走50步，但是有51个时间状态，所以要减1
            if i == len(times) - 1:
                x = x0_pred
                break
            
            t_prev = int(times[i + 1].item())
            # 取前一时刻的alpha_bar（越靠前越大）
            alpha_bar_prev = self.alphas_cumprod[t_prev].view(1, 1)
            # 重新加噪为噪声，但是x_0的权重升高，真实信号更多；噪声权重降低，噪声更少。
            # 普通 DDPM 去噪器并没有被训成「看一眼纯噪声就吐出正确 x_0」,只能猜个大概
            # 用预测的x0 和 eps 重新加噪为噪声，得到新的x， 是「沿着刚才那条猜测，走到一个更干净的等级」
            # 第一步决定地图上的大区；后面每步是在缩小的噪声下重画同一张图，不是从头再画一张。
            # 最终从一个脏的起点，一步步走到更干净的终点。
            x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1.0 - alpha_bar_prev) * eps
            
        z = self.denormalize(x)
        return (z, trace) if return_trace else z


def load_latent_ddpm(ckpt_path, device):
    """从 checkpoint 还原 LatentDDPM。"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = LatentDDPM(
        latent_dim=ckpt["latent_dim"],
        hidden_dim=ckpt.get("hidden_dim", 256),
        num_timesteps=ckpt["num_timesteps"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
