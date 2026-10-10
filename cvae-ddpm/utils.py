"""
utils.py - 各脚本共用的小工具：随机种子、设备、参数统计、生成质量快检、曲线绘图
"""

import random

import matplotlib
import numpy as np
import torch

# 训练时没有图形界面，必须先切到非交互后端，否则 savefig 在某些环境下会卡住
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# 中文标签在图里显示成方块的问题
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def set_seed(seed):
    """
    固定 python / numpy / torch 的随机种子，让实验可复现。

    输入：
        seed: int
    输出：
        无
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """
    选择可用设备。

    输入：
        无
    输出：
        torch.device，有 CUDA 用 cuda，否则 cpu
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    """
    统计可训练参数量。

    输入：
        model: nn.Module
    输出：
        int，参数个数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def canonical_or_none(smiles):
    """
    尝试用 RDKit 解析并规范化 SMILES。

    输入：
        smiles: str
    输出：
        str | None，无法解析时返回 None
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def quick_sample_report(model, tokenizer, device, num=200, cond_value=None,
                        max_len=128, temperature=1.0):
    """
    训练途中的快检：采一小批分子，看有效率和唯一率，判断模型有没有在往正确方向走。

    只看 loss 是不够的——重构 loss 很低但从先验采样全是废串，是 VAE 的常见失败模式，
    必须真的采样解析一次才知道。

    输入：
        model      : SmilesVAE
        tokenizer  : SMILESTokenizer
        device     : torch.device
        num        : int，采样条数
        cond_value : float | None，归一化后的目标条件
        max_len    : int，最大生成长度
        temperature: float，采样温度
    输出：
        dict：
            validity : float，能被 RDKit 解析的比例
            unique   : float，有效分子中不重复的比例
            examples : list[str]，最多 3 个有效分子示例
    """
    was_training = model.training
    model.eval()

    # mosel.sample返回生成样本的 ids（已去sos，包含pad） 张量，形状为[num=200, 128]
    ids = model.sample(num, max_len=max_len, device=device,
                       cond_value=cond_value, temperature=temperature)
    # tokenizer.decode将ids转换为smiles字符串列表，遇到 eos 结束，后面pad直接丢掉
    smiles = [tokenizer.decode(row.tolist()) for row in ids.cpu()]
    valid = [c for c in (canonical_or_none(s) for s in smiles) if c is not None]

    if was_training:
        model.train()

    return {
        "validity": len(valid) / max(num, 1),
        "unique": len(set(valid)) / max(len(valid), 1),
        "examples": list(dict.fromkeys(valid))[:3],
    }    # dict.fromkeys(valid)：将valid列表转换为字典，键为列表元素，值为None。然后取前3个键，转换为列表。


def plot_training_curves(history, save_path, title="训练曲线"):
    """
    画重构损失 / KL / token 准确率 / 采样有效率四联图。

    输入：
        history  : dict[str, list[float]]，键取自 recon / kl / acc / validity / prop 等
        save_path: str | Path，图片输出路径
        title    : str，总标题
    输出：
        无（副作用：保存 png）
    """
    panels = [
        ("recon", "重构损失 (每分子交叉熵)"),
        ("mse", "扩散噪声 MSE"),
        ("kl", "KL 散度"),
        ("acc", "token 重构准确率"),
        ("validity", "先验采样有效率"),
    ]
    available = [(k, name) for k, name in panels if history.get(k)]
    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (key, name) in zip(axes, available):
        epochs = range(1, len(history[key]) + 1)
        ax.plot(epochs, history[key], marker="o", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(alpha=0.3)

    # 属性损失只在阶段2 存在，单独叠到 KL 面板上不合适，这里作为标题补充
    if history.get("prop") and max(history["prop"]) > 0:
        title = f"{title} | 属性损失末值 {history['prop'][-1]:.4f}"

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[utils] 训练曲线已保存: {save_path}")
