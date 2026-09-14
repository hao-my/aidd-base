"""
losses.py - VAE 损失：重构 + KL + 属性回归

总损失 = 重构损失 + beta * KL + lambda * 属性损失

三项各自的作用：
  - 重构损失：保证 z 解得回原分子，模型才算真的"看懂"了 SMILES 语法。
  - KL 项：把后验 q(z|x) 拉向先验 N(0, I)。有它，从先验随机采样的 z 才能解出合理分子；
           没有它，VAE 就退化成普通 AE，潜空间到处是空洞，采样出来全是垃圾。
  - 属性损失：从 z 回归 LogP，把亲脂性这条信息硬塞进潜空间（只在阶段2 启用）。

beta 的两个稳定技巧：
  - KL 退火（annealing）：训练初期 beta 从 0 慢慢升上去。
    一上来就给足 KL 权重，模型会走捷径——直接把后验塌成先验（KL=0），
    重构完全靠解码器背分布，也就是 posterior collapse。
  - free bits：每一维 KL 低于阈值时不再施压，给潜空间保留一点"必须使用"的容量。
"""

import torch
import torch.nn.functional as F


def reconstruction_loss(logits, target, pad_id):
    """
    自回归重构损失（逐 token 交叉熵，忽略 padding 位置）。

    输入：
        logits: FloatTensor [B, L, V]，解码器每个位置对下一 token 的打分
        target: LongTensor  [B, L]，真实的下一 token
        pad_id: int，padding 的 token id，这些位置不计入损失
    输出：
        (loss_per_sample, token_accuracy)
            loss_per_sample: 标量张量，每条序列的平均交叉熵之和 / batch
            token_accuracy : float，非 padding 位置上的 token 预测准确率
    """
    vocab_size = logits.size(-1)
    flat_logits = logits.reshape(-1, vocab_size)
    flat_target = target.reshape(-1)

    # reduction='sum' 先求batch个分子的总损失后再除以 batch：得到"每条分子"的重构代价。
    # 若用 'mean'（按 token 平均），长分子和短分子的权重会被拉平，
    # 而 KL 是按分子算的，两项量级对不上，beta 就很难调。
    loss_sum = F.cross_entropy(
        flat_logits, flat_target, ignore_index=pad_id, reduction="sum"
    )
    loss = loss_sum / logits.size(0)

    with torch.no_grad():    # 准确率只是监控数字，不需要梯度。关掉计算图，省内存，也不会误把准确率接到损失上。
        mask = flat_target != pad_id    # 非padding位置的mask  [B, L]
        correct = (flat_logits.argmax(dim=-1) == flat_target) & mask    # 非padding位置token的预测正确与否
        accuracy = correct.sum().item() / max(int(mask.sum().item()), 1)    # 非padding位置的预测正确率

    return loss, accuracy


def kl_divergence(mu, logvar, free_bits=0.0):
    """
    q(z|x)=N(mu, sigma^2) 与先验 p(z)=N(0, I) 之间的 KL 散度。

    闭式解：KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    输入：
        mu, logvar: FloatTensor [B, latent_dim]
        free_bits : float，每一维 KL 的下限。该维 KL 小于此值时不再继续压缩，
                    防止潜空间被压成纯噪声（posterior collapse）
    输出：
        (kl_for_backward, kl_raw)
            kl_for_backward: 标量张量，加了 free bits 后用于反传的 KL
            kl_raw         : float，未做 free bits 的真实 KL（只用于日志监控）
    """
    # 先按维度算，不要一次求和：free bits 是逐维施加的
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())          # [B, latent_dim]

    kl_raw = kl_per_dim.sum(dim=1).mean().item()

    if free_bits > 0:
        # 对每一维在 batch 上取平均后再做 clamp，比逐样本 clamp 更稳定
        kl_per_dim = torch.clamp(kl_per_dim.mean(dim=0), min=free_bits)
        kl = kl_per_dim.sum()
    else:
        kl = kl_per_dim.sum(dim=1).mean()

    return kl, kl_raw


def property_loss(pred, target):
    """
    属性回归损失（归一化 LogP 的 MSE）。

    输入：
        pred  : FloatTensor [B, 1]，property_head 的输出
        target: FloatTensor [B, 1]，归一化后的真实 LogP
    输出：
        标量张量
    """
    return F.mse_loss(pred, target)


def kl_weight_schedule(epoch, total_anneal_epochs, max_weight):
    """
    线性 KL 退火：epoch 0 时权重为 0，到 total_anneal_epochs 时升满。

    输入：
        epoch              : int，当前 epoch（从 0 开始）
        total_anneal_epochs: int，退火所占的 epoch 数；<=0 表示不退火，直接给满权重
        max_weight         : float，退火结束后的 beta
    输出：
        float，当前 epoch 应使用的 beta
    """
    if total_anneal_epochs <= 0:
        return max_weight
    return max_weight * min(1.0, (epoch + 1) / total_anneal_epochs)


def vae_loss(outputs, batch, pad_id, beta, free_bits=0.0, prop_weight=0.0):
    """
    汇总三项损失。

    输入：
        outputs    : dict，SmilesVAE.forward 的返回值
        batch      : dict，collate_fn 产出的 batch（需含 dec_target、cond）
        pad_id     : int
        beta       : float，KL 权重（由 kl_weight_schedule 给出）
        free_bits  : float，见 kl_divergence
        prop_weight: float，属性损失权重；0 表示不启用（阶段1）
    输出：
        (total_loss, log_dict)
            total_loss: 标量张量，用于 backward
            log_dict  : dict[str, float]，各项数值，供进度条与曲线记录
    """
    # 计算分子重构损失和教师强制的token准确率
    recon, acc = reconstruction_loss(outputs["logits"], batch["dec_target"], pad_id)
    # 计算KL散度
    kl, kl_raw = kl_divergence(outputs["mu"], outputs["logvar"], free_bits=free_bits)

    total = recon + beta * kl
    log = {"recon": recon.item(), "kl": kl_raw, "acc": acc, "beta": beta, "prop": 0.0}

    if prop_weight > 0:
        p_loss = property_loss(outputs["prop_pred"], batch["cond"])
        total = total + prop_weight * p_loss
        log["prop"] = p_loss.item()

    log["total"] = total.item()
    return total, log
