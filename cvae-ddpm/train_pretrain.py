"""
train_pretrain.py - 阶段1：在 ZINC 上做无条件 VAE 预训练

目的：
    先用大规模无标签 ZINC 让模型学会 SMILES 语法和一般药物化学分布，
    阶段2 再在同一批分子上用 LogP 做条件微调。这是典型的预训练-微调范式。

关键设计：
    - 条件通道在这一阶段恒为空条件（cond=None -> 全零向量）。
      模型结构里已经预留了条件模块，所以阶段2 可以直接 load 本阶段权重，不会缺 key。
    - 词表只用 ZINC 构建。阶段2 还是这批分子，不会出现"微调数据里有预训练没见过的 token"。
    - 每个 epoch 结束都从先验采样一批分子看有效率。重构 loss 低但采样全废，
      是后验坍缩 posterior collapse 的典型信号，只看 loss 发现不了。

运行：
    python train_pretrain.py               # 默认配置（config.py）
    python train_pretrain.py --smoke       # 极小规模，只验证流程能跑通
    python train_pretrain.py --max-molecules 200000 --epochs 20
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import config
from datasets import SmilesDataset, load_zinc, make_collate_fn
from losses import kl_weight_schedule, vae_loss
from model import SmilesVAE
from tokenizer import build_or_load_vocab
from utils import (count_parameters, get_device, plot_training_curves,
                   quick_sample_report, set_seed)


def parse_args():
    """
    解析命令行参数，未指定的项回退到 config.py 的默认值。

    输入：
        无（从 sys.argv 读）
    输出：
        argparse.Namespace
    """
    p = argparse.ArgumentParser(description="阶段1：ZINC 无条件 VAE 预训练")
    p.add_argument("--epochs", type=int, default=config.PRETRAIN_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.PRETRAIN_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.PRETRAIN_LR)
    p.add_argument("--max-molecules", type=int, default=config.PRETRAIN_MAX_MOLECULES,
                   help="最多使用多少条 ZINC 分子，-1 表示全量")
    p.add_argument("--kl-weight", type=float, default=config.PRETRAIN_KL_WEIGHT)
    p.add_argument("--kl-anneal-epochs", type=int, default=config.PRETRAIN_KL_ANNEAL_EPOCHS)
    p.add_argument("--smoke", action="store_true",
                   help="冒烟模式：2000 条分子、2 个 epoch，只验证流程")
    p.add_argument("--rebuild-vocab", action="store_true", help="强制重建词表")
    return p.parse_args()


def run_epoch(model, loader, optimizer, device, beta, pad_id, train=True):
    """
    跑一个 epoch。

    输入：
        model    : SmilesVAE
        loader   : DataLoader
        optimizer: torch.optim.Optimizer | None，train=False 时可为 None
        device   : torch.device
        beta     : float，本 epoch 的 KL 权重
        pad_id   : int
        train    : bool，True 为训练（更新参数），False 为验证
    输出：
        dict[str, float]，本 epoch 各项损失/指标的平均值
    """
    model.train() if train else model.eval()
    totals = {"total": 0.0, "recon": 0.0, "kl": 0.0, "acc": 0.0}
    n_batches = 0

    desc = "train" if train else "valid"
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:       
        # batch是collate_fn返回的字典，包含input_ids, lengths, dec_input, dec_target, cond
        batch = {k: v.to(device) for k, v in batch.items()}   # 将batch中的所有张量移动到指定设备

        with torch.set_grad_enabled(train):   # train=True时，允许计算梯度；train=False时，禁止计算梯度
            # 前向传播返回字典，得到logits, mu, logvar, z, prop_pred
            out = model(batch["input_ids"], batch["lengths"], batch["dec_input"], cond=None)
            # 阶段1 不用属性损失：这一步还没有 LogP 条件
            # 计算重建+KL散度总损失 和 total_loss, recon, kl, acc, beta, prop各项指标的日志字典
            loss, log = vae_loss(out, batch, pad_id, beta=beta,
                                 free_bits=config.KL_FREE_BITS, prop_weight=0.0)

        if train:
            optimizer.zero_grad()
            loss.backward()
            # RNN 上梯度爆炸很常见，裁剪是必需项而非可选项
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # 累加每个batch的损失和指标
        for k in totals:
            totals[k] += log[k]
        n_batches += 1
        pbar.set_postfix({"recon": f"{log['recon']:.1f}", "kl": f"{log['kl']:.2f}",
                          "acc": f"{log['acc']:.3f}"})

    # 返回一个epoch内，平均每个batch的损失和指标字典
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def main():
    """
    主流程：建词表 -> 建数据集 -> 训练 -> 存权重与曲线。

    输入：
        无
    输出：
        无（副作用：写 vocab.json / vae_pretrained.pt / pretrain_curves.png）
    """
    args = parse_args()
    set_seed(config.RANDOM_SEED)
    device = get_device()
    config.CKPT_DIR.mkdir(parents=True, exist_ok=True)

    max_molecules = None if args.max_molecules == -1 else args.max_molecules
    epochs = args.epochs
    if args.smoke:
        max_molecules, epochs = 2000, 2
        print("[pretrain] 冒烟模式：2000 分子 / 2 epoch")

    print("=" * 70)
    print(f"阶段1：ZINC 无条件 VAE 预训练 | device={device}")
    print("=" * 70)

    # ---------- 数据 ----------
    zinc_smiles = load_zinc(config.ZINC_FILE, max_molecules=max_molecules,
                            seed=config.RANDOM_SEED)

    tokenizer = build_or_load_vocab(
        config.VOCAB_FILE,
        smiles_for_build=zinc_smiles,
        max_len=config.MAX_LEN,
        rebuild=args.rebuild_vocab,
    )

    # 输入smiles的索引idx，返回对应的ids（未pad）和归一化后的条件值（无条件时为0.0）
    dataset = SmilesDataset(zinc_smiles, tokenizer, conditions=None)
    # 验证集只起监控作用——看重构损失和 token 准确率有没有在降、有没有过拟合
    n_valid = max(int(len(dataset) * 0.02), 1)
    # 用 PyTorch 的 random_split，把完整数据集随机切成训练集和验证集。generator参数设置随机种子，保证每次运行结果相同
    train_dataset, valid_dataset = random_split(
        dataset, [len(dataset) - n_valid, n_valid],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED),
    )

    # collate 函数，将batch内的样本统一补齐到最长长度，用pad_id填充
    # 返回值是一个字典，键为：
    #     input_ids : LongTensor [B, L]   pad后的ids（含 sos/eos），编码器输入
    #     dec_input : LongTensor [B, L-1] 教师强制的解码器输入（去掉最后一个 token）
    #     dec_target: LongTensor [B, L-1] 解码器目标（去掉第一个 token，即右移一位）
    #     lengths   : LongTensor [B]      每条序列的真实长度，用于 pack_padded_sequence
    #     cond      : FloatTensor [B, 1]  归一化条件值
    collate = make_collate_fn(tokenizer.pad_id)   
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate)
    print(f"[pretrain] 训练集 {len(train_dataset)} / 验证集 {len(valid_dataset)} | 词表 {len(tokenizer)}")

    # ---------- 模型 ----------
    model = SmilesVAE(
        vocab_size=len(tokenizer),
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        dim_feedforward=config.DIM_FEEDFORWARD,
        num_encoder_layers=config.ENCODER_LAYERS,
        num_decoder_layers=config.DECODER_LAYERS,
        latent_dim=config.LATENT_DIM,
        cond_dim=config.COND_EMB_DIM,
        dropout=config.DROPOUT,
        # 位置嵌入表比最长序列多留一点余量，生成时前面还要占一个 <sos>
        max_len=config.MAX_LEN + 8,
        pad_id=tokenizer.pad_id,
        sos_id=tokenizer.sos_id,
        eos_id=tokenizer.eos_id,
    ).to(device)
    print(f"[pretrain] 参数量 {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"recon": [], "kl": [], "acc": [], "validity": []}
    best_score = -1.0
    start = time.time()

    for epoch in range(epochs):
        # 线性 KL 退火：epoch 0 时 KL权重为 0，损失几乎全部来自重构。让模型先学会写合法SMILES
        # 再慢慢增加KL权重，让潜空间往N(0, I)靠拢。
        beta = kl_weight_schedule(epoch, args.kl_anneal_epochs, args.kl_weight)
        # 返回一个epoch内，每个batch的平均损失和指标的字典，包括total总损失，recon, kl, acc（token预测准确率）
        train_log = run_epoch(model, train_loader, optimizer, device, beta,
                              tokenizer.pad_id, train=True)
        valid_log = run_epoch(model, valid_loader, None, device, beta,
                              tokenizer.pad_id, train=False)
        # 从先验采样生成一批分子，做质量快检：这才是生成模型真正关心的指标！！！！！！！！
        # 返回一个字典，包括生成200个分子的validity有效率，unique唯一率，examples最多3个有效分子示例
        report = quick_sample_report(model, tokenizer, device, num=200,
                                     cond_value=None, max_len=config.MAX_LEN)

        history["recon"].append(valid_log["recon"])
        history["kl"].append(valid_log["kl"])
        history["acc"].append(valid_log["acc"])
        history["validity"].append(report["validity"])

        print(
            f"Epoch {epoch + 1}/{epochs} | beta={beta:.3f} "
            f"| train recon={train_log['recon']:.2f} "
            f"| valid recon={valid_log['recon']:.2f} kl={valid_log['kl']:.2f} "
            f"| valid acc={valid_log['acc']:.3f} "
            f"| 采样 validity={report['validity']:.3f} unique={report['unique']:.3f}"
        )
        if report["examples"]:
            print(f"           示例: {report['examples']}")

        # 用"有效率 + 重构准确率"联合挑最佳，单看其中一个都可能选到坏模型：！！！！！！！！！！！！！！
        # 只看有效率会选到 后验坍缩 后只会写有效、简单、重复分子的模型！！！！！！！！！！！！！！！
        # acc 是教师强制下的 token 准确率，只看准确率坏模型在教师强制下会背训练集，acc很高但z没进生成过程。
        # 一旦从先验采样自己生成，写错一个后面全崩，rdkit有效率接近0.只盯 acc 就会把这种「补全很强、生成全废」的权重存下来。
        score = report["validity"] + valid_log["acc"]
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model.config_dict(),
                    "epoch": epoch + 1,
                    "validity": report["validity"],
                    "token_acc": valid_log["acc"],
                    "vocab_file": str(config.VOCAB_FILE),
                },
                config.PRETRAIN_CKPT,
            )

    elapsed = (time.time() - start) / 60
    print(f"\n[pretrain] 完成，用时 {elapsed:.1f} 分钟，最佳权重已保存: {config.PRETRAIN_CKPT}")
    plot_training_curves(history, config.OUTPUT_DIR / "pretrain_curves.png",
                         title="阶段1 ZINC 无条件预训练")


if __name__ == "__main__":
    main()
