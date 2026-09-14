"""
train_conditional.py - 阶段2：在 ZINC 上用 RDKit LogP 做条件微调，把无条件 VAE 变成 CVAE

做什么：
    加载阶段1 在 ZINC 上学到的"怎么写合法分子"，用同一批分子的 Crippen LogP
    当条件，教模型"给定目标亲脂性时该写什么样的分子"。
    LogP 通过 cond_mlp 注入 encoder 和 decoder，同时用属性头从 z 回归归一化 LogP。
    评测时用 RDKit MolLogP 对齐目标，不用模型自己的属性头当最终指标。

为什么阶段1 的权重能直接加载：
    模型结构从一开始就带条件通路，阶段1 只是把条件恒置为零向量。
    所以这里 load_state_dict(strict=True) 不会缺 key，cond_mlp 与 property_head
    带着阶段1 的初始值（几乎没被训练过）进入阶段2 开始学习。

防过拟合：
    1. 学习率比预训练小一个量级（1e-4 vs 1e-3），避免把预训练知识冲掉。
    2. scaffold split 出验证集，按验证集指标挑最佳权重而不是按训练集。
    3. 可选 SMILES 枚举增强（--augment N），默认关闭。

运行：
    python train_conditional.py
    python train_conditional.py --smoke
    python train_conditional.py --epochs 12
"""

import argparse
import time

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from datasets import (SmilesDataset, load_zinc_with_logp, make_collate_fn,
                      normalize_cond, scaffold_split)
from losses import kl_weight_schedule, vae_loss
from model import SmilesVAE
from tokenizer import SMILESTokenizer
from utils import (count_parameters, get_device, plot_training_curves,
                   quick_sample_report, set_seed)


def parse_args():
    """
    解析命令行参数。

    输入：
        无
    输出：
        argparse.Namespace
    """
    p = argparse.ArgumentParser(description="阶段2：ZINC LogP 条件微调（CVAE）")
    p.add_argument("--epochs", type=int, default=config.FINETUNE_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.FINETUNE_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.FINETUNE_LR)
    p.add_argument("--kl-weight", type=float, default=config.FINETUNE_KL_WEIGHT)
    p.add_argument("--kl-anneal-epochs", type=int, default=config.FINETUNE_KL_ANNEAL_EPOCHS)
    p.add_argument("--prop-weight", type=float, default=config.PROPERTY_WEIGHT)
    p.add_argument("--cond-dropout", type=float, default=config.COND_DROPOUT)
    p.add_argument("--max-molecules", type=int, default=config.FINETUNE_MAX_MOLECULES,
                   help="最多使用多少条 ZINC 分子做条件微调")
    p.add_argument("--augment", type=int, default=0,
                   help="每个分子额外生成多少条随机（非规范）SMILES 做数据增强，0 = 关闭")
    p.add_argument("--from-scratch", action="store_true",
                   help="不加载阶段1 权重，直接在 ZINC LogP 上从零训练（用于做对照实验）")
    p.add_argument("--smoke", action="store_true", help="冒烟模式：3 个 epoch")
    return p.parse_args()


def enumerate_smiles(smiles, n, seed=42):
    """
    SMILES 枚举增强：把同一个分子写成若干条等价但不同写法的 SMILES。

    原理是从不同起始原子、不同遍历顺序输出 SMILES，分子本身不变。
    对小数据集来说，这相当于告诉模型"这些不同的字符串是同一个分子"，
    能显著扩大有效样本量。

    输入：
        smiles: str，规范 SMILES
        n     : int，额外生成的条数
        seed  : int
    输出：
        list[str]，去重后的非规范 SMILES（不含原始规范形式）
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or n <= 0:
        return []
    rng = np.random.RandomState(seed)
    out = set()
    # 多试几次：doRandom 可能反复给出同一种写法，或撞回原来的规范 SMILES
    for _ in range(n * 3):
        if len(out) >= n:
            break
        try:
            # canonical=False：不规范SMILES。起点、环编号、分支顺序都可以不同，但仍是同一个分子
            # doRandom=True：在非规范模式下，让 RDKit 随机选遍历方式
            alt = Chem.MolToSmiles(mol, canonical=False, doRandom=True,
                                   randomSeed=int(rng.randint(0, 10 ** 6)))
        except Exception:
            continue
        if alt and alt != smiles:
            out.add(alt)
    return list(out)


def build_split_data(smiles, values, augment, seed):
    """
    骨架划分 + 可选增强，返回训练/验证两套 (SMILES, LogP)。

    输入：
        smiles : list[str]
        values : np.ndarray，条件原值（当前为 RDKit LogP）
        augment: int，增强倍数
        seed   : int
    输出：
        (train_smiles, train_values, valid_smiles, valid_values)
        其中 valid_* 不做增强，保证验证指标干净可比
    """
    # 骨架划分，返回训练集和验证集的索引。骨架组内分子数多的先进训练集
    train_idx, valid_idx = scaffold_split(smiles, test_ratio=config.COND_TEST_RATIO, seed=seed)

    train_smiles = [smiles[i] for i in train_idx]
    train_values = [float(values[i]) for i in train_idx]

    # augment每个分子额外生成多少条随机（非规范）SMILES 做数据增强，相当于扩增训练集
    # 原样本过少容易过拟合标准拼写。
    # 生成时模型自己写的顺序往往不是规范式，和训练分布不一致，有效率、条件控制都会差
    # 教师强制每一步都在猜规范式的下一个字符，模型可能把规范式书写习惯当成化学本身
    if augment > 0:
        extra_s, extra_v = [], []
        for smi, val in zip(list(train_smiles), list(train_values)):
            # 使用RDkit对 SMILES 进行非规范SMILES 枚举增强，返回一个列表，列表元素为非规范SMILES
            for alt in enumerate_smiles(smi, augment, seed=seed):
                extra_s.append(alt)
                extra_v.append(val)
        print(f"[finetune] SMILES 枚举增强: {len(train_smiles)} -> {len(train_smiles) + len(extra_s)}")
        train_smiles += extra_s
        train_values += extra_v

    valid_smiles = [smiles[i] for i in valid_idx]
    valid_values = [float(values[i]) for i in valid_idx]
    return train_smiles, np.array(train_values), valid_smiles, np.array(valid_values)


def run_epoch(model, loader, optimizer, device, beta, pad_id, prop_weight,
              cond_dropout=0.0, train=True):
    """
    跑一个 epoch（条件版本：cond 来自 batch 而不是 None）。

    输入：
        model       : SmilesVAE
        loader      : DataLoader
        optimizer   : torch.optim.Optimizer | None
        device      : torch.device
        beta        : float，KL 权重
        pad_id      : int
        prop_weight : float，属性损失权重
        cond_dropout: float，条件 dropout 概率（只在 train=True 时生效）
        train       : bool
    输出：
        dict[str, float]，各指标的 epoch 平均值
    """
    model.train() if train else model.eval()
    totals = {"total": 0.0, "recon": 0.0, "kl": 0.0, "acc": 0.0, "prop": 0.0}
    n_batches = 0

    pbar = tqdm(loader, desc="train" if train else "valid", leave=False)
    for batch in pbar:
        # 每个batch返回一个字典，键为input_ids, dec_input, dec_target, lengths, cond归一化后的条件值
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):

            # 传入encoder，得到均值和log方差，重参数化采样，得到潜向量，传入decoder，得到logits，传入属性预测头，得到属性预测
            out = model(batch["input_ids"], batch["lengths"], batch["dec_input"],
                        cond=batch["cond"], cond_dropout=cond_dropout)
            
            # 计算 重构损失、KL散度、属性回归损失（条件生成）
            # 返回总损失 和 日志字典（total_loss, recon, kl, acc, beta, prop_loss）
            loss, log = vae_loss(out, batch, pad_id, beta=beta,
                                 free_bits=config.KL_FREE_BITS, prop_weight=prop_weight)
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        for k in totals:
            totals[k] += log[k]
        n_batches += 1
        pbar.set_postfix({"recon": f"{log['recon']:.1f}", "kl": f"{log['kl']:.2f}",
                          "prop": f"{log['prop']:.3f}"})

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def main():
    """
    主流程：加载阶段1 权重 -> ZINC LogP 条件微调 -> 存权重与曲线。

    输入：
        无
    输出：
        无（副作用：写 cvae_zinc_logp.pt / finetune_curves.png）
    """
    args = parse_args()
    set_seed(config.RANDOM_SEED)
    device = get_device()
    epochs = 3 if args.smoke else args.epochs

    print("=" * 70)
    print(f"阶段2：ZINC LogP 条件微调（CVAE）| device={device}")
    print("=" * 70)

    if not config.VOCAB_FILE.exists():
        raise FileNotFoundError(f"词表不存在: {config.VOCAB_FILE}，请先运行 train_pretrain.py")
    tokenizer = SMILESTokenizer.load(config.VOCAB_FILE)

    # ---------- 数据 ----------
    smiles, values = load_zinc_with_logp(
        config.ZINC_FILE, max_molecules=args.max_molecules, seed=config.RANDOM_SEED
    )
    train_s, train_v, valid_s, valid_v = build_split_data(
        smiles, values, args.augment, config.RANDOM_SEED
    )

    # 条件归一化：训练、生成、评测必须用同一组端点，端点存进 checkpoint 备查
    train_cond = normalize_cond(train_v, config.LOGP_MIN, config.LOGP_MAX)
    valid_cond = normalize_cond(valid_v, config.LOGP_MIN, config.LOGP_MAX)

    # 条件注入
    # Dataset中，每个样本包含 ids 和 cond（条件值）
    train_set = SmilesDataset(train_s, tokenizer, conditions=train_cond)
    valid_set = SmilesDataset(valid_s, tokenizer, conditions=valid_cond)

    # 创建一个collate函数，用于将batch中的样本合并成一个batch
    # 每个batch返回一个字典：
    # input_ids : LongTensor [B, L]  完整序列（含 sos/eos），送给 encoder
    # dec_input : LongTensor [B, L-1] 教师强制的解码器输入（去掉最后一个 token）
    # dec_target: LongTensor [B, L-1] 解码器目标（去掉第一个 token，即右移一位）
    # lengths   : LongTensor [B]      每条序列的真实长度，用于 pack_padded_sequence
    # cond      : FloatTensor [B, 1]  归一化条件值
    collate = make_collate_fn(tokenizer.pad_id)  
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, collate_fn=collate)
    print(f"[finetune] 训练集 {len(train_set)} / 验证集 {len(valid_set)}")

    # ---------- 模型 ----------
    if args.from_scratch:
        print("[finetune] 对照实验：不加载预训练权重，从零开始")
        model = SmilesVAE(
            vocab_size=len(tokenizer), d_model=config.D_MODEL,
            nhead=config.NHEAD, dim_feedforward=config.DIM_FEEDFORWARD,
            num_encoder_layers=config.ENCODER_LAYERS,
            num_decoder_layers=config.DECODER_LAYERS,
            latent_dim=config.LATENT_DIM, cond_dim=config.COND_EMB_DIM,
            dropout=config.DROPOUT, max_len=config.MAX_LEN + 8,
            pad_id=tokenizer.pad_id,
            sos_id=tokenizer.sos_id, eos_id=tokenizer.eos_id,
        ).to(device)
    else:
        if not config.PRETRAIN_CKPT.exists():
            raise FileNotFoundError(
                f"未找到阶段1 权重: {config.PRETRAIN_CKPT}，请先运行 train_pretrain.py"
            )
        

        '''
        从 vae_pretrained.pt 反序列化出一个 Python 字典 ckpt。阶段 1 保存时大致是：
        model_state：所有层的权重
        model_config：vocab_size、d_model、层数等，用来重建类
        还有epoch、validity 等日志字段
        '''
        # weights_only=False：允许文件里有完整 Python 对象（字典、路径字符串等），不只是纯张量。
        ckpt = torch.load(config.PRETRAIN_CKPT, map_location=device, weights_only=False)

        # config_dict() 存下来的超参解包进构造函数，先搭出和存档一模一样的网络（层数、宽度、词表大小等）。此时权重还是随机初始化。
        model = SmilesVAE(**ckpt["model_config"]).to(device)

        # 把 model_state 里每个参数名对应的数值，拷进当前模型。
        # strict=True：结构完全一致才允许加载，一旦哪里改了尺寸立刻报错而不是静默跳过
        model.load_state_dict(ckpt["model_state"], strict=True)
        print(
            f"[finetune] 已加载阶段1 权重（epoch {ckpt['epoch']}，"
            f"采样 validity={ckpt['validity']:.3f}，token acc={ckpt['token_acc']:.3f}）"
        )
    print(f"[finetune] 参数量 {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"recon": [], "kl": [], "acc": [], "prop": [], "validity": []}
    best_score = -1e9

    # 快检时用的目标条件：取分布中间的 LogP，看模型能不能在该条件下稳定产出分子
    probe_cond = float(normalize_cond(config.MAIN_TARGET_LOGP,
                                      config.LOGP_MIN, config.LOGP_MAX))
    # 得到验证集条件的方差cond_variance，下一步用来把属性 MSE 换算成 R^2（R^2 = 1 - MSE / Var）
    cond_variance = float(np.var(valid_cond)) or 1e-8   # 或1e-8防止分母为0
    start = time.time()

    for epoch in range(epochs):
        beta = kl_weight_schedule(epoch, args.kl_anneal_epochs, args.kl_weight)

        # 返回一个字典，记录一个epoch内的平均 总损失 重构损失、KL散度、教师强制的token准确率、属性回归损失
        # cond_dropout表示每个训练样本有 15% 概率把 cond_emb 换成全零，encoder/decoder用这些样本做无条件训练
        # 随机失活是训练技巧，用来逼模型别完全依赖条件、也别完全忽略条件。测能力时要把那条噪声关掉，否则测到的是「带噪声的前向」，不是部署时的前向。
        # prop_weight 属性损失权重。不加权重时属性项只占总损失很小一截，无法推动 z 去编码 LogP
        train_log = run_epoch(model, train_loader, optimizer, device, beta,
                              tokenizer.pad_id, args.prop_weight,          
                              cond_dropout=args.cond_dropout, train=True)
        
        # 验证集不做cond_dropout，和普通 Dropout 同一套逻辑：训练时故意加噪声，评估时看模型在真实输入上的表现
        valid_log = run_epoch(model, valid_loader, None, device, beta,
                              tokenizer.pad_id, args.prop_weight, train=False)

        # 快速采样报告，生成200个新样本，返回字典：能被RDKit解析的比例、有效分子中不重复的比例、最多3个有效分子示例
        report = quick_sample_report(model, tokenizer, device, num=200,
                                     cond_value=probe_cond, max_len=config.MAX_LEN)    # probe_cond 为归一化后的目标 LogP

        # 添加这个epoch验证集的 平均指标到history字典
        history["recon"].append(valid_log["recon"])
        history["kl"].append(valid_log["kl"])
        history["acc"].append(valid_log["acc"])
        history["prop"].append(valid_log["prop"])
        history["validity"].append(report["validity"])

        # 归一化之后属性回归损失valid_log["prop"]（MSE） 大约是 0.02 量级
        # 有效率和 token acc 都在 0–1。若把 MSE 原样加进去，0.02 相对 0.6 的 validity 等于没参与；换成 R^2 三项才在同一量级。
        prop_r2 = 1.0 - valid_log["prop"] / cond_variance

        print(
            f"Epoch {epoch + 1}/{epochs} | beta={beta:.3f} "
            f"| train recon={train_log['recon']:.2f} "
            f"| valid recon={valid_log['recon']:.2f} kl={valid_log['kl']:.2f} "
            f"acc={valid_log['acc']:.3f} propR2={prop_r2:.3f} "
            f"| 采样 validity={report['validity']:.3f} unique={report['unique']:.3f}"
        )

        # 挑权重的标准要贴近最终用途：既要能生成有效分子，也要属性头学得准。
        # 属性项换算成 R^2 再参与打分，这样三项都在 0~1 尺度上，不会被量级差异带偏
        # （属性 MSE 本身只有 0.02 量级，直接相减等于没参与选择）。
        score = report["validity"] + valid_log["acc"] + prop_r2
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model.config_dict(),
                    "epoch": epoch + 1,
                    "validity": report["validity"],
                    "token_acc": valid_log["acc"],
                    "prop_mse": valid_log["prop"],
                    "prop_r2": prop_r2,
                    "cond_min": config.LOGP_MIN,
                    "cond_max": config.LOGP_MAX,
                    "logp_min": config.LOGP_MIN,
                    "logp_max": config.LOGP_MAX,
                    "vocab_file": str(config.VOCAB_FILE),
                    # 存下训练集分子，评测 novelty 时直接用，避免两处划分不一致
                    "train_smiles": train_s,
                    "valid_smiles": valid_s,
                },
                config.CVAE_CKPT,
            )

    elapsed = (time.time() - start) / 60
    print(f"\n[finetune] 完成，用时 {elapsed:.1f} 分钟，最佳权重已保存: {config.CVAE_CKPT}")
    plot_training_curves(history, config.OUTPUT_DIR / "finetune_curves.png",
                         title="阶段2 ZINC LogP 条件微调")


if __name__ == "__main__":
    main()
