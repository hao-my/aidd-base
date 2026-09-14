"""
训练脚本 - 训练基于模板填空的 GAN
超参见 config.py。
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm    # 进度条库。给可迭代对象包一层后，终端会显示进度、速度、剩余时间等。

import config
import data
from data import get_dataloader
from model import MoleculeGAN

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, QED
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    print("警告: RDKit 未安装，无法生成分子图片")
    RDKIT_AVAILABLE = False


def visualize_molecules(smiles_list, save_path, mols_per_row=5):
    """将 有效SMILES 画成图并保存， 函数返回有效分子个数。"""
    if not RDKIT_AVAILABLE:
        return 0

    mols, valid_smiles = [], []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append(mol)
                valid_smiles.append(smiles)
        except Exception:
            pass

    valid_count = len(mols)
    if valid_count == 0:
        print(f"  - 有效分子: 0/{len(smiles_list)} (0.0%) - 跳过绘制")
        return 0

    try:
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=mols_per_row,
            subImgSize=(300, 300),
            legends=valid_smiles,
            returnPNG=False,
        )
        img.save(save_path)
        print(f"  - 分子图片已保存: {save_path}")
        print(f"  - 有效分子: {valid_count}/{len(smiles_list)} ({valid_count / len(smiles_list) * 100:.1f}%)")
    except Exception as e:
        print(f"  错误: 绘制分子失败 - {e}")
    return valid_count


def calculate_qed_scores(smiles_list):
    """只对 RDKit 能解析的分子算 QED(药物相似性定量估计,越接近1越像药,更符合常见药物性质组合)。"""
    if not RDKIT_AVAILABLE:
        return []
    scores = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                scores.append(QED.qed(mol))
        except Exception:
            pass
    return scores


def plot_qed_distribution(generated_qeds, real_qeds, save_path):
    '''绘制QED分布图,并保存。真实分子的QED分布用红色,生成分子的QED分布用蓝色。'''
    plt.figure(figsize=(8, 6))
    if real_qeds:
        plt.hist(real_qeds, bins=20, alpha=0.7, label=f"Real (n={len(real_qeds)})",
                 color="red", edgecolor="black", density=True)
    if generated_qeds:
        plt.hist(generated_qeds, bins=20, alpha=0.7, label=f"Generated (n={len(generated_qeds)})",
                 color="blue", edgecolor="black", density=True)
    plt.xlabel("QED Score")
    plt.ylabel("Density")
    plt.title("QED Distribution Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    stats_text = ""
    if generated_qeds:
        stats_text += f"Generated: μ={np.mean(generated_qeds):.3f}, σ={np.std(generated_qeds):.3f}\n"
    if real_qeds:
        stats_text += f"Real: μ={np.mean(real_qeds):.3f}, σ={np.std(real_qeds):.3f}"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, va="top", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  - QED distribution plot saved: {save_path}")


def _checkpoint_dict(gan, optimizer_g, optimizer_d, epoch, extra=None):
    '''把当前训练状态打包成一个可保存到 checkpoint 的字典,方便后续恢复训练。'''
    payload = {
        "epoch": epoch,
        "generator_state_dict": gan.generator.state_dict(),
        "discriminator_state_dict": gan.discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "num_templates": data.NUM_TEMPLATES,
        "num_substituents": data.NUM_SUBSTITUENTS,
        "max_r_count": data.MAX_R_COUNT,
        "noise_dim": gan.noise_dim,
        # 片段库必须一起存，生成时 decode 依赖同一套索引
        "templates": data.TEMPLATES,
        "substituents": data.SUBSTITUENTS,
        "template_r_counts": data.TEMPLATE_R_COUNTS,
        "g_hidden_dims": config.G_HIDDEN_DIMS,
        "g_dropout": config.G_DROPOUT,
        "d_template_emb": config.D_TEMPLATE_EMB,
        "d_substituent_emb": config.D_SUBSTITUENT_EMB,
        "d_hidden_dims": config.D_HIDDEN_DIMS,
        "d_dropout": config.D_DROPOUT,
    }
    if extra:
        payload.update(extra)
    return payload


def train_gan():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print(f"数据: {config.DATASET_FILE}")
    print(f"MAX_MOLECULES: {config.MAX_MOLECULES}")

    print("=" * 60)
    print("加载真实分子数据...")

    # dataset对每个样本返回一个张量二元组，储存母核库中对应的索引 和 取代基库中对应的索引列表（有pad填充）
    # dataloader 传入batch_size，返回一个迭代器, dataset形状整齐默认堆叠方式
    dataloader, dataset = get_dataloader(
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        dataset_file=config.DATASET_FILE,
        max_molecules=config.MAX_MOLECULES,
        seed=config.RANDOM_SEED,
    )

    gan = MoleculeGAN(
        num_templates=data.NUM_TEMPLATES,
        num_substituents=data.NUM_SUBSTITUENTS,
        max_r_count=data.MAX_R_COUNT,
        noise_dim=config.NOISE_DIM,
        device=device,
        g_hidden_dims=config.G_HIDDEN_DIMS,
        g_dropout=config.G_DROPOUT,
        d_template_emb=config.D_TEMPLATE_EMB,
        d_substituent_emb=config.D_SUBSTITUENT_EMB,
        d_hidden_dims=config.D_HIDDEN_DIMS,
        d_dropout=config.D_DROPOUT,
    )

    # betas(β₁=0.9, β₂=0.999)动量衰减系数，把 β₁ 从默认 0.9 改成 0.5，让一阶动量更“短视”，有时能减轻震荡、训练更稳
    optimizer_g = optim.Adam(gan.generator.parameters(), lr=config.LR_G, betas=(0.5, 0.999))  
    optimizer_d = optim.Adam(gan.discriminator.parameters(), lr=config.LR_D, betas=(0.5, 0.999))
    criterion_bce = nn.BCELoss()

    print("\n" + "=" * 60)
    print("计算真实分子的 QED 分布...")
    real_qeds = []
    if RDKIT_AVAILABLE:
        real_smiles_remake = []
        sample_size = min(2000, len(dataset))
        for i in range(sample_size):
            t_idx, s_idx = dataset.samples[i]     # 获取dataset中前2000个smiles的母核索引和取代基索引列表
            real_smiles_remake.append(dataset.decode(t_idx, s_idx))     # 将母核索引和取代基索引列表解码为smiles
        real_qeds = calculate_qed_scores(real_smiles_remake)     # 计算真实分子重组后的QED药物相似性定量估计分数做基线，和生成分子对比看效果
        print(f"  - Valid molecules: {len(real_qeds)}/{len(real_smiles_remake)}")
        if real_qeds:
            print(f"  - Real QED: mean={np.mean(real_qeds):.3f}, std={np.std(real_qeds):.3f}")

    print("\n" + "=" * 60)
    print("开始训练...")
    g_losses, d_losses = [], []

    for epoch in range(config.EPOCHS):
        epoch_g_loss = epoch_d_loss = 0.0
        num_batches = 0
        # pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}")  #pbar 包好后的迭代器，用法和原来的 dataloader 一样

        for real_template_idx, real_sub_idx in dataloader:  # 遍历取这一批样本的母核索引和取代基索引列表
            batch_size_actual = real_template_idx.size(0)     # 获取当前batch的大小
            real_template_idx = real_template_idx.to(device)
            real_sub_idx = real_sub_idx.to(device)

            # ---------- 训练判别器，生成器参数保持不动----------
            optimizer_d.zero_grad()
            # 标签平滑策略：Label smoothing：真 0.9 / 假 0.1，避免判别器过度自信
            real_labels = torch.ones(batch_size_actual, 1, device=device) * 0.9
            fake_labels = torch.zeros(batch_size_actual, 1, device=device) + 0.1

            real_scores = gan.discriminator(real_template_idx, real_sub_idx, use_soft=False)
            d_loss_real = criterion_bce(real_scores, real_labels)   # 学习目标：认真

            noise = gan.sample_noise(batch_size_actual)  # 生成形状[B, noise_dim]的噪音张量
            fake_template_idx, fake_sub_idx = gan.generator.generate(noise)  # 调用生成器中的生成函数，得到离散索引张量[B]和[B, max_r_count]
            fake_scores = gan.discriminator(
                fake_template_idx.detach(), fake_sub_idx.detach(), use_soft=False   #假样本detach，切断通向G的梯度，也只更新G
            )
            d_loss_fake = criterion_bce(fake_scores, fake_labels)    # 学习目标：认假
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()  # 只用 D 的优化器改 D 的参数

            # ---------- 训练生成器，判别器参数保持不动----------
            optimizer_g.zero_grad()
            noise = gan.sample_noise(batch_size_actual)
            # 生成器forward过程，得到模板未归一化分数向量和取代基未归一化分数向量
            fake_template_logits, fake_sub_logits = gan.generator(noise)  
            # softmax在最后一维做归一化，得到模板概率分布和取代基概率分布
            fake_template_probs = torch.softmax(fake_template_logits / config.TEMPERATURE, dim=-1)
            fake_sub_probs = torch.softmax(fake_sub_logits / config.TEMPERATURE, dim=-1)
            # 将模板概率分布和取代基概率分布送入判别器，进行判断打分
            fake_scores = gan.discriminator(fake_template_probs, fake_sub_probs, use_soft=True)
            g_loss = criterion_bce(fake_scores, real_labels)       # 学习目标：以假乱真
            g_loss.backward()
            optimizer_g.step()   # 只用 G 的优化器改 G 的参数
            # 虽然前向会经过D（use_soft=True），g_loss.backward() 时梯度会算到 D 上，但没有人用这些梯度去改 D 的权重。

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1
            # pbar.set_postfix({          #tqdm 进度条右侧追加实时指标，方便看当前 batch 训练情况。
            #     "D_loss": f"{d_loss.item():.4f}",     #本 batch 判别器总损失
            #     "G_loss": f"{g_loss.item():.4f}",     #本 batch 生成器总损失
            #     "D(real)": f"{real_scores.mean().item():.2f}",     #本 batch 判别器对真实样本的平均得分
            #     "D(fake)": f"{fake_scores.mean().item():.2f}",     #本 batch 判别器对假样本的平均得分
            # })

        avg_g = epoch_g_loss / max(num_batches, 1)
        avg_d = epoch_d_loss / max(num_batches, 1)
        g_losses.append(avg_g)
        d_losses.append(avg_d)
        print(f"Epoch {epoch + 1}/{config.EPOCHS} - D_loss: {avg_d:.4f}, G_loss: {avg_g:.4f}")

        if (epoch + 1) % config.SAMPLE_INTERVAL == 0:    # 每隔10 epoch 存 checkpoint + 画分子/QED
            ckpt_path = config.CHECKPOINT_DIR / f"gan_epoch_{epoch + 1}.pt"
            torch.save(_checkpoint_dict(gan, optimizer_g, optimizer_d, epoch + 1), ckpt_path)
            print(f"  - 模型已保存: {ckpt_path}")

            print("  - 生成样本示例:")  # 每隔10 epoch 生成100个样本，并画图
            gan.generator.eval()
            generated_smiles = []
            with torch.no_grad():
                sample_noise = gan.sample_noise(100)  # 生成形状[100, noise_dim]的噪音张量
                sample_t, sample_s = gan.generator.generate(sample_noise)    # 得到离散索引张量[100]和[100, max_r_count]
                for i in range(100):
                    smiles = dataset.decode(sample_t[i], sample_s[i])   # 将离散索引张量解码为smiles
                    generated_smiles.append(smiles)
                    if i < 5:
                        print(f"    {i + 1}. {smiles}")     # 打印前5个样本的smiles
            gan.generator.train()

            if RDKIT_AVAILABLE:
                img_path = config.CHECKPOINT_DIR / f"molecules_epoch_{epoch + 1}.png"
                # 将 有效SMILES 画成图并保存， 函数返回有效分子个数
                visualize_molecules(generated_smiles, str(img_path), mols_per_row=10)
                # 计算有效分子QED药物相似性定量估计分数
                generated_qeds = calculate_qed_scores(generated_smiles)
                qed_path = config.CHECKPOINT_DIR / f"qed_distribution_epoch_{epoch + 1}.png"
                # 绘制QED分布图,并保存。真实分子的QED分布用红色,生成分子的QED分布用蓝色。
                plot_qed_distribution(generated_qeds, real_qeds, str(qed_path))

    final_path = config.CHECKPOINT_FINAL
    torch.save(_checkpoint_dict(gan, optimizer_g, optimizer_d, config.EPOCHS), final_path)
    print(f"\n最终模型已保存: {final_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss")
    plt.legend()
    plt.grid(True)
    loss_plot = config.OUTPUT_DIR / "loss_curve.png"
    plt.savefig(loss_plot)
    print(f"Loss curve saved: {loss_plot}")
    print("\nTraining completed!")


if __name__ == "__main__":
    torch.manual_seed(config.RANDOM_SEED)
    train_gan()
