"""
生成脚本 - 使用训练好的 GAN 生成新分子
"""

import random

import numpy as np
import pandas as pd
import torch

import config
import data
from model import MoleculeGAN

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    print("警告: RDKit 未安装")
    RDKIT_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.pyplot as plt

    TSNE_AVAILABLE = True
except ImportError:
    print("警告: sklearn 未安装，无法 t-SNE")
    TSNE_AVAILABLE = False


def restore_fragment_library(checkpoint):
    """从 checkpoint 恢复片段库（decode 必须用训练时同一套索引）。"""
    if "templates" not in checkpoint or "substituents" not in checkpoint:
        raise RuntimeError("checkpoint 中没有片段库，请用本项目 train.py 重新训练")
    data.TEMPLATES = checkpoint["templates"]
    data.SUBSTITUENTS = checkpoint["substituents"]
    data.TEMPLATE_R_COUNTS = checkpoint["template_r_counts"]
    data.NUM_TEMPLATES = len(data.TEMPLATES)
    data.NUM_SUBSTITUENTS = len(data.SUBSTITUENTS)
    data.MAX_R_COUNT = checkpoint["max_r_count"]
    print("✓ 已从 checkpoint 恢复片段库:")
    print(f"  - 模板数量: {data.NUM_TEMPLATES}")
    print(f"  - 取代基数量: {data.NUM_SUBSTITUENTS}")
    print(f"  - 最大取代位点: {data.MAX_R_COUNT}")


def decode(template_idx, substituent_indices):
    if isinstance(template_idx, torch.Tensor):
        template_idx = template_idx.item()
    if isinstance(substituent_indices, torch.Tensor):
        substituent_indices = substituent_indices.cpu().numpy().tolist()
    smiles = data.TEMPLATES[template_idx]
    r_count = data.TEMPLATE_R_COUNTS[template_idx]
    for i in range(r_count):
        smiles = smiles.replace(f"{{R{i + 1}}}", data.SUBSTITUENTS[substituent_indices[i]])
    return smiles


def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    if not RDKIT_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except Exception:
        return None


def visualize_chemical_space(generated_smiles, real_smiles, save_path):
    if not RDKIT_AVAILABLE or not TSNE_AVAILABLE:
        print("跳过 t-SNE（需要 RDKit 和 sklearn）")
        return

    print("\n" + "=" * 60)
    print("计算分子指纹...")
    generated_fps, generated_valid = [], []
    for smi in generated_smiles:
        fp = smiles_to_fingerprint(smi)
        if fp is not None:
            generated_fps.append(fp)
            generated_valid.append(smi)
    real_fps = []
    for smi in real_smiles:
        fp = smiles_to_fingerprint(smi)
        if fp is not None:
            real_fps.append(fp)
    print(f"  - Generated valid fingerprints: {len(generated_fps)}/{len(generated_smiles)}")
    print(f"  - Real valid fingerprints: {len(real_fps)}/{len(real_smiles)}")
    if len(generated_fps) == 0 or len(real_fps) == 0:
        print("没有足够的有效分子，跳过 t-SNE")
        return

    generated_fps_array = np.array(generated_fps)
    avg_gen_similarity = None
    if len(generated_fps) > 1:
        gen_similarity = cosine_similarity(generated_fps_array)   # cosine_similarity计算生成分子指纹之间的余弦相似性度矩阵
        mask = np.ones(gen_similarity.shape, dtype=bool)  # 创建一个与gen_similarity形状相同的布尔掩码，初始值为True
        np.fill_diagonal(mask, False)   # 对角线改成 False（排除“自己和自己”，相似度恒为 1）
        avg_gen_similarity = gen_similarity[mask].mean()   #只对非对角元素求平均 → 生成分子平均两两相似度
        print(f"  - 生成分子平均指纹相似度: {avg_gen_similarity:.3f}")
        if avg_gen_similarity > 0.95:
            print("  ⚠️ 警告: 相似度极高，可能 mode collapse")
        elif avg_gen_similarity > 0.85:
            print("  ⚠️ 警告: 相似度较高，多样性不足")

    all_fps = np.array(generated_fps + real_fps)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_fps) - 1))    # 使用t-SNE将指纹降维到2维
    embeddings = tsne.fit_transform(all_fps)    # 将指纹矩阵转换为2维坐标，方便可视化
    gen_emb = embeddings[: len(generated_fps)]    # 生成分子坐标
    real_emb = embeddings[len(generated_fps) :]    # 真实分子坐标

    plt.figure(figsize=(10, 8))
    plt.scatter(real_emb[:, 0], real_emb[:, 1], c="blue", alpha=0.5, s=30,
                label=f"Real (n={len(real_fps)})", edgecolors="none")
    plt.scatter(gen_emb[:, 0], gen_emb[:, 1], c="red", alpha=0.5, s=30,
                label=f"Generated (n={len(generated_fps)})", edgecolors="none")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title("Chemical Space (Morgan + t-SNE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if avg_gen_similarity is not None:
        plt.text(0.02, 0.02, f"Generated avg similarity: {avg_gen_similarity:.3f}",
                 transform=plt.gca().transAxes, va="bottom", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  - t-SNE saved: {save_path}")


def generate_molecules(checkpoint_path, num_molecules, device="cpu", save_image=True, image_path=None):
    '''
    加载模型
    根据参数num_molecules，生成num_molecules个smiles分子
    如果save_image为True，对前50个有效分子画图并保存
    返回smiles列表
    '''
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    restore_fragment_library(checkpoint)

    gan = MoleculeGAN(
        num_templates=checkpoint["num_templates"],
        num_substituents=checkpoint["num_substituents"],
        max_r_count=checkpoint["max_r_count"],
        noise_dim=checkpoint.get("noise_dim", config.NOISE_DIM),
        device=device,
        g_hidden_dims=checkpoint.get("g_hidden_dims", config.G_HIDDEN_DIMS),
        g_dropout=checkpoint.get("g_dropout", config.G_DROPOUT),
        d_template_emb=checkpoint.get("d_template_emb", config.D_TEMPLATE_EMB),
        d_substituent_emb=checkpoint.get("d_substituent_emb", config.D_SUBSTITUENT_EMB),
        d_hidden_dims=checkpoint.get("d_hidden_dims", config.D_HIDDEN_DIMS),
        d_dropout=checkpoint.get("d_dropout", config.D_DROPOUT),
    )
    gan.generator.load_state_dict(checkpoint["generator_state_dict"])
    gan.generator.eval()

    print(f"\n生成 {num_molecules} 个分子...\n")
    smiles_list = []
    with torch.no_grad():
        noise = gan.sample_noise(num_molecules)
        template_indices, sub_indices = gan.generator.generate(noise)
        for i in range(num_molecules):
            smiles = decode(template_indices[i], sub_indices[i])
            smiles_list.append(smiles)
            if i < 20:
                print(f"{i + 1:2d}. 模板{template_indices[i].item():2d} | {smiles}")

    if RDKIT_AVAILABLE:
        valid_mols, valid_smiles, valid_count = [], [], 0
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_mols.append(mol)
                valid_smiles.append(smiles)
                valid_count += 1
        print(f"\n有效率: {valid_count}/{num_molecules} ({valid_count / num_molecules * 100:.1f}%)")
        if save_image and valid_count > 0 and image_path is not None:
            try:
                img = Draw.MolsToGridImage(
                    valid_mols[:50],
                    molsPerRow=5,
                    subImgSize=(300, 300),
                    legends=valid_smiles[:50],
                    returnPNG=False,
                )
                img.save(image_path)
                print(f"✓ 图片已保存: {image_path}")
            except Exception as e:
                print(f"✗ 保存图片失败: {e}")
    return smiles_list


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}\n")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = config.CHECKPOINT_FINAL
    if not ckpt.exists():
        raise FileNotFoundError(f"找不到 {ckpt}，请先运行 python train.py")

    smiles_list = generate_molecules(
        checkpoint_path=str(ckpt),
        num_molecules=config.NUM_GENERATE,
        device=device,
        save_image=True,
        image_path=str(config.OUTPUT_DIR / "generated_molecules.png"),
    )

    csv_path = config.OUTPUT_DIR / "generated_molecules.csv"
    pd.DataFrame({"smiles": smiles_list}).to_csv(csv_path, index=False)
    unique_smiles = set(smiles_list)
    print(f"  - 已保存 {len(smiles_list)} 条到 {csv_path}")
    print(f"  - 唯一分子数: {len(unique_smiles)}/{len(smiles_list)} "
          f"({len(unique_smiles) / max(len(smiles_list), 1) * 100:.1f}%)")
    if len(unique_smiles) < len(smiles_list) * 0.5:
        print("  ⚠️ 警告: 重复率高，可能 mode collapse")

    # 真实分子用于 t-SNE
    try:
        from data import load_smiles_list

        real_all = load_smiles_list(config.DATASET_FILE, max_molecules=None, seed=config.RANDOM_SEED)
        if len(real_all) > config.NUM_REAL_FOR_TSNE:
            random.seed(config.RANDOM_SEED)        #设置随机种子，保证每次运行结果相同
            real_smiles = random.sample(real_all, config.NUM_REAL_FOR_TSNE)    #从真实分子库中随机采样config.NUM_REAL_FOR_TSNE个分子
        else:
            real_smiles = real_all
        visualize_chemical_space(
            generated_smiles=smiles_list,
            real_smiles=real_smiles,
            save_path=str(config.OUTPUT_DIR / "tsne_chemical_space.png"),
        )
    except Exception as e:
        print(f"t-SNE 跳过: {e}")

    print("\n全部完成！")
