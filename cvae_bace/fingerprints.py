"""
fingerprints.py - Morgan(ECFP4) 指纹工具

单独抽成一个模块是为了让 evaluate.py 计算化学空间相似度时复用，
而不把指纹逻辑塞进评测脚本。
"""

import numpy as np
from rdkit import Chem, DataStructs


def morgan_generator(radius=2, n_bits=2048):
    """
    构造 Morgan 指纹生成函数，兼容新旧 RDKit API。

    输入：
        radius: int，半径 2 对应 ECFP4
        n_bits: int，指纹位数
    输出：
        callable，输入 rdkit Mol 返回 ExplicitBitVect
    """
    try:
        from rdkit.Chem import rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        return gen.GetFingerprint
    except (ImportError, AttributeError):
        # 老版本 RDKit 的接口，新版会给 deprecation 警告但仍可用
        from rdkit.Chem import AllChem
        return lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def smiles_to_fps(smiles_list, radius=2, n_bits=2048):
    """
    批量计算指纹（位向量形式，适合算 Tanimoto 相似度）。

    输入：
        smiles_list: list[str]
        radius, n_bits: 见 morgan_generator
    输出：
        (fps, kept_smiles)
            fps        : list[ExplicitBitVect]
            kept_smiles: list[str]，成功解析的那些 SMILES
    """
    fp_fn = morgan_generator(radius, n_bits)
    fps, kept = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(fp_fn(mol))
        kept.append(smi)
    return fps, kept


def smiles_to_array(smiles_list, radius=2, n_bits=2048):
    """
    批量计算指纹并转成 numpy 矩阵（适合喂给 sklearn 模型 / t-SNE）。

    输入：
        smiles_list: list[str]
        radius, n_bits: 见 morgan_generator
    输出：
        (X, kept_smiles)
            X          : np.ndarray[float32]，形状 [n_kept, n_bits]
            kept_smiles: list[str]
    """
    fps, kept = smiles_to_fps(smiles_list, radius, n_bits)
    X = np.zeros((len(fps), n_bits), dtype=np.float32)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, X[i])
    return X, kept
