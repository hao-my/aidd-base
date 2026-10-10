"""
datasets.py - 数据读取、清洗、划分与 batch 组装

包含三件事：
  1. 读 ZINC：阶段1 无标签预训练；阶段2 用 RDKit 计算 LogP 当条件。
     规范化很关键：同一个分子可以写成很多种 SMILES，不统一的话模型要额外浪费容量
     去学"同一分子的多种写法"，而且后面算 novelty 时字符串比对会失真。
  2. Murcko 骨架划分（scaffold split）：把骨架完全相同的分子锁在同一侧，
     避免"训练集里有个类似物、测试集里换个甲基"这种伪高分。
  3. SmilesDataset + collate_fn：按 batch 动态 padding，而不是一律补到 128。
     CPU 上这一步能省掉大量无效计算。
"""

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import Dataset

# RDKit 对不合法 SMILES 会打一大堆 warning，训练时刷屏，这里关掉
RDLogger.DisableLog("rdApp.*")


# ==================== 清洗 ====================

def canonicalize(smiles):
    """
    把 SMILES 转成 RDKit 规范形式；无法解析则返回 None。

    输入：
        smiles: str
    输出：
        str | None，规范化后的 SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _clean_smiles_list(raw_list, tokenizer_pattern_max_tokens=None):
    """
    批量规范化并去重，顺带丢掉解析失败的条目。

    输入：
        raw_list: Iterable[str]
        tokenizer_pattern_max_tokens: 未使用（长度过滤放在 Dataset 里按 token 数做更准）
    输出：
        list[str]，规范化、去重后的 SMILES（保持首次出现顺序）
    """
    seen = set()
    out = []
    for smi in raw_list:
        if not isinstance(smi, str):
            continue
        can = canonicalize(smi.strip())
        if can is None or len(can) == 0:
            continue
        if can in seen:
            continue
        seen.add(can)
        out.append(can)
    return out


# ==================== 读取 ====================

def load_zinc(path, max_molecules=None, seed=42):
    """
    读取 ZINC 生成数据集（阶段1 预训练语料）。

    输入：
        path         : str | Path，CSV 路径，需含 smiles 列
        max_molecules: int | None，最多保留多少条（None = 全量）。CPU 上建议限量。
        seed         : int，抽样随机种子
    输出：
        list[str]，规范化后的 SMILES 列表
    """
    df = pd.read_csv(path)
    col = "smiles" if "smiles" in df.columns else df.columns[0]
    smiles = df[col].astype(str).tolist()

    if max_molecules is not None and len(smiles) > max_molecules:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(smiles), size=max_molecules, replace=False)
        smiles = [smiles[i] for i in sorted(idx)]

    cleaned = _clean_smiles_list(smiles)
    print(f"[data] ZINC: 读入 {len(smiles)} 条 -> 规范化去重后 {len(cleaned)} 条")
    return cleaned


def mol_logp(smiles):
    """
    计算一条规范 SMILES 的 RDKit Crippen LogP；失败返回 None。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return float(Crippen.MolLogP(mol))
    except Exception:
        return None


def load_zinc_with_logp(path, max_molecules=None, seed=42):
    """
    读取 ZINC，为每条分子计算 LogP，作为阶段2 的条件标签。

    输入：
        path, max_molecules, seed: 同 load_zinc
    输出：
        (smiles_list, logp_array)
    """
    smiles = load_zinc(path, max_molecules=max_molecules, seed=seed)
    kept_s, kept_v = [], []
    for smi in smiles:
        val = mol_logp(smi)
        if val is None:
            continue
        kept_s.append(smi)
        kept_v.append(val)
    values = np.asarray(kept_v, dtype=np.float32)
    print(
        f"[data] ZINC+LogP: {len(kept_s)} 条 | LogP 范围 "
        f"[{values.min():.2f}, {values.max():.2f}] 均值 {values.mean():.2f}"
    )
    return kept_s, values


# ==================== 骨架划分 ====================

def murcko_scaffold(smiles, include_chirality=False):
    """
    取分子的 Bemis-Murcko 骨架（去掉侧链只留环系与连接键）。

    输入：
        smiles           : str
        include_chirality: bool，是否保留手性信息
    输出：
        str，骨架 SMILES；解析失败返回空串
    """
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
    except Exception:
        return ""


def scaffold_split(smiles_list, test_ratio=0.2, seed=42):
    """
    按骨架划分训练 / 测试集：同一骨架的分子全部落在同一侧。

    做法：先把分子按骨架分组，再按"组内分子数从多到少"依次填充训练集，
    直到训练集装满 (1 - test_ratio)，剩下的组进测试集。
    这样测试集里都是训练时没见过的骨架，评测更接近真实的"新化学系列"场景。

    输入：
        smiles_list: list[str]
        test_ratio : float，测试集比例
        seed       : int，用于打乱同尺寸组的顺序
    输出：
        (train_idx, test_idx)，两个 list[int]，是 smiles_list 上的下标
    """
    scaffold_to_idx = {}
    for i, smi in enumerate(smiles_list):
        # setdefault 是 Python 字典的方法：键已经在就取出对应的值；键不在就先放进一个默认值[]，再返回这个默认值
        scaffold_to_idx.setdefault(murcko_scaffold(smi), []).append(i)

    # .values() 取出所有的值（不要键），转换为列表
    groups = list(scaffold_to_idx.values())
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    # .sort默认从小到大排。大组优先进训练集，避免出现"训练集被一堆单例骨架占满"的情况
    groups.sort(key=len, reverse=True)

    n_train = int(len(smiles_list) * (1 - test_ratio))
    train_idx, test_idx = [], []
    for group in groups:
        if len(train_idx) + len(group) <= n_train:
            train_idx.extend(group)
        else:
            test_idx.extend(group)

    print(
        f"[data] scaffold split: train={len(train_idx)} test={len(test_idx)} "
        f"(骨架数 {len(groups)})"
    )
    return train_idx, test_idx


# ==================== 条件归一化 ====================

def normalize_cond(values, vmin, vmax):
    """
    条件标量 -> [0, 1]。训练、生成、评测三处必须用同一组 (vmin, vmax)。

    输入：
        values: float | np.ndarray | torch.Tensor
        vmin, vmax: float，归一化区间端点（见 config.LOGP_MIN / LOGP_MAX）
    输出：
        与输入同类型的归一化结果
    """
    return (values - vmin) / (vmax - vmin)


def denormalize_cond(values, vmin, vmax):
    """
    [0, 1] -> 原始条件尺度，normalize_cond 的逆运算。

    输入：
        values: float | np.ndarray | torch.Tensor
        vmin, vmax: float
    输出：
        与输入同类型的反归一化结果
    """
    return values * (vmax - vmin) + vmin


# ==================== Dataset / collate ====================

class SmilesDataset(Dataset):
    """
    SMILES 数据集。既支持无条件（阶段1），也支持带标量条件（阶段2，当前为归一化 LogP）。

    每个样本返回 (ids, cond)：
        ids : list[int]，[<sos>] + tokens + [<eos>]
        cond: float，归一化后的条件值；无条件时为 0.0（下游用 has_cond 区分）
    """

    def __init__(self, smiles_list, tokenizer, conditions=None, min_tokens=4):
        """
        输入：
            smiles_list: list[str]
            tokenizer  : SMILESTokenizer
            conditions : np.ndarray | None，归一化后的条件值，长度与 smiles_list 一致
            min_tokens : int，token 数少于该值的分子丢弃（过短的碎片对训练没帮助）
        输出：
            Dataset 实例
        """
        self.tokenizer = tokenizer
        self.smiles = []
        self.encoded = []
        self.conditions = []

        for i, smi in enumerate(smiles_list):
            ids = tokenizer.encode(smi, add_special=True)
            if len(ids) < min_tokens:
                continue
            self.smiles.append(smi)
            self.encoded.append(ids)
            self.conditions.append(0.0 if conditions is None else float(conditions[i]))

        self.has_cond = conditions is not None

    def __len__(self):
        """样本总数"""
        return len(self.encoded)

    def __getitem__(self, idx):
        """
        输入：
            idx: int
        输出：
            (list[int], float)，token id 序列与条件值
        """
        return self.encoded[idx], self.conditions[idx]


def make_collate_fn(pad_id):
    """
    生成 collate_fn：把一个 batch 的变长序列补齐到"本 batch 最长"，而不是全局 max_len。

    输入：
        pad_id: int，padding 用的 token id
    输出：
        callable，可直接传给 DataLoader(collate_fn=...)
    """

    def collate(batch):
        """
        输入：
            batch: list[(list[int], float)]
        输出：
            dict，键为：
                input_ids : LongTensor [B, L]  完整序列（含 sos/eos），送给 encoder
                dec_input : LongTensor [B, L-1] 教师强制的解码器输入（去掉最后一个 token）
                dec_target: LongTensor [B, L-1] 解码器目标（去掉第一个 token，即右移一位）
                lengths   : LongTensor [B]      每条序列的真实长度，用于 pack_padded_sequence
                cond      : FloatTensor [B, 1]  归一化条件值
        """
        seqs = [item[0] for item in batch]     # 每个样本的ids（未pad）组成的列表
        conds = [item[1] for item in batch]    # 每个样本的归一化条件值（无条件时为0.0）列表
        lengths = torch.tensor([len(ids) for ids in seqs], dtype=torch.long)  # 每个样本的ids长度列表
        max_len = int(lengths.max())  # 本batch最长序列长度
        padded = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)  # 生成一个[B, 本batch最长序列长度]的矩阵，矩阵元素为pad_id
        for i, ids in enumerate(seqs):  # 将每个样本的ids填充到padded矩阵中
            padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        return {
            "input_ids": padded,
            # 自回归错位：用 t 位置的 token 预测 t+1 位置的 token
            "dec_input": padded[:, :-1],
            "dec_target": padded[:, 1:],
            "lengths": lengths,
            "cond": torch.tensor(conds, dtype=torch.float32).unsqueeze(1),
        }

    return collate
