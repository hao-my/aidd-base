"""
数据处理模块 - 基于模板填空的分子生成
从真实数据集自动提取分子片段库（BRICS）

训练样本不是 SMILES 字符串，而是：
    (模板索引, 取代基索引列表)
生成时再 decode 拼回 SMILES。
"""

import re
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

# 这些变量会在 load 时由 TemplateDataset 填充（训练前必须先建 Dataset）
TEMPLATES = []           # 骨架模板/母核   eg：c1ccc({R1})cc1，这里{R1}是后面要填取代基的位置
SUBSTITUENTS = []        # 取代基/侧链
TEMPLATE_R_COUNTS = []   # 模板中{R}占位符数量
MAX_R_COUNT = 0          # 模板中{R}占位符最大数量
NUM_TEMPLATES = 0        # 模板数量
NUM_SUBSTITUENTS = 0     # 取代基数量


# 预先编译两条正则，后面用来反复找/替换字符串，避免每次现写正则
# BRICS dummy：[*]、[1*] … [99*]；与参考 GAN 默认模板里的 {R1}/{R2} 对应
_DUMMY_RE = re.compile(r"\[\d*\*\]")          # 匹配 BRICS 的 dummy 连接点 [*]、[1*]等
_PLACEHOLDER_RE = re.compile(r"\{R\d+\}")     # 匹配模板里的 取代基占位符：{R1}、{R2} 等


def strip_dummies(farg):
    """去掉 BRICS 连接点标记，得到可解析的片段 SMILES。"""
    return _DUMMY_RE.sub("", farg)  # re.Pattern.sub(替换内容, 原字符串) = 把原字符串中的所有匹配项替换为替换内容


def strip_placeholders(template):
    """去掉 {R1}/{R2}…，用于和清洗后的片段对齐。"""
    return _PLACEHOLDER_RE.sub("", template)


def count_substitution_sites(template):
    """计算模板中有多少个 {R} 占位符（多位点模板的核心计数）。"""
    return len(_PLACEHOLDER_RE.findall(template))


def brics_frag_to_template(frag_with_dummy):
    """
    把 BRICS 片段上的每个 dummy 转成一个占位符：
        c1cc([16*])cc([16*])c1  →  c1cc({R1})cc({R2})c1
    没有 dummy 时退回「末尾补一个 {R1}」（旧逻辑兜底）。
    """
    n = 0

    def _repl(_match):
        nonlocal n     # 将外层全局变量n 拉进函数里，可对其更改 
        n += 1
        return f"{{R{n}}}"

    # sub 的第一个参数可以是字符串，也可以是 函数。如果是函数，则每次匹配到 dummy 时调用一次该函数，并将匹配结果作为参数传入。
    template = _DUMMY_RE.sub(_repl, frag_with_dummy)
    if n == 0:
        template = strip_dummies(frag_with_dummy) + "{R1}"
    return template


# 输入一个smiles，返回一个字典列表。每个字典包含brics片段的一个原始片段raw、去掉连接点dummy之后的片段cleaned、core_key、重原子数num_heavy
def decompose_molecule_with_brics(smiles):
    """
    使用 BRICS 算法把分子切成片段。

    Returns:
        list of dict:
            raw: 带 dummy 的 BRICS SMILES
            cleaned: 去掉 dummy 后的 SMILES（可能非法，仅小片段可靠）
            core_key: 模板去 {R} 后的字符串，用来和 TEMPLATES 对齐
            num_heavy: 不含 dummy / H 的重原子数
        失败返回 None
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import BRICS

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fragments = list(BRICS.BRICSDecompose(mol))
        out = []
        seen_raw = set()
        for frag in fragments:
            if frag in seen_raw:      # 还未加入seen_raw集合，则加入并处理
                continue
            mol_raw = Chem.MolFromSmiles(frag)    # 解析片段为分子
            if mol_raw is None:
                continue

            # atom.GetAtomicNum() = 获取原子序数
            # dummy 原子序数为 0，H 为 1；重原子用 >1 统计，避免 * 被算进去
            # num_heavy计算片段fargments中重原子总数
            num_heavy = sum(1 for atom in mol_raw.GetAtoms() if atom.GetAtomicNum() > 1)
            cleaned = strip_dummies(frag)                     # 去除fragment中的dummy
            template_like = brics_frag_to_template(frag)      # 把fragment中的dummy转成{R1}、{R2}等占位符
            core_key = strip_placeholders(template_like)      # 去掉占位符
            seen_raw.add(frag)
            out.append({
                "raw": frag,
                "cleaned": cleaned,
                "core_key": core_key,
                "num_heavy": num_heavy,
            })

        return out if out else None
    except Exception:
        return None


# 输入dataset文件，返回smiles列表
def load_smiles_list(dataset_file, max_molecules=None, seed=42):
    """
    从 txt（每行一条）或 csv（优先 smiles 列，否则第一列）读取 SMILES。

    Args:
        dataset_file: 路径
        max_molecules: None=全量；整数=随机采样这么多条
        seed: 采样随机种子

    Returns:
        smiles_list: str 列表
    """
    import random

    import pandas as pd

    path = Path(dataset_file)
    if not path.exists():
        raise FileNotFoundError(f"找不到数据集: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        col = "smiles" if "smiles" in df.columns else df.columns[0]
        all_smiles = df[col].astype(str).str.strip().tolist()
        all_smiles = [s for s in all_smiles if s and s.lower() != "nan"]
    else:
        with open(path, "r", encoding="utf-8") as f:
            all_smiles = [line.strip() for line in f if line.strip()]

    print(f"  - 数据集总分子数: {len(all_smiles)}")

    if max_molecules and len(all_smiles) > max_molecules:
        random.seed(seed)
        smiles_list = random.sample(all_smiles, max_molecules)
        print(f"  - 随机采样: {max_molecules} 个分子用于训练")
    else:
        smiles_list = all_smiles
        print(f"  - 使用全部分子: {len(smiles_list)} 个")

    return smiles_list

# 输入dataset文件，返回母核/骨架库列表（包含R）、取代基库列表  和
# 字典列表, 一个分子一个字典。字典内两个键值对：一个smiles，一个fragments字典列表，每个字典内四个键值对：原始片段raw、去掉连接点dummy之后的片段cleaned、core_key、重原子数num_heavy
def build_fragment_library_from_dataset(dataset_file, max_molecules=None, seed=42):
    f"""
    从真实数据集提取片段库，并分成模板（大片段）与取代基（小片段）。

    分类规则：
        重原子 < 5  → 取代基（用去掉 dummy 的 SMILES）
        重原子 ≥ 5  → 模板：BRICS 每个连接点 → R1/R2/…（多位点）

    Returns:
        templates, substituents, molecule_fragments                       片段1                    片段2                       片段3                    
        molecule_fragments: [{"original_smiles", "fragments": [{ cleaned, num_heavy} ...]}, ...]
    """
    try:
        from rdkit import Chem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        print("错误: RDKit 未安装，无法从真实数据集提取片段")
        return None, None, None

    print("\n" + "=" * 60)
    print(f"从真实数据集提取片段库: {dataset_file}")

    try:
        smiles_list = load_smiles_list(dataset_file, max_molecules=max_molecules, seed=seed)
    except Exception as e:
        print(f"错误: 无法读取数据集 - {e}")
        return None, None, None

    all_raw_frag = {}   # 键为原始片段raw → 值为decompose_molecule_with_brics(smi)返回值中对应片段的的字典
    molecule_fragments = []
    successful_molecules = 0

    print("  - 分解分子中...")
    for i, smi in enumerate(smiles_list):
        if (i + 1) % 5000 == 0:
            print(f"    已处理 {i + 1}/{len(smiles_list)} 个分子...")

        # items是一个字典列表 每个字典包含brics片段的一个原始片段raw、去掉连接点dummy之后的片段cleaned、core_key、重原子数num_heavy
        items = decompose_molecule_with_brics(smi)   
        if items and len(items) > 0:
            for item in items:
                all_raw_frag[item["raw"]] = item
            molecule_fragments.append({"original_smiles": smi, "fragments": items})
            successful_molecules += 1

    print(f"  - 成功分解: {successful_molecules}/{len(smiles_list)} 个分子")
    print(f"  - 提取到 {len(all_raw_frag)} 个唯一 BRICS 片段（含连接点）")

    templates = []   #带{R}占位符的母核模板列表
    substituents = [""]  # 取代基库列表。空取代基，表示 H / 不填  

    from rdkit import Chem

    for raw_frag, item in all_raw_frag.items():
        try:
            num_heavy = item["num_heavy"]
            cleaned = item["cleaned"]
            if num_heavy < 5:
                if cleaned and Chem.MolFromSmiles(cleaned) is not None:
                    if cleaned not in substituents:
                        substituents.append(cleaned)
            else:
                template_smi = brics_frag_to_template(raw_frag)
                if template_smi not in templates:
                    templates.append(template_smi)
                    
        except Exception:
            continue

    print(f"  - 模板数量: {len(templates)} (重原子≥5，按连接点数填 {{R}})")
    print(f"  - 取代基数量: {len(substituents)} (重原子<5)")

    print("\n  - 模板样本 (前5个):")
    for i, t in enumerate(templates[:5]):
        print(f"    {i + 1}. {t}  (R位点={count_substitution_sites(t)})")
    print("\n  - 取代基样本 (前10个):")
    for i, s in enumerate(substituents[:10]):
        print(f"    {i + 1}. (H)" if s == "" else f"    {i + 1}. {s}")
    print("=" * 60)

    return templates, substituents, molecule_fragments


class TemplateDataset(Dataset):
    """
    init：读取数据集，构建母核/骨架库列表(包含R)、取代基库列表 和 分子片段字典列表
           遍历每一个smiles，对应出母核索引和取代基索引列表，组成样本

    getitem：返回一个smiles的模板索引和取代基索引列表的张量
    """

    def __init__(self, dataset_file, max_molecules=None, seed=42):
        global TEMPLATES, SUBSTITUENTS, TEMPLATE_R_COUNTS, MAX_R_COUNT
        global NUM_TEMPLATES, NUM_SUBSTITUENTS

        print("\n" + "=" * 60)
        print("正在从真实数据集构建训练数据...")

        result = build_fragment_library_from_dataset(dataset_file, max_molecules=max_molecules, seed=seed)

        if result[0] is None:
            raise RuntimeError(f"无法从数据集提取片段库，请检查 {dataset_file}")

        TEMPLATES, SUBSTITUENTS, molecule_fragments = result

        TEMPLATE_R_COUNTS = [count_substitution_sites(t) for t in TEMPLATES]   # 与母核库索引对应的占位符数量列表
        MAX_R_COUNT = max(TEMPLATE_R_COUNTS) if TEMPLATE_R_COUNTS else 1
        NUM_TEMPLATES = len(TEMPLATES)
        NUM_SUBSTITUENTS = len(SUBSTITUENTS)

        # molecule_fragments列表中每个元素都是来自dataset中真实smiles的信息
        # 返回值samples是一个二元组的列表，每个元素都是一个smiles的(母核索引, 取代基索引列表)
        self.samples, self.original_smiles_list = self._build_samples_from_real_molecules(molecule_fragments)

        self.max_r_count = MAX_R_COUNT
        self.num_templates = NUM_TEMPLATES
        self.num_substituents = NUM_SUBSTITUENTS

        print("\n训练数据集信息：")
        print(f"  - 真实分子样本数: {len(self.samples)}")
        print(f"  - 模板数量: {NUM_TEMPLATES}")
        print(f"  - 取代基数量: {NUM_SUBSTITUENTS}")
        print(f"  - 最大取代位点: {MAX_R_COUNT}")
        print(f"  - 数据来源: {dataset_file}")
        print("=" * 60)


    # 
    def _build_samples_from_real_molecules(self, molecule_fragments):
        """
        一个smiles中, 切割出的片段按重原子数大小排序
        → 最大且合法的一块作为母核，并获取其在母核库中的索引（取代基最多的那个母核索引）
        → 剩下的按这个母核的取代基位点个数取前几个 
        → 得到取代基库中对应的索引列表
        → pad 到所有母核中最大取代基位点数，方便组成定长张量（多出来的位填 0）
        → 将 母核索引和取代基索引列表 追加到samples列表
        → 返回所有smiles的母核索引和取代基索引列表 和 原始smiles列表
        """
        samples = []                  # 二维元组列表  dataset中每个smiles样本的母核索引和取代基索引列表
        original_smiles_list = []     # 原始smiles列表

        print("  - 构建训练样本（使用真实分子的 BRICS 片段）...")

        # 建立干净母核core_key 与 母核库中取代基位点最多母核索引 的映射关系
        # 样本中core_key母核  对应到  母核库中取代基位点最多的母核索引
        template_to_idx = {}   
        for i, t in enumerate(TEMPLATES):
            base = strip_placeholders(t)  # 去掉所有 {R1}/{R2}…，与 cleaned 对齐
            prev = template_to_idx.get(base)
            # 同一母核若有多种连接点数，保留位点更多的模板（有去重作用）
            if prev is None or TEMPLATE_R_COUNTS[i] > TEMPLATE_R_COUNTS[prev]:
                template_to_idx[base] = i

        substituent_to_idx = {s: i for i, s in enumerate(SUBSTITUENTS)}

        
        # molecule_fragments: [{original_smiles：smi, fragments: [{raw ,cleaned, core_key, num_heavy} ...]}, ...]
        for mol_data in molecule_fragments:      # 遍历每一个分子的原始smiles和片段列表字典
            original_smiles = mol_data["original_smiles"]
            frags = mol_data["fragments"]        # 又是一个四键值对的片段字典列表，每个字典对应一个片段的信息
            if not frags:
                continue

            try:
                items = list(frags)
                items.sort(key=lambda x: x["num_heavy"], reverse=True)  # 按重原子数对这个smiles的片段字典列表降序排序

                template_idx = None
                remaining = []       # 除模板外的其他片段，字典列表
                for item in items:    #遍历每一个片段字典，找到第一个合格模板（已降序，所以就是记录在母核库中重原子数最大的那个）
                    if item["num_heavy"] >= 5 and item["core_key"] in template_to_idx:
                        template_idx = template_to_idx[item["core_key"]]    # 记下这个模板在 母核库TEMPLATES 里的下标
                        remaining = [x for x in items if x is not item]     # 记下除模板外的其他片段，作为候选取代基
                        break    # 找到第一个合格模板就停（因为已按大小排序，这是最大的可用母核）

                if template_idx is None:
                    continue    # 这个smiles找不到可用模板 → 丢掉，不建样本。

                r_count = TEMPLATE_R_COUNTS[template_idx]
                substituent_indices = []
                for item in remaining[:r_count]:    # 只取排序后的前 r_count 个剩余片段（位点不够就少取；多了的丢掉）
                    cleaned = item["cleaned"]
                    # 用去掉 dummy 的片段查取代基表；查不到就用 0（通常是空取代基 / H
                    substituent_indices.append(substituent_to_idx.get(cleaned, 0))

                # 补齐到所有母核中最大取代基位点数，方便组成定长张量（多出来的位填 0）
                while len(substituent_indices) < MAX_R_COUNT:
                    substituent_indices.append(0)

                samples.append((template_idx, substituent_indices))   # 一个样本的模板索引和取代基索引列表
                original_smiles_list.append(original_smiles)          # 原始smiles
            except Exception:
                continue

        print(f"  - 成功构建: {len(samples)} 个真实分子样本")
        return samples, original_smiles_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        template_idx, substituent_indices = self.samples[idx]
        return (
            torch.tensor(template_idx, dtype=torch.long),
            torch.tensor(substituent_indices, dtype=torch.long),
        )

    def decode(self, template_idx, substituent_indices):
        """把 (模板, 取代基) 解码为 SMILES 字符串（占位符替换，非严格化学键合）。"""
        if isinstance(template_idx, torch.Tensor):  # 判断传入的是否是张量，如果是则转换为标量
            template_idx = template_idx.item()
        if isinstance(substituent_indices, torch.Tensor):  # 判断传入的是否是张量，如果是则转换为列表
            substituent_indices = substituent_indices.cpu().numpy().tolist()

        template = TEMPLATES[template_idx]
        smiles = template
        r_count = TEMPLATE_R_COUNTS[template_idx]
        for i in range(r_count):
            placeholder = f"{{R{i + 1}}}"
            sub_idx = substituent_indices[i]
            smiles = smiles.replace(placeholder, SUBSTITUENTS[sub_idx])
        return smiles


def get_dataloader(batch_size=32, shuffle=True, dataset_file=None, max_molecules=None, seed=42):
    """创建 DataLoader；首次加载时尝试画出片段库示意图。"""
    if dataset_file is None:
        from config import DATASET_FILE, MAX_MOLECULES, RANDOM_SEED

        dataset_file = DATASET_FILE
        if max_molecules is None:
            max_molecules = MAX_MOLECULES
        seed = RANDOM_SEED

    dataset = TemplateDataset(
        dataset_file=str(dataset_file), max_molecules=max_molecules, seed=seed
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    _visualize_fragment_library()
    return dataloader, dataset


def _visualize_fragment_library():
    """把模板 / 取代基画成一张网格图（已存在则跳过）。"""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return

    from config import OUTPUT_DIR

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "fragment_library.png"
    if out_path.exists():
        return

    print("\n" + "=" * 60)
    print("可视化片段库...")

    template_mols, template_legends = [], []
    for i, template in enumerate(TEMPLATES[:18]):
        smiles = strip_placeholders(template)
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            template_mols.append(mol)
            template_legends.append(f"Template {i}")

    substituent_mols, substituent_legends = [], []    
    for i, sub in enumerate(SUBSTITUENTS[:15]):
        # c1ccccc1 是苯；把取代基接到苯上只为了画图好看、能解析，不是建库或训练逻辑的一部分。
        smiles = "c1ccccc1" if sub == "" else f"c1ccc({sub})cc1" 
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            substituent_mols.append(mol)
            substituent_legends.append(
                f"Substituent {i}: (H)" if sub == "" else f"Substituent {i}: {sub}"
            )

    all_mols = template_mols + substituent_mols
    all_legends = template_legends + substituent_legends
    if all_mols:
        img = Draw.MolsToGridImage(
            all_mols,
            molsPerRow=6,
            subImgSize=(300, 300),
            legends=all_legends,
            returnPNG=False,
        )
        img.save(str(out_path))
        print(f"  - Fragment library saved: {out_path}")
        print(f"  - Templates: {len(template_mols)}, Substituents: {len(substituent_mols)}")
    print("=" * 60)


if __name__ == "__main__":
    from config import DATASET_FILE, RANDOM_SEED

    print("测试数据加载（仅 200 分子冒烟）...\n")
    dataloader, dataset = get_dataloader(
        batch_size=4, dataset_file=DATASET_FILE, max_molecules=200, seed=RANDOM_SEED
    )
    template_batch, substituent_batch = next(iter(dataloader))
    print("\nBatch shapes:")
    print(f"  - 模板索引: {template_batch.shape}")
    print(f"  - 取代基索引: {substituent_batch.shape}")
    print("\n前4个样本解码:")
    for i in range(min(4, template_batch.size(0))):
        smiles = dataset.decode(template_batch[i], substituent_batch[i])
        print(f"  {i + 1}. 模板{template_batch[i].item()} -> {smiles}")
