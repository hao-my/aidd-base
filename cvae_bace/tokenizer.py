"""
tokenizer.py - SMILES 字符级分词器

作用：把 SMILES 字符串和整数 id 序列互相转换，供 VAE 的 embedding 层使用。

关键点（容易搞混的地方）：
  1. 不能按单个字符切！"Cl"、"Br"、"[nH]" 都是一个化学单元，
     按字符切会把 Cl 拆成 C 和 l，模型永远学不会写氯原子。
     所以用正则一次匹配一个 token。
  2. 词表必须在阶段1（ZINC 预训练）和阶段2（LogP 微调）之间共享。
     词表一变，Embedding 的行数就变，预训练权重会加载失败。
     因此这里提供 build_or_load()：第一次用 ZINC 建好并存盘，之后一律读盘。
  3. 特殊 token 的 id 固定在最前面（pad=0, unk=1, sos=2, eos=3），
     这样 CrossEntropyLoss(ignore_index=0) 之类的写法在任何脚本里都成立。
"""

import json
import re
from collections import Counter
from pathlib import Path


class SMILESTokenizer:
    """
    SMILES 分词器。

    属性：
        vocab      : dict, token 字符串 -> 整数 id
        inv_vocab  : dict, 整数 id -> token 字符串
        max_len    : int, 编码后允许的最大长度（含 <sos>/<eos>）
    """

    # 特殊 token 顺序固定，不要随意调整（下游代码依赖 pad=0）
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]

    # SMILES 正则：优先匹配方括号原子和双字符元素
    PATTERN = (
        r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|"
        r"\(|\)|\.|=|#|-|\+|\\\\|/|:|~|@|\?|>|\*|\\*|\$|%[0-9]{2}|[0-9])"
    )

    def __init__(self, max_len=128):
        """
        输入：
            max_len: 编码后序列的最大长度（含首尾特殊符）
        输出：
            无返回值，构造一个只含 4 个特殊 token 的空词表分词器
        """
        self.max_len = max_len
        self.regex = re.compile(self.PATTERN)
        self.vocab = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        self.inv_vocab = {i: tok for tok, i in self.vocab.items()}

    # ---------- 基础 ----------

    def tokenize(self, smiles):
        """
        把一条 SMILES 切成 token 列表。

        输入：
            smiles: str，例如 "CCOc1ccccc1"
        输出：
            list[str]，例如 ['C', 'C', 'O', 'c', '1', 'c', 'c', 'c', 'c', 'c', '1']

        注：上面正则里的 `\\*`（零个或多个反斜杠）在其它分支都匹配不上时会命中空串，
        findall 于是会掺进一堆 ''。空 token 没有化学含义，还会白白拉长序列
        （实测多出约 15%~20% 的长度），所以这里显式过滤。
        """
        return [tok for tok in self.regex.findall(smiles) if tok]

    def build_vocab(self, smiles_list):
        """
        统计所有 token 并写入词表（高频 token 拿到更小的 id，纯属习惯，不影响正确性）。

        输入：
            smiles_list: Iterable[str]，用来统计的 SMILES 集合
        输出：
            self（链式调用方便：SMILESTokenizer().build_vocab(...)）
        """
        counter = Counter()
        for smi in smiles_list:
            counter.update(self.tokenize(smi))
        for token, _ in counter.most_common():
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.inv_vocab[idx] = token
        return self

    # ---------- 编码 / 解码 ----------

    def encode(self, smiles, add_special=True):
        """
        SMILES -> 整数 id 列表（不做 padding，padding 交给 collate_fn 按 batch 动态补，
        这样短分子不用陪着最长分子一起算 128 步，CPU 上能快 2-3 倍）。

        输入：
            smiles      : str
            add_special : bool，是否在首尾加 <sos>/<eos>
        输出：
            list[int]，长度 <= max_len
        """
        tokens = self.tokenize(smiles)
        # 预留 2 个位置给 <sos>/<eos>，超长直接截断
        limit = self.max_len - 2 if add_special else self.max_len
        if len(tokens) > limit:
            tokens = tokens[:limit]

        unk_id = self.vocab[self.UNK_TOKEN]
        ids = [self.vocab.get(tok, unk_id) for tok in tokens]
        if add_special:
            ids = [self.vocab[self.SOS_TOKEN]] + ids + [self.vocab[self.EOS_TOKEN]]
        return ids

    def decode(self, ids, stop_at_eos=True):
        """
        整数 id 列表 -> SMILES 字符串。

        输入：
            ids         : Iterable[int]，模型生成或数据集里的 id 序列
            stop_at_eos : bool，遇到 <eos> 立即停止（生成时必须 True，
                          否则 <eos> 之后的随机 token 会被拼进 SMILES 导致解析失败）
        输出：
            str，拼接好的 SMILES（不含任何特殊符）
        """
        out = []
        for token_id in ids:
            token = self.inv_vocab.get(int(token_id), self.UNK_TOKEN)
            if token == self.EOS_TOKEN:
                if stop_at_eos:
                    break
                continue
            if token in self.SPECIAL_TOKENS:
                # <pad>/<sos>/<unk> 一律跳过，不进入 SMILES
                continue
            out.append(token)
        return "".join(out)

    # ---------- 便捷属性 ----------

    @property
    def pad_id(self):
        """<pad> 的 id（恒为 0）"""
        return self.vocab[self.PAD_TOKEN]

    @property
    def sos_id(self):
        """<sos> 的 id"""
        return self.vocab[self.SOS_TOKEN]

    @property
    def eos_id(self):
        """<eos> 的 id"""
        return self.vocab[self.EOS_TOKEN]

    def __len__(self):
        """词表大小，等于 Embedding 的行数"""
        return len(self.vocab)

    # ---------- 存盘 / 读盘 ----------

    def save(self, path):
        """
        把词表写成 JSON。

        输入：
            path: str | Path，输出文件路径
        输出：
            无（副作用：写文件）
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"max_len": self.max_len, "vocab": self.vocab}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        """
        从 JSON 读回词表。

        输入：
            path: str | Path
        输出：
            SMILESTokenizer 实例
        """
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tok = cls(max_len=payload["max_len"])
        tok.vocab = {k: int(v) for k, v in payload["vocab"].items()}
        tok.inv_vocab = {v: k for k, v in tok.vocab.items()}
        return tok


def build_or_load_vocab(vocab_file, smiles_for_build=None, max_len=128, rebuild=False):
    """
    获取"全流程唯一"的分词器：有缓存就读缓存，没有就用给定 SMILES 建好并存盘。

    为什么要这样做：
        阶段1 在 ZINC 上训练，阶段2 在同一批分子上用 LogP 微调。如果两次各自建词表，
        新出现的 token 会让词表变大，Embedding 行数改变，
        阶段2 加载阶段1 权重时直接报形状错误。所以词表必须一次建好、全程复用。

    输入：
        vocab_file       : str | Path，词表缓存路径
        smiles_for_build : Iterable[str] | None，缓存不存在时用来建词表的 SMILES
        max_len          : int，最大序列长度
        rebuild          : bool，True 则忽略缓存强制重建
    输出：
        SMILESTokenizer 实例
    """
    vocab_file = Path(vocab_file)
    if vocab_file.exists() and not rebuild:
        tok = SMILESTokenizer.load(vocab_file)
        print(f"[tokenizer] 读取已有词表: {vocab_file} (size={len(tok)})")
        return tok

    if smiles_for_build is None:
        raise FileNotFoundError(
            f"词表 {vocab_file} 不存在，且未提供用于构建词表的 SMILES。"
            f"请先运行 train_pretrain.py 生成词表。"
        )

    tok = SMILESTokenizer(max_len=max_len).build_vocab(smiles_for_build)
    tok.save(vocab_file)
    print(f"[tokenizer] 新建词表: {vocab_file} (size={len(tok)})")
    return tok
