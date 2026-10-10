"""
集中配置：路径与超参。改数据源或冒烟截断时只动本文件。

默认数据方案 A：与 08 notebook 同一套全量 ZINC canonical 缓存。
"""

from pathlib import Path

# ---------- 路径 ----------
_MOL_GAN_DIR = Path(__file__).resolve().parent
_AIDD_DIR = _MOL_GAN_DIR.parent
DATA_DIR = _AIDD_DIR / "data"

# 方案 A：08 写出的全量 canonical SMILES（约 25 万，列名 smiles）
DATASET_FILE = DATA_DIR / "zinc_gen_full.csv"
# 备选（改上面一行即可）：
# DATASET_FILE = DATA_DIR / "zinc250k_raw.csv"
# DATASET_FILE = DATA_DIR / "bace_pic50.csv"          # 仅冒烟，片段库会很小

OUTPUT_DIR = _MOL_GAN_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_FINAL = CHECKPOINT_DIR / "gan_final.pt"

# ---------- 数据规模 ----------
# None = 全量；整数 = 调试时最多用这么多分子（随机采样，seed 见 RANDOM_SEED）
MAX_MOLECULES = None

RANDOM_SEED = 42

# ---------- 训练 ----------
EPOCHS = 100
BATCH_SIZE = 64
LR_G = 0.0002    # 学得相对快一点，更容易追上、去骗 D
LR_D = 0.0001    # 学得慢一点，避免过早过强
SAMPLE_INTERVAL = 10  # 每隔多少 epoch 存 checkpoint + 画分子/QED

# ---------- 模型 ----------
NOISE_DIM = 512
G_HIDDEN_DIMS = [256, 512, 256]
G_DROPOUT = 0.3
D_TEMPLATE_EMB = 32
D_SUBSTITUENT_EMB = 16
D_HIDDEN_DIMS = [256, 128, 64]
D_DROPOUT = 0.3
TEMPERATURE = 0.8  # softmax 温度：越大采样越分散

# ---------- 生成 / 评测空壳 ----------
NUM_GENERATE = 1000
NUM_REAL_FOR_TSNE = 1000
