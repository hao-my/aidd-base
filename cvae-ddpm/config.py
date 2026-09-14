"""
config.py - 集中配置：路径、数据、模型、训练、生成的全部超参

设计原则：
  1. 所有脚本（train_pretrain / train_conditional / train_latent_diffusion / generate / evaluate）只从这里读默认值，
     命令行参数可以覆盖，避免同一个数字散落在多个文件里对不上。
  2. 阶段1（ZINC 无条件预训练）与阶段2（ZINC LogP 条件微调）必须共用同一套
     词表大小 / embedding 维度 / latent 维度 / 条件维度，否则 load_state_dict 会形状不匹配。
     因此这些"结构类"超参统一放在 MODEL 区，两个阶段都引用同一份。
"""

from pathlib import Path

# ==================== 路径 ====================
PROJECT_DIR = Path(__file__).resolve().parent
AIDD_DIR = PROJECT_DIR.parent
DATA_DIR = AIDD_DIR / "data"

# 阶段1 / 阶段2 共用：约 25 万条 ZINC canonical SMILES（只有一列 smiles）
ZINC_FILE = DATA_DIR / "zinc_gen_full.csv"

# 所有产物（词表、权重、生成 CSV、评测图）统一落在 outputs 下
OUTPUT_DIR = PROJECT_DIR / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"

# 词表文件：阶段1、阶段2、生成、评测都必须读同一份，保证 token id 含义一致
VOCAB_FILE = CKPT_DIR / "vocab.json"
# 阶段1 产物：无条件 VAE 权重
PRETRAIN_CKPT = CKPT_DIR / "vae_pretrained.pt"
# 阶段2 产物：ZINC LogP 条件 CVAE 权重
CVAE_CKPT = CKPT_DIR / "cvae_zinc_logp.pt"
# 阶段3 产物：在 encoder 的 z 上训的条件 DDPM（换掉 N(0,I) 先验）
LATENT_DIFFUSION_CKPT = CKPT_DIR / "latent_ddpm.pt"

RANDOM_SEED = 42

# ==================== 数据 ====================
# SMILES 序列最大 token 数（含 <sos>/<eos>）
MAX_LEN = 128
# 阶段1 最多使用多少条 ZINC 分子。None = 全量（约 25 万）。
PRETRAIN_MAX_MOLECULES = None
# 阶段2 骨架划分测试集比例
COND_TEST_RATIO = 0.2

# 条件：RDKit MolLogP，线性压到 [0,1]。端点取常见药物空间，训练/生成/评测必须一致。
LOGP_MIN = -2.0
LOGP_MAX = 8.0

# ==================== 模型结构（两阶段必须一致） ====================
# 骨干是 Transformer（见 model.py）。d_model 必须能被 nhead 整除。
D_MODEL = 256          # 特征维度，同时也是 token embedding 维度
NHEAD = 8              # 注意力头数
DIM_FEEDFORWARD = 512  # 每层前馈网络的中间维度，惯例取 2~4 倍 D_MODEL
ENCODER_LAYERS = 3     # 编码器层数
DECODER_LAYERS = 3     # 解码器层数
LATENT_DIM = 128       # 潜变量 z 的维度
COND_EMB_DIM = 32      # 条件（归一化 LogP）嵌入后的维度
DROPOUT = 0.1

# ==================== 阶段1：ZINC 无条件预训练 ====================
PRETRAIN_EPOCHS = 8
PRETRAIN_BATCH_SIZE = 256
PRETRAIN_LR = 1e-3
# KL 权重 β 的最终值。太大会后验塌缩 posterior collapse（z 被压成噪声、解码器只会背训练集分布）
PRETRAIN_KL_WEIGHT = 0.3
# KL 退火：前多少个 epoch 把 β 从 0 线性升到 PRETRAIN_KL_WEIGHT
PRETRAIN_KL_ANNEAL_EPOCHS = 3
# free bits：每一维 KL 低于该值时不再继续压，进一步防止 collapse。
# 乘以 LATENT_DIM 就是 KL 的下限（0.1 * 128 = 12.8 nats），
# 冒烟实验里 0.05 的下限使 KL 直接贴地，潜空间几乎没携带信息，故上调。
KL_FREE_BITS = 0.1

# ==================== 阶段2：ZINC LogP 条件微调 ====================
FINETUNE_EPOCHS = 12
FINETUNE_BATCH_SIZE = 128
FINETUNE_MAX_MOLECULES = PRETRAIN_MAX_MOLECULES
FINETUNE_LR = 1e-4
FINETUNE_KL_WEIGHT = 0.3
FINETUNE_KL_ANNEAL_EPOCHS = 4

# 属性预测辅助损失权重：从 z 回归归一化 LogP，强制潜空间携带该性质。
# 这个数必须按损失量级来定，不能想当然写 1.0。
# 重构损失是"每分子的交叉熵求和"，量级在 12~20；
# 归一化后的条件方差大约 0.02 量级，MSE 天然很小。
# 权重取 1.0 时属性项只占总损失的一小截，梯度会被重构项淹没。
# 取 100 后属性项与重构项处于同一数量级，才能推动 z 去编码该性质。
PROPERTY_WEIGHT = 100.0

# 条件 dropout：训练时以该概率把条件换成空条件（classifier-free guidance 的训练侧）。
# 模型必须同时学会有条件和无条件两种生成，生成时才能用两者之差放大条件的影响。
COND_DROPOUT = 0.15

# ==================== 阶段3：潜空间条件 DDPM ====================
# 冻结（或只读）VAE encoder，把真实分子的 mu 当成 x_0，在 128 维 z 上学 p(z|LogP)。
# 生成时先按目标 LogP 把 z 去噪出来，再交给原来的 SMILES 解码器。
LD_TIMESTEPS = 1000
LD_SAMPLE_STEPS = 50          # DDIM 跳步，比 1000 步 DDPM 快一个数量级
LD_HIDDEN_DIM = 256
LD_EPOCHS = 80
LD_BATCH_SIZE = 1024
LD_LR = 1e-3
LD_COND_DROPOUT = 0.1         # 扩散器侧的空条件，便于以后做潜空间 CFG；生成默认不用
# Cosine 调度末端 alpha_bar[999]≈2e-9，DDIM 用它除出来的 x0 会爆到 1e4。
# 采样从 alpha_bar >= 该下限的最大 t 起步（约 t=907），无需重训。
LD_ALPHA_BAR_MIN = 0.02
# 标准化后验大约 N(0,1)，把预测 x0 夹在此范围，挡住残余数值毛刺
LD_X0_CLIP = 8.0

# ==================== 生成 / 评测 ====================
# 每个目标 LogP 生成多少个"有效且不重复"的分子
NUM_GENERATE = 1000
# 采样温度：>1 更发散，<1 更保守
SAMPLE_TEMPERATURE = 1.0
# classifier-free guidance 强度：1.0 = 关闭；>1 放大条件的作用，但过大会损害有效率。
GUIDANCE_SCALE = 8.0
# 是否在解码前用属性头把 z 导向目标 LogP
USE_LATENT_STEERING = True
STEER_STEPS = 60
STEER_LR = 0.05
STEER_PRIOR_WEIGHT = 0.05
# 条件扫描：目标 LogP（落在 ZINC 主体分布内，避免过远的分布外条件）
TARGET_LOGP_LIST = [1.0, 2.0, 3.0, 4.0, 5.0]
MAIN_TARGET_LOGP = 3.0
# t-SNE 里真实分子采样多少个作背景
NUM_REAL_FOR_TSNE = 1000
