"""
GAN 模型定义 - 基于模板填空的分子生成
生成器：噪声 → (模板索引, 取代基索引列表)
判别器：(模板索引, 取代基索引列表) → 真/假概率
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    生成器：将随机噪声转换为分子模板和取代基的选择

    输入：随机噪声张量 [batch_size, noise_dim]
    输出：
        - 模板 logits(未归一化分数，未经softmax,softmax单调递增，谁logits最大，谁概率也最大) [batch_size, num_templates]
        - 取代基 logits(未归一化分数，未经softmax,softmax单调递增，谁logits最大，谁概率也最大) [batch_size, max_r_count, num_substituents]

    froward过程: 从正态分布中随机取一个噪声张量, 形状[B, noise_dim], 进入MLP, 输出形状[B, hidden_dims[-1]], 
    再接两个头分别输出母核和取代基的概率分布, 输出形状[B, num_templates]  [B, max_r_count, num_substituents]二元组

    生成过程：对母核和取代基概率分布每个位置取 argmax, 得到离散索引, 作为最终生成结果。
    """

    def __init__(
        self,
        noise_dim,           # 辨别器没有
        num_templates,
        num_substituents,
        max_r_count,
        hidden_dims=None,
        dropout=0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 512, 256]

        self.noise_dim = noise_dim
        self.num_templates = num_templates
        self.num_substituents = num_substituents
        self.max_r_count = max_r_count

        # 共享的 MLP 主干
        layers = []
        input_dim = noise_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        self.shared_mlp = nn.Sequential(*layers)   # 相当于对噪音数字进行emdedding，用向量表示，输出形状[B, hidden_dims[-1]]

        # 模板分支：输出模板选择
        self.template_head = nn.Linear(input_dim, num_templates)
        # 取代基分支：输出每个位置的取代基选择
        self.substituent_head = nn.Linear(input_dim, max_r_count * num_substituents)

    def forward(self, noise_dim):   # 输入形状[B, noise_dim]，返回母核和取代基的隐式概率向量（未归一化分数向量）
        features = self.shared_mlp(noise_dim)  
        template_logits = self.template_head(features)  # 噪音向量接母核头输出母核的未归一化分数向量，形状[B, num_templates]
        sub_logits = self.substituent_head(features)    # 噪音向量接取代基头输出取代基的未归一化分数向量，形状[B, max_r_count * num_substituents]
        substituent_logits = sub_logits.view(-1, self.max_r_count, self.num_substituents)   # 输出形状[B, max_r_count, num_substituents]
        return template_logits, substituent_logits

    
    def generate(self, noise_dim):
        """推理：对每个位置取 argmax, 得到离散索引, 作为最终生成结果。"""
        with torch.no_grad():
            template_logits, substituent_logits = self.forward(noise_dim)
            # argmax取最大值所在位置的下标。
            # 输出维度位置 = 类别 ID，所以这个下标就是模板库里的索引，取代基同理
            template_indices = torch.argmax(template_logits, dim=-1)     # 输出形状[B]
            substituent_indices = torch.argmax(substituent_logits, dim=-1)   # 输出形状[B, max_r_count]
        return template_indices, substituent_indices


class Discriminator(nn.Module):
    """
    判别器：判断 (模板, 取代基) 组合是真实的还是生成的。

    输入：
        - use_soft=True时 输入是概率分布，表示训练生成器，用概率加权 Embedding, 让梯度能回到生成器。
            - 模板  输入[B, num_templates]概率分布 → 和embedding.weight矩阵相乘   输出按概率混出的模板向量表示[B, template_emb_dim]
            - 取代基  输入[B, max_r_count, num_substituents]概率分布 → 和embedding.weight矩阵相乘   输出按概率混出的取代基向量表示[B, max_r_count, substituent_emb_dim]

        - use_soft=False时 输入是离散索引，表示训练判别器，用离散索引。
            - 模板 输入[B]形状索引 → emdedding  输出模板的向量表示[B, substituent_emb_dim]
            - 取代基 输入[B, max_r_count]形状索引 → embedding  输出取代基的向量表示[B, max_r_count, substituent_emb_dim]
    输出：
        - 真/假概率 [B, 1]。概率越大，越可能是真实样本/假样本

    froward过程：将模板和所有取代基向量表示拼接为一整条长向量后送进MLP，输出真/假概率，输出形状[B, 1]
    
    
    """

    def __init__(
        self,
        num_templates,
        num_substituents,
        max_r_count,
        template_emb_dim=32,           #生成器没有
        substituent_emb_dim=16,        #生成器没有
        hidden_dims=None,
        dropout=0.5,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.num_templates = num_templates
        self.num_substituents = num_substituents
        self.max_r_count = max_r_count

        # nn.Embedding()会创建一个可学习参数矩阵/“编号 → 向量”查找表。在 PyTorch 里它就是 nn.Embedding 的 weight 矩阵。
        # 参数矩阵wight的 行号被约定绑到模板 ID，并由数据更新出来，这样就完成了“行 → 模板向量”的映射。
        # 不是矩阵天生按行等于模板，而是“行号 = 模板 ID”的约定 + 训练，让每一行变成该模板的向量表示。
        self.template_embedding = nn.Embedding(num_templates, template_emb_dim)   
        self.substituent_embedding = nn.Embedding(num_substituents, substituent_emb_dim)

        layers = []
        input_dim = template_emb_dim + max_r_count * substituent_emb_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)   # * 参数解包，把layers列表中的所有层按顺序组合成一个Sequential模型。

    def forward(self, template_input, substituent_input, use_soft=False):
        if use_soft:   # 训练生成器！！！输入是softmax后的 概率分布，梯度经D回到G，更新 G

            # 软 Embedding：概率分布 × embedding 矩阵（生成器训练时回传梯度）
            # 输入template_input为  [B, num_templates] 概率 → 输出[B, template_emb_dim]
            # 矩阵乘法。对每个样本，用各类模板的概率当权重，对所有模板向量做加权平均（特殊的按行线性组合），得到一个按概率混出的模板向量的表示
            template_emb = torch.matmul(template_input, self.template_embedding.weight)  
            # 输入substituent_input为  [B, max_r_count, num_substituents] 概率 → 输出[B, max_r_count , substituent_emb_dim]
            sub_emb = torch.matmul(substituent_input, self.substituent_embedding.weight)  


        else:   # 训练判别器！！！输入是离散索引（真样本 或 生成器argmax后的确定索引）， 更新 D

            # 硬 Embedding：直接用离散索引
            template_emb = self.template_embedding(template_input)     # 输入[B] 索引 → 输出[B, substituent_emb_dim]
            sub_emb = self.substituent_embedding(substituent_input)    # 输入[B, max_r_count] 索引 → 输出[B, max_r_count, substituent_emb_dim]

        sub_emb_flat = sub_emb.view(sub_emb.size(0), -1)    # 把每个样本中所有取代基的embedding首尾拼成一条长向量：[B, max_r_count * substituent_emb_dim]
        combined = torch.cat([template_emb, sub_emb_flat], dim=1)   # 把模板向量和取代基向量拼成一条长向量，用一条向量代表整个母核+取代基，准备送进MLP
        return self.mlp(combined)


class MoleculeGAN:
    """生成器 + 判别器包装。"""

    def __init__(
        self,
        num_templates,
        num_substituents,
        max_r_count,
        noise_dim=128,
        device="cpu",
        g_hidden_dims=None,
        g_dropout=0.3,
        d_template_emb=32,
        d_substituent_emb=16,
        d_hidden_dims=None,
        d_dropout=0.5,
    ):
        if g_hidden_dims is None:
            g_hidden_dims = [256, 512, 256]
        if d_hidden_dims is None:
            d_hidden_dims = [128, 64]

        self.num_templates = num_templates
        self.num_substituents = num_substituents
        self.max_r_count = max_r_count
        self.noise_dim = noise_dim
        self.device = device

        self.generator = Generator(
            noise_dim=noise_dim,
            num_templates=num_templates,
            num_substituents=num_substituents,
            max_r_count=max_r_count,
            hidden_dims=g_hidden_dims,
            dropout=g_dropout,
        ).to(device)

        self.discriminator = Discriminator(
            num_templates=num_templates,
            num_substituents=num_substituents,
            max_r_count=max_r_count,
            template_emb_dim=d_template_emb,
            substituent_emb_dim=d_substituent_emb,
            hidden_dims=d_hidden_dims,
            dropout=d_dropout,
        ).to(device)

        print("GAN 模型已创建：")
        print(f"  - 生成器参数量: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"  - 判别器参数量: {sum(p.numel() for p in self.discriminator.parameters()):,}")

    # 从标准正态分布N(0,1)里随机采样，得到形状[batch_size, noise_dim]的噪音张量
    def sample_noise(self, batch_size):
        return torch.randn(batch_size, self.noise_dim, device=self.device)
