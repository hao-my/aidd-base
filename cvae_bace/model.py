"""
model.py - SMILES 条件变分自编码器（CVAE），Transformer 骨干

整体结构：
    SMILES ids ──> Embedding + 位置 ──> TransformerEncoder ──> 掩码均值池化 ──> h ──┐
                                                                                    ├──> fc_mu / fc_logvar ──> z ~ N(mu, sigma)
    条件 LogP ──> cond MLP ──> c ──────────────────────────────────────────────────┘
                                                                                    │
    z ──> z_token ┐                                                                 │
    c ──> c_token ┴──> memory（长度为 2 的"源序列"）<───────────────────────────────┘
    token_emb + 位置 ──> TransformerDecoder（因果掩码 + 交叉注意力看 memory）──> Linear ──> 词表 logits
    z ──> property_head ──> 预测 LogP（辅助任务，逼 z 携带亲脂性信息）

这版是从 GRU 骨干改过来的。换骨干时有四个地方必须重新想清楚，
它们都是 GRU"天生自带"而 Transformer 需要显式补上的东西：

    GRU 的做法                            Transformer 的做法
    ----------------------------------    ----------------------------------
    循环结构天然知道 token 的先后顺序      自注意力对顺序无感，要额外加位置嵌入
    pack_padded_sequence 跳过 padding     src_key_padding_mask 屏蔽 padding
    单向 GRU 天然看不到未来               因果掩码（下三角）挡住未来位置
    取最后一步隐状态当整句表示            对所有非 pad 位置做掩码均值池化

编码器、解码器本体直接用 PyTorch 自带的 nn.TransformerEncoder / nn.TransformerDecoder，
不自己拼注意力——这两个模块内部就是标准的多头注意力 + 前馈网络，
自己重写一遍除了增加出错的机会没有别的好处。

三个容易踩的坑，这里都做了处理：

1) 【两阶段权重要能对上】
   阶段1 在 ZINC 上做无条件预训练，阶段2 在同一批分子上用 LogP 做条件微调。
   如果阶段1 的模型压根没有条件模块，阶段2 再加进去，load_state_dict 就会缺 key。
   解决办法：模型结构从一开始就带条件通路，阶段1 传 y=None 时用全零向量当"空条件"
   （相当于 classifier-free guidance 里的 null condition）。
   这样阶段2 可以直接 load 阶段1 的权重，只是条件 MLP 从随机初始化开始学。

2) 【posterior collapse】
   解码器太强时，它会无视 z 直接背下训练集的字符分布，KL 掉到 0，潜空间就废了。
   换成 Transformer 之后这个风险比 GRU 更大——自注意力看前缀的能力更强，
   更容易"光靠前缀就把后面猜出来"，压根不需要 z。
   对策一：把 z 放进 memory，让解码器每一层都通过交叉注意力去看它
   （而不是只用来初始化一下隐状态，那样 z 的影响会随着生成越走越淡）。
   对策二：训练脚本里用 KL 退火 + free bits（见 losses.py）。

3) 【生成时必须在 <eos> 处停】
   采样循环里一旦某条序列吐出 <eos>，后面就一直补 <pad>，
   否则 <eos> 之后的随机 token 会被拼进 SMILES 字符串导致解析失败。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmilesVAE(nn.Module):
    """
    SMILES (条件) 变分自编码器，Transformer 骨干。

    构造参数：
        vocab_size         : int,   词表大小（= len(tokenizer)）
        d_model            : int,   Transformer 特征维度，也是 token embedding 维度
        nhead              : int,   注意力头数（d_model 必须能被它整除）
        dim_feedforward    : int,   每层前馈网络的中间维度
        num_encoder_layers : int,   编码器层数
        num_decoder_layers : int,   解码器层数
        latent_dim         : int,   潜变量 z 的维度
        cond_dim           : int,   条件嵌入维度
        dropout            : float, dropout 比例
        max_len            : int,   位置嵌入表长度，要不小于最长序列
        pad_id             : int,   <pad> 的 token id
        sos_id             : int,   <sos> 的 token id
        eos_id             : int,   <eos> 的 token id
    """

    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        dim_feedforward=512,
        num_encoder_layers=3,
        num_decoder_layers=3,
        latent_dim=128,
        cond_dim=32,
        dropout=0.1,
        max_len=160,
        pad_id=0,
        sos_id=2,
        eos_id=3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.dropout_p = dropout
        self.max_len = max_len       # 实际传入128+8
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id

        # ---------- 共享 embedding ----------
        # encoder 和 decoder 共用一张词向量表：参数更少，且两边对同一 token 的理解一致
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # ---------- 位置嵌入 ----------
        # 这里用一张可学习的位置查找表（第 i 个位置对应第 i 行向量），
        # 比正弦位置编码那套公式直观得多，序列长度固定在 max_len 136 以内时效果没有区别。
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # ---------- 条件通路 ----------
        # 输入是 1 维标量（归一化后的 LogP），升到 cond_dim 维再送进网络。
        # 直接把一个 0~1 的标量拼到几百维的特征上，信号会被淹没，所以要先升维。
        self.cond_mlp = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.ReLU(),
        )

        # ---------- Encoder ----------
        # 编码器不加因果掩码，每个位置都能看到整条序列，相当于 GRU 版里的"双向"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,        # 输入形状按 [B, L, D] 而不是 PyTorch 默认的 [L, B, D]
            # norm_first=True 就是 Pre-LN：先做 LayerNorm 再进子层。
            # 原版 Transformer 的 Post-LN 必须配学习率 warmup 才稳，否则前几百步梯度就炸；
            # Pre-LN 不用 warmup 也能收敛，正好这个项目的训练脚本用的是固定学习率。
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,   # 与 norm_first=True 不兼容，关掉免得刷警告
        )
        self.fc_mu = nn.Linear(d_model + cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(d_model + cond_dim, latent_dim)

        # ---------- Decoder ----------
        # 标准 seq2seq Transformer 的解码器有两种注意力：
        #   self-attention  ：看自己已经生成的前缀（要加因果掩码）
        #   cross-attention ：看编码器给的 memory
        # 在 VAE 里"编码器的输出"已经被压成了定长向量，所以 memory 只有两个 token：
        # 一个装 z（分子结构信息），一个装 c（目标 LogP）。
        # 解码器每一层都会去看它们，z 和条件的影响就不会随着生成越走越淡。
        self.z_token = nn.Linear(latent_dim, d_model)
        self.c_token = nn.Linear(cond_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, vocab_size)

        # ---------- 属性预测头（辅助任务） ----------
        # 从 z 直接回归归一化 LogP。它的梯度会反传进 encoder，
        # 迫使潜空间沿某个方向编码亲脂性，这正是后面条件生成能生效的前提。
        self.property_head = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    # ==================== 小工具 ====================

    def _add_position(self, token_emb):
        """
        给词向量加上位置嵌入。

        输入：
            token_emb: FloatTensor [B, L, d_model]
        输出：
            FloatTensor [B, L, d_model]
        """
        positions = torch.arange(token_emb.size(1), device=token_emb.device)
        return token_emb + self.pos_embedding(positions)                 # 广播到整个 batch

    # ==================== 条件 ====================

    def embed_condition(self, cond, batch_size, device):
        """
        把条件标量变成 cond_dim 维向量；cond 为 None 时返回全零"空条件"。

        输入：
            cond      : FloatTensor [B, 1] | None，归一化到 [0,1] 的 LogP
            batch_size: int，cond 为 None 时用来确定输出形状
            device    : torch.device
        输出：
            FloatTensor [B, cond_dim]
        """
        if cond is None:
            # 阶段1（无条件预训练）走这条分支：条件通道恒为 0，
            # 相当于告诉模型"这次不给 LogP 要求"，阶段2 再学非零条件的含义
            return torch.zeros(batch_size, self.cond_dim, device=device)
        return self.cond_mlp(cond)

    # ==================== Encoder ====================

    def encode(self, input_ids, lengths, cond_emb):
        """
        把 SMILES 编码成后验分布参数 (mu, logvar)。

        输入：
            input_ids: LongTensor  [B, L]，含 <sos>/<eos>，右侧 <pad> 补齐
            lengths  : LongTensor  [B]，每条序列的真实长度。
                       Transformer 直接用 pad_id 推出掩码，这个参数只是为了和
                       GRU 版保持同一个函数签名（训练脚本不用改），内部没有用到。
            cond_emb : FloatTensor [B, cond_dim]
        输出：
            (mu, logvar)，均为 FloatTensor [B, latent_dim]
        """
        pad_mask = input_ids.eq(self.pad_id)                             # [B, L]，True 表示 pad

        h = self._add_position(self.embedding(input_ids))                # [B, L, D]
        # src_key_padding_mask 让注意力权重跳过 pad 位置。
        h = self.encoder(h, src_key_padding_mask=pad_mask)               # [B, L, D]  通过自注意力获得每个token的上下文表示

        # 掩码均值池化，得到整分子的定长表示。
        # 对所有真实 token 求平均，pad 位置不参与。
        keep = (~pad_mask).unsqueeze(-1).to(h.dtype)                     # [B, L] → [B, L, 1]
        pooled = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)  # [B, D] 分子级向量表示

        feat = torch.cat([pooled, cond_emb], dim=-1)                     # [B, D + cond]，预训练时cond_emb为全零
        mu = self.fc_mu(feat)                                            # [B, latent_dim]
        # logvar 夹在 [-6, 2]：不夹的话早期训练容易出现方差爆炸 -> exp 溢出 -> NaN
        logvar = torch.clamp(self.fc_logvar(feat), min=-6.0, max=2.0)    # [B, latent_dim]
        return mu, logvar

    @torch.no_grad()
    def encode_mu(self, input_ids, lengths, cond=None):
        """
        编码一批分子，只返回后验均值 mu（给潜空间扩散当 x_0）。

        用均值而不是重参数化采样，避免同一分子每次编码漂到不同位置。
        """
        was_training = self.training
        self.eval()
        # 把条件[B, 1]变成 [B, cond_dim] ；cond 为 None 时返回[B, cond_dim]的全零"空条件"。
        cond_emb = self.embed_condition(cond, input_ids.size(0), input_ids.device)
        # 传入编码器获得样本的均值和log方差，俩形状都为[B, latent_dim]
        mu, _ = self.encode(input_ids, lengths, cond_emb)
        # 恢复训练模式
        if was_training:
            self.train()
        return mu

    @staticmethod
    def reparameterize(mu, logvar):
        """
        重参数化采样：z = mu + sigma * eps。   z是潜变量，mu是均值，sigma是标准差，eps是随机噪声

        记号说明：N(mu, sigma^2) 的第二个参数按惯例是方差而不是标准差，
        所以从 logvar（= log(sigma^2)）还原 sigma 要先 exp 再开方，合起来就是 exp(0.5 * logvar)。

        直接从 N(mu, sigma^2) 采样是不可导的；把随机性挪到与参数无关的 eps 上之后，
        z 对 mu / logvar 就变成了可导的确定性函数，梯度才能回传到 encoder。

        输入：
            mu, logvar: FloatTensor [B, latent_dim]
        输出：
            z: FloatTensor [B, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ==================== Decoder ====================

    def build_memory(self, z, cond_emb):
        """
        把 z 和条件打包成解码器要看的 memory（一条长度为 2 的"源序列"）。

        输入：
            z       : FloatTensor [B, latent_dim]
            cond_emb: FloatTensor [B, cond_dim]
        输出：
            FloatTensor [B, 2, d_model]
        """
        return torch.stack([self.z_token(z), self.c_token(cond_emb)], dim=1)

    def decode(self, memory, dec_input):
        """
        教师强制（teacher forcing）解码：给定真实的前缀 token，预测下一个 token。

        整段序列一次性并行算完，靠因果掩码保证第 i 个位置只能看到 <= i 的位置。
        这正是 Transformer 相对 GRU 在训练上的优势：GRU 的 L 个时间步必须一步步串行，
        Transformer 的 L 个位置一次矩阵乘法就全出来了。

        输入：
            memory   : FloatTensor [B, 2, d_model]，build_memory 的输出
            dec_input: LongTensor  [B, L]，解码器输入（原序列去掉最后一个 token）
        输出：
            logits: FloatTensor [B, L, vocab_size]，每个位置对下一个 token 的打分
        """
        seq_len = dec_input.size(1)

        # 因果掩码：一个 [L, L] 的上三角矩阵，被遮住的位置填 -inf，
        # 过 softmax 后权重变成 0，第 i 个位置就看不到 i 之后的任何 token。
        # 没有这个掩码，训练时模型能直接"看答案"，生成时却看不到，两边对不上。
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=dec_input.device
        )

        h = self._add_position(self.embedding(dec_input))                # [B, L, D]
        h = self.dropout(h)
        # 这里不传 tgt_key_padding_mask：因为 pad 全在序列末尾，
        # 而因果掩码保证真实 token 只会往前看，永远看不到后面的 pad；
        # pad 位置自己的输出会被损失函数的 ignore_index 丢掉，算了也不影响。
        # TransformerDecoder官方设计就是「前缀走 tgt、外部信息走 memory」，所以必须造这份长度为 2 的源
        h = self.decoder(tgt=h, memory=memory, tgt_mask=causal_mask)     # [B, L, D]
        return self.out_proj(h)                                          # [B, L, V]

    # ==================== 前向 ====================

    def forward(self, input_ids, lengths, dec_input, cond=None, cond_dropout=0.0):
        """
        完整前向：编码 -> 采样 -> 解码 -> 属性预测。

        输入：
            input_ids   : LongTensor  [B, L]
            lengths     : LongTensor  [B]
            dec_input   : LongTensor  [B, L-1]
            cond        : FloatTensor [B, 1] | None
            cond_dropout: float，训练时以该概率把某些样本的条件替换成空条件。
                          这是 classifier-free guidance 的训练侧做法：模型必须同时学会
                          "有条件"和"无条件"两种生成，生成时才能用两者的差值放大条件的作用。
                          顺带也防止模型把条件当成可有可无的装饰而直接忽略。
        输出：
            dict：
                logits    : [B, L-1, V] 解码 logits
                mu        : [B, latent_dim]
                logvar    : [B, latent_dim]
                z         : [B, latent_dim]
                prop_pred : [B, 1] 由 z 预测的归一化 LogP
        """

        # 把条件[B, 1]变成 [B, cond_dim] ；cond 为 None 时返回[B, cond_dim]的全零"空条件"。
        cond_emb = self.embed_condition(cond, input_ids.size(0), input_ids.device)

        # self.training 是nn.Module 自带的布尔标志，表示当前模块是否在训练模式下
        if self.training and cond is not None and cond_dropout > 0:
            # torch.rand(size)：创建在[0,1) 上均匀分布的、形状[B, 1]的张量。
            # < cond_dropout逐元素比较，True/False
            drop = torch.rand(cond_emb.size(0), 1, device=cond_emb.device) < cond_dropout
            # torch.where(condition, x, y)：逐位置的三元选择，如果条件为True，则返回x，否则返回y。
            # 这三行是在做条件 dropout：训练时按概率把某些样本的条件向量清零
            cond_emb = torch.where(drop, torch.zeros_like(cond_emb), cond_emb)

        mu, logvar = self.encode(input_ids, lengths, cond_emb)   # 传入编码器获得样本的均值和log方差，俩形状都为[B, latent_dim]
        z = self.reparameterize(mu, logvar)    # 重参数化采样，得到[B, latent_dim]的潜向量
        logits = self.decode(self.build_memory(z, cond_emb), dec_input)   # 输出[B, L, V]未归一化分数，V是词表长度
        
        # ---------- 属性预测头（辅助任务） ----------
        # 从 z 直接回归归一化 LogP。它的梯度会反传进 encoder，
        # 迫使潜空间沿某个方向编码亲脂性，这正是后面条件生成能生效的前提。
        prop_pred = self.property_head(z)   # [B, 1]
        
        return {"logits": logits, "mu": mu, "logvar": logvar, "z": z, "prop_pred": prop_pred}

    # ==================== 采样生成 ====================

    @torch.no_grad()
    def sample(self, num_samples, max_len, device, cond_value=None, z=None,
               temperature=1.0, greedy=False, guidance_scale=1.0):
        """
        从先验 N(0, I) 采样 z 并自回归生成 SMILES 的 token id 序列。

        生成方式是最朴素的那种：把"已经生成的整段前缀"重新喂一遍解码器，
        只取最后一个位置的预测当作下一个 token。
        GRU 只需要传一个隐状态就能往前走一步，Transformer 没有这种状态，
        所以每一步都要重算一遍——写起来简单，代价是慢，这里选了简单。

        输入：
            num_samples   : int,   生成条数
            max_len       : int,   最大生成步数（防止不收敛的序列无限长）
            device        : torch.device
            cond_value    : float | FloatTensor[B,1] | None，归一化到 [0,1] 的目标 LogP。
                            None 表示无条件生成（走空条件通路）。
            z             : FloatTensor [B, latent_dim] | None，指定潜向量；None 则从先验采样
            temperature   : float, 采样温度。>1 更发散、结构更多样但有效率下降；
                            <1 更保守、更接近训练分布
            greedy        : bool,  True 则每步取 argmax（同一个 z 结果确定），
                            False 则按温度缩放后的分布多项式采样
            guidance_scale: float, classifier-free guidance 强度。
                            =1 表示普通条件生成；>1 时每一步同时算"有条件"和"无条件"两套
                            logits，按 uncond + w * (cond - uncond) 外推，
                            把条件带来的那部分差异放大 w 倍。
                            代价是每步要跑两遍解码器，速度减半。
        输出：
            LongTensor [num_samples, <=max_len]，生成的 token id（不含起始 <sos>）
        """
        self.eval()

        if z is None:
            # 从先验N(0, I)中采样num_samples条潜向量，形状为[num_samples, latent_dim]
            z = torch.randn(num_samples, self.latent_dim, device=device)
        else:
            num_samples = z.size(0)

        # 位置嵌入表是定长的136，序列长度不能超过它。位置编码查的是下标 0 ... 136-1
        # max_len 这个参数数的是生成步数（不含 <sos>）。若不减 1，调用方传 136 时，最后一步前缀长度会到 137，arange(137) 会去查下标 136，表里没有这一行。
        max_len = min(max_len, self.max_len - 1)

        # 组装条件张量为[num_samples, 1]的矩阵
        if cond_value is None:
            cond = None
        elif isinstance(cond_value, torch.Tensor):          # 给了一批条件，不同的目标 LogP
            cond = cond_value.to(device).view(-1, 1).float()
        else:   # 只给了一个条件，所有样本都一样
            cond = torch.full((num_samples, 1), float(cond_value), device=device)

        # 把条件变成[num_samples, cond_dim]的矩阵；cond 为 None 时返回的全零"空条件"。
        cond_emb = self.embed_condition(cond, num_samples, device)

        # 先将z和cond_emb线性变换为d_model维，然后stack成[num_samples, 2, d_model]的memory，作为交叉注意力读取的源序列
        memory = self.build_memory(z, cond_emb)

        # guidance 需要一条并行的"无条件"解码流：同样的 z，条件换成全零，
        # 两条流喂入完全相同的前缀，只有 memory 里的条件那一格不一样
        use_guidance = guidance_scale != 1.0 and cond is not None
        if use_guidance:
            null_emb = torch.zeros(num_samples, self.cond_dim, device=device)
            memory_null = self.build_memory(z, null_emb)

        # 所有序列都从 <sos> 起步；prefix 会随着生成一步步变长
        prefix = torch.full((num_samples, 1), self.sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(num_samples, dtype=torch.bool, device=device)

        for _ in range(max_len):
            # decode 会给出每个位置的预测，但只有最后一个位置是"还没生成的下一个 token"的未归一化分数
            # prefix是已经生成的token id序列，形状为[num_samples, 当前长度]
            # NumPy / PyTorch 的规则是：用整数取出某一维上的一个位置，这一维会从形状里消失；只有切片才会把维数留下来。
            logits = self.decode(memory, prefix)[:, -1, :]               # [num_samples, V]

            # 放大条件影响：classifier-free guidance
            if use_guidance:   
                logits_null = self.decode(memory_null, prefix)[:, -1, :]
                logits = logits_null + guidance_scale * (logits - logits_null)

            # 禁止生成 <pad>：它只是补位符号，出现在中间会让序列提前"断掉"
            # 取出的是每一行里「<pad> 那一列」，把它们改成 -1e9（一个很大的负数），后面无论走哪条路，<pad> 都几乎不可能被选中
            logits[:, self.pad_id] = -1e9

            # greedy贪心解码=True：每步取 argmax（同一个 z 结果确定，改条件看是否生效），False 则按温度缩放后的分布多项式采样
            if greedy:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                # 按温度缩放后的概率分布，得到每个token的归一化概率
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)    # [num_samples, V]
                # torch.multinomial 是按给定概率做有放回（默认）抽样：每一行是一个离散分布，函数按这些概率随机抽出类别下标。
                next_token = torch.multinomial(probs, num_samples=1)      # [num_samples, 1]

            # 已经结束的序列强制吐 <pad>，避免 <eos> 之后的内容被拼进 SMILES
            # torch.where(condition, x, y)：逐位置选择。如果condition为True的位置，返回x，否则返回y。
            next_token = torch.where(finished.unsqueeze(1),                       # 比较（如果有）发生在造 condition 的时候。where 只根据已经算好的真假，在两份取值里挑一个
                                     torch.full_like(next_token, self.pad_id),    # 条件为真，造一个和next_token形状相同的全0张量
                                     next_token)                                  # 如果条件为False，则返回next_token
            prefix = torch.cat([prefix, next_token], dim=1)                  # 将next_token拼接到prefix后面，更新已经生成的token序列

            # 这一轮生成 eos 结束符的样本，之后每一位都强制吐 pad，直到所有样本都生成 eos
            finished = finished | (next_token.squeeze(1) == self.eos_id)      # 如果next_token中某一1等于eos_id，则将finished的对应位置设为True
            if bool(finished.all()):
                break

        return prefix[:, 1:]                                             # 去掉开头的 <sos>

    def steer_latent(self, z, target_norm, steps=60, lr=0.05, prior_weight=0.05):
        """
        潜空间导向：微调 z，使属性头对它的预测靠近目标 LogP。

        为什么需要这一步：
            生成时 z 是从先验 N(0, I) 里随便采的，它和目标条件 y 在统计上是独立的。
            如果 z 本身就编码了偏离目标的 LogP，再怎么给条件也拧不过来。
            这里用属性头当可微的导航仪，把 z 主动挪到目标 LogP 那一侧，
            再连同条件一起解码，z 和 y 就不再互相打架了。
            这就是 ChemVAE 那一类工作里的 latent space optimization，只是把
            贝叶斯优化换成了更直接的梯度下降。

        prior_weight 那一项不能省：z 一旦被推到离原点很远的地方，
        就跑出了解码器训练时见过的区域，解出来的全是废串。
        这一项把 z 拉回先验的高密度区，是在"性质对准"和"还能解码"之间做权衡。

        输入：
            z           : FloatTensor [B, latent_dim]，先验采样得到的初始潜向量
            target_norm : float，归一化到 [0,1] 的目标 LogP
            steps       : int，梯度步数
            lr          : float，学习率
            prior_weight: float，先验正则项权重
        输出：
            FloatTensor [B, latent_dim]，调整后的 z（已 detach）
        """
        was_training = self.training
        self.eval()
        # 只优化 z，模型参数保持不动
        for p in self.parameters():
            p.requires_grad_(False)

        # 切断和采样图的联系，给 z 单独开梯度通道
        z = z.clone().detach().requires_grad_(True)
        target = torch.full((z.size(0), 1), float(target_norm), device=z.device)
        # 生成时把优化对象从模型权重换成已经采出的z 的梯度
        optimizer = torch.optim.Adam([z], lr=lr)

        for _ in range(steps):
            pred = self.property_head(z)
            loss = F.mse_loss(pred, target) + prior_weight * z.pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for p in self.parameters():
            p.requires_grad_(True)
        if was_training:
            self.train()
        return z.detach()

    # ==================== 存盘辅助 ====================

    def config_dict(self):
        """
        导出重建该模型所需的全部结构超参，随权重一起存进 checkpoint。

        这样 generate.py / evaluate.py 不必再去读 config.py 猜结构，
        即使以后改了 config 的默认值，旧 checkpoint 也能正确加载。

        输入：
            无
        输出：
            dict，可直接 SmilesVAE(**cfg) 还原模型
        """
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "dim_feedforward": self.dim_feedforward,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "latent_dim": self.latent_dim,
            "cond_dim": self.cond_dim,
            "dropout": self.dropout_p,
            "max_len": self.max_len,
            "pad_id": self.pad_id,
            "sos_id": self.sos_id,
            "eos_id": self.eos_id,
        }
