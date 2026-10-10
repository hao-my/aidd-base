# 从头设计 PD-L1 结合蛋白

以 PD-L1（PDB 4ZQK 链 A，残基 18–132）为靶点，按 RFdiffusion、可溶 ProteinMPNN、AF2 initial guess 做从头结合剂设计，并用噪声消融和背面热点对照检验漏斗是否在看界面。全量筛选之后又做了两步加深：在已通过的骨架上增加序列采样，以及对通过设计计算 Rosetta 结合能和结合剂 SAP。

设计热点为 A56、A115、A121、A123，阴性对照热点为 A31、A32、A33、A105，结合剂长度 80–120。配置在 `config/target_pdl1.yaml` 与 `config/campaign_full.yaml`。

三条主过滤来自 Watson 等（*Nature* 2023）的 in silico success 定义，阈值沿用 Bennett 等（*Nat Commun* 2023）的实验校准，写在 `config/filters.yaml`：

| 指标 | 阈值 | 含义 |
|---|---|---|
| `plddt_binder` | > 80 | binder 折叠置信度 |
| `pae_interaction` | < 10 | 链间预测对齐误差 |
| `binder_aligned_rmsd` | < 1 Å | AF2 预测的 binder 相对设计骨架的偏移 |

这些指标用于判断计算上是否像会结合，不能预测亲和力。

## 流程

```
靶点 4ZQK
   │  01  裁出 PD-L1 IgV，从 PD-1 界面确定 hotspot
   ▼
RFdiffusion ── 02 ── 骨架（甘氨酸占位）
   │                 noise 0 / noise 0.5 / 背面 hotspot
   ├─ TM-score  结合剂链按 0.6 聚类，比较 noise 0 与 0.5 的骨架多样性
   ▼
ProteinMPNN ── 03 ── 可溶权重，每骨架 2 条序列
   ▼
AF2 initial guess ── 04 ── 复合物 + 置信度
   ▼
05  三级过滤、序列去冗余、top 20
   │
   ├─ 加深 1  数据增强：界面 pAE 最好的 150 个骨架
   │          温度 0.1、每骨架 8 条序列，再走 03 → 04 → 05
   │
   └─ 加深 2  Rosetta：1 轮 FastRelax 后计算 dG_separated，
              删掉靶点链，计算结合剂 SAP
```

RFdiffusion 只生成主链。ProteinMPNN 给结合剂填序列，靶点序列保持晶体原样。AF2 initial guess 把设计坐标送进第一轮 recycle，固定靶点、预测结合剂，检查这条序列是否折回设计构象。GPU 阶段（02、03、04）的调用方式没有改过。

对照有两项，参数除标注的轴以外与全量一致：

- 消融：推理噪声 `noise_scale` 取 0 与 0.5。
- 阴性对照：hotspot 换到靶点背面。

## 怎么跑

在 `binder_design/` 目录下执行。外部程序的安装和权重见 [`env/SETUP.md`](env/SETUP.md)。02、03、04 需要：

```bash
export RFDIFFUSION_DIR=$HOME/program/RFdiffusion
export DL_BINDER_DIR=$HOME/program/dl_binder_design
```

### 01 靶点准备

本机。环境：`biopython`、`pyyaml`。不需要 GPU。

```bash
python scripts/01_prepare_target.py --target config/target_pdl1.yaml
```

热点已写入 `config/target_pdl1.yaml`，正式计算没有重跑这一步。

### 02 骨架

Linux GPU。环境：`SE3nv`。

```bash
conda activate SE3nv
python scripts/02_diffuse.py --campaign config/campaign_full.yaml
```

按组调用 RFdiffusion。`noise00`、`noise05` 各 1000 条骨架，`negctrl` 300 条。输出在 `outputs/diffusion/<组名>/`。

### 03 序列

同一台 Linux 机器，CPU 即可。环境：`proteinmpnn_binder_design`。

```bash
conda activate proteinmpnn_binder_design
python scripts/03_design_seqs.py --campaign config/campaign_full.yaml
```

可溶权重 `v_48_020.pt`，温度 `1e-6`，每个骨架 2 条序列，`relax_cycles=0`。输出在 `outputs/sequences/<组名>/`。

### 04 AF2

Linux GPU。环境：`af2_binder_design`。模型固定为 `model_1_ptm`，recycle 3。

```bash
conda activate af2_binder_design
python scripts/04_predict_af2.py --campaign config/campaign_full.yaml
```

按 250 条一片写出 `outputs/af2/<组名>/scores_*.sc` 和 `pred_*/`。binder 必须是输入 PDB 的第一条链。

### 05 过滤

本机。环境：`pandas`、`pyyaml`。不需要 GPU。

```bash
python scripts/05_filter_rank.py --campaign config/campaign_full.yaml
```

写出 `outputs/scores.csv`、`passing.csv`、`topn.csv`、`funnel.json`。全量图：

```bash
python analysis/plots.py
```

环境再加 `matplotlib`。图在 `outputs/figures/`。

### 结构多样性：TM-score 聚类

本机 CPU。不参与过滤，不改 `scores.csv`、`passing.csv`、`topn.csv`、`funnel.json`。对 `noise00` 和 `noise05` 各 1000 条扩散骨架只取结合剂链 B，用 TM-align 计算 TM-score（按较短链归一化），以 0.6 为界贪心聚类。这是 Watson 等 Extended Data Fig. 1i 用来衡量降噪损失多样性的做法。05 的序列去冗余仍然负责 top 20。

环境是 Python 3.12 虚拟环境，因为 tmtools 的 Windows 安装包不到 3.13。首次：

```powershell
py -3.12 -m venv .venv-tmtools
.\.venv-tmtools\Scripts\python.exe -m pip install tmtools
```

之后：

```powershell
.\.venv-tmtools\Scripts\python.exe analysis\tmscore_cluster.py
```

写出 `outputs/tmscore/noise00_clusters.csv`、`noise05_clusters.csv` 和 `summary.json`。配对矩阵写在同目录的 `*_scores.f32`，中断后重跑只补未完成的配对。全量约每组 50 万次比较，多核也要数小时。

### 加深 1：序列数据增强

不重跑扩散。从全量通过设计里，每个骨架只留界面 pAE 最低的序列，再取前 150 个骨架。本机，仅用 Python 标准库：

```bash
python scripts/select_refine150.py
```

骨架以符号链接放到 `outputs/diffusion/refine150/`，名单在 `outputs/refine150_selection.csv`。然后用加深配置重跑序列设计和 AF2。03、04 的环境与上面相同：

```bash
conda activate proteinmpnn_binder_design
python scripts/03_design_seqs.py --campaign config/campaign_refine150.yaml

conda activate af2_binder_design
python scripts/04_predict_af2.py --campaign config/campaign_refine150.yaml
```

`campaign_refine150.yaml` 把温度改为 0.1、每个骨架 8 条序列。过滤写到单独目录，避免覆盖全量结果。本机：

```bash
python scripts/05_filter_rank.py \
  --campaign config/campaign_refine150.yaml \
  --output-dir outputs/refine150_rank
```

### 加深 2：结合能与结合剂 SAP

本机或 Linux CPU。环境：PyRosetta。不需要 GPU。对 05 的通过表逐条做 1 轮 `ref2015` FastRelax，用 InterfaceAnalyzer 计算 `dG_separated`（`pack_separated` 打开），再删掉链 B，用 `SapScoreMetric` 计算结合剂 SAP。`pass_ddg`（< −30）和 `pass_sap`（< 45）只作标记，不删行。

```bash
python scripts/06_interface_metrics.py
```

默认读 `outputs/passing.csv`，PDB 在 `outputs/af2/<组名>/pred_*/`。全部完成后写出 `outputs/rosetta_interface.csv`。检查点为同目录的 `rosetta_interface.checkpoint.csv`，中断后重跑只补未完成的条。给序列加深的通过表单独打分时：

```bash
python scripts/06_interface_metrics.py \
  --scores outputs/refine150_rank/passing.csv
```

结果写到 `outputs/refine150_rank/rosetta_interface.csv`，不覆盖全量表。

## 结果

全量 4600 次 AF2，通过 914 条。`noise00` 通过率 0.248，`noise05` 为 0.209，背面对照为 0。背面组的 pLDDT 和骨架 RMSD 仍然大多通过，界面 pAE 全部失败，漏斗卡的是界面。去冗余后 843 簇，top 20 在 `outputs/topn.csv`。最佳为 `noise00_249`，界面 pAE 3.569。

序列加深 1200 次预测，通过 874 条，通过率 0.728，聚类 616。最佳为 `noise00_119`，界面 pAE 3.488。漏斗在 `outputs/refine150_rank/funnel.json`。

已完成的 Rosetta 分数在 `outputs/interface_scores.csv`（914 行）：882 条 `dG_separated` < −30，中位数 −43.05；结合剂 SAP 中位数 35.87，744 条 < 45。

## 目录

```text
binder_design/
├─ config/
│  ├─ target_pdl1.yaml         # 靶点、热点、binder 长度
│  ├─ campaign_full.yaml       # 全量规模与阴性对照
│  ├─ campaign_refine150.yaml  # 序列加深：温度 0.1，每骨架 8 条
│  └─ filters.yaml
├─ scripts/
│  ├─ 01_prepare_target.py
│  ├─ 02_diffuse.py
│  ├─ 03_design_seqs.py
│  ├─ 04_predict_af2.py
│  ├─ 05_filter_rank.py
│  ├─ select_refine150.py      # 从全量通过设计中选 150 个骨架
│  └─ 06_interface_metrics.py  # dG_separated 与结合剂 SAP
├─ analysis/plots.py
├─ analysis/tmscore_cluster.py  # noise00 与 noise05 的 TM-score 聚类
├─ env/SETUP.md
└─ outputs/
```

## 已知局限

- AF2 置信度只做计算上的结合判断，与亲和力没有对应关系。
- 没有湿实验。结合能是 Rosetta 能量差（REU），不是实验结合自由能。
- top 20 的去冗余仍按序列一致度 0.70。TM-score 聚类只比较两组骨架的结构多样性，不改通过率和候选名单。
- 单靶点，结论不能外推到其他蛋白。

## 参考

- Watson et al. De novo design of protein structure and function with RFdiffusion. *Nature* 2023. <https://www.nature.com/articles/s41586-023-06415-8>
- Zhang and Skolnick. Scoring function for automated assessment of protein structure template quality. *Proteins* 2004. TM-score。
- Bennett et al. Improving de novo protein binder design with deep learning. *Nat Commun* 2023. <https://www.nature.com/articles/s41467-023-38328-5>
- Dauparas et al. ProteinMPNN. <https://github.com/dauparas/ProteinMPNN>
- Pacesa et al. One-shot design of functional protein binders with BindCraft. *Nature* 2025. <https://www.nature.com/articles/s41586-025-09429-6>
