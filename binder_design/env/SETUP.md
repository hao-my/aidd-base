# 租卡环境部署清单

本机（AMD 核显、无 CUDA）跑不了 RFdiffusion 和 AF2，GPU 阶段必须在租的
Linux 实例上跑。本机只负责阶段 1 的靶点准备和阶段 5 之后的分析出图。

推荐实例：24 GB 显存（4090 / A5000 / 3090 均可），32 GB 内存起，
盘至少 80 GB（AF2 权重约 5 GB，加上中间 PDB）。

## 分工

| 阶段 | 跑在哪 | 环境 |
|---|---|---|
| 01 靶点准备 | 本机 | `pip install biopython pyyaml` |
| 02 扩散 | 租的 GPU | `SE3nv` |
| 03 序列设计 | 租的机器（CPU 即可） | `proteinmpnn_binder_design` |
| 04 AF2 预测 | 租的 GPU | `af2_binder_design` |
| 05 过滤 / 出图 | 本机 | `pip install pandas matplotlib pyyaml` |

阶段 4 结束后只需要把 `outputs/af2/*/scores_*.sc` 和
`outputs/sequences/*/​*.pdb` 拉回本机就能做分析，不用下载全部结构。

## 前置：PyRosetta 许可

非商业用途（学术、非营利、个人学习）**不用申请**，也没有用户名密码。
2024 年 3 月起，许可随下载自动生效，下载页即视为接受
[Non-Commercial License](https://www.pyrosetta.org/downloads)。
商业用途或学术机构的有偿代工仍要向华盛顿大学购买许可：license@uw.edu。

网上和 `dl_binder_design` 旧文档里的
`https://USERNAME:PASSWORD@conda.graylab.jhu.edu` 已经过时。
conda 源去掉账号即可：

```bash
conda install -c https://conda.rosettacommons.org -c conda-forge pyrosetta
```

国内访问西海岸镜像慢时，换东海岸：

```bash
conda install -c https://conda.graylab.jhu.edu -c conda-forge pyrosetta
```

`dl_binder_design` 的 yml 把 PyRosetta 钉在 2023 年的某个构建上，
且 channel 仍写着旧的账号地址。创建环境前删掉 yml 里的 `pyrosetta` 那一行，
环境建好后再用上面的命令装当前版本。装完在对应环境里确认：

```bash
python -c "import pyrosetta; pyrosetta.init(); print('pyrosetta ok')"
```

## 1. RFdiffusion

```bash
git clone https://github.com/RosettaCommons/RFdiffusion.git
cd RFdiffusion
mkdir -p models && cd models
# binder 设计只需要 Complex_base_ckpt.pt；其余权重按需下载
wget http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt
cd ..

conda env create -f env/SE3nv.yml
conda activate SE3nv
cd env/SE3Transformer && pip install --no-cache-dir -r requirements.txt && python setup.py install
cd ../.. && pip install -e .
```

**最常见的坑**：装完发现 `未检测到 GPU！回退到 CPU`。原因是装成了 CPU 版
PyTorch。验证方式：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

返回 `False` 就重装匹配当前驱动 CUDA 版本的 PyTorch。先跑一个无条件设计确认能用：

```bash
./scripts/run_inference.py 'contigmap.contigs=[100-100]' \
  inference.output_prefix=test_out/test inference.num_designs=1
```

## 2. dl_binder_design

```bash
git clone https://github.com/nrbennet/dl_binder_design.git
cd dl_binder_design/include

# 序列设计环境
conda env create -f proteinmpnn_fastrelax.yml
conda activate proteinmpnn_binder_design
python importtests/proteinmpnn_importtest.py   # 应打印 passes all import tests

# AF2 环境：先用 nvidia-smi 看驱动支持的 CUDA 版本再选
#   CUDA 12 -> af2_binder_design.yml（新卡如 4090 选这个）
#   CUDA 11 -> dl_binder_design.yml
conda env create -f af2_binder_design.yml
conda activate af2_binder_design
python importtests/af2_importtest.py
```

`af2_importtest.py` 必须在能看到 GPU 的节点上跑，它会打印是否找到 GPU。

ProteinMPNN 代码要克隆到 `mpnn_fr` 下，脚本按这个相对路径找权重：

```bash
cd ../mpnn_fr && git clone https://github.com/dauparas/ProteinMPNN.git
```

AF2 权重放到 `af2_initial_guess/model_weights/params`：

```bash
cd ../af2_initial_guess
mkdir -p model_weights/params && cd model_weights/params
wget https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar --extract --verbose --file=alphafold_params_2022-12-06.tar
```

只需要 `params_model_1_ptm.npz`，脚本里写死用 `model_1_ptm`。
Google 存储在国内可能连不上，可以先下到本地再上传。

已知报错与处理：

| 报错 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: No module named 'mock'` | 环境缺 mock | `conda install mock` |
| `RuntimeError: Internal: Could not find the corresponding function` | JAX 找不到 nvcc | `conda install -c nvidia cuda-nvcc=<版本>` |
| `Struct with tag X failed in 0 seconds with error: <class 'EXCEPTION'>` | 主循环吞掉了真实异常 | 重跑时加 `-debug` 看真实堆栈 |

## 3. 环境变量

每次开新实例都要设：

```bash
export RFDIFFUSION_DIR=$HOME/program/RFdiffusion
export DL_BINDER_DIR=$HOME/program/dl_binder_design
export BD_GPU_LABEL="RTX4090-24G"   # 会记进 throughput.json，便于对比不同卡
```

## 4. 校验拼出来的命令

所有阶段脚本都支持 `--dry-run`，只打印命令不执行，不需要装任何外部工具。
上机前先在本机确认参数拼对了：

```bash
python scripts/02_diffuse.py --campaign config/campaign_pilot.yaml --dry-run
python scripts/03_design_seqs.py --campaign config/campaign_pilot.yaml --dry-run
python scripts/04_predict_af2.py --campaign config/campaign_pilot.yaml --dry-run
```
