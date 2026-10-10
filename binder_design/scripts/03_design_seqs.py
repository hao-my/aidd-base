#!/usr/bin/env python
"""阶段 3：ProteinMPNN-FastRelax 给骨架填序列。

RFdiffusion 输出的 PDB 序列全是甘氨酸占位，这一步产生真正的氨基酸序列。

以下四点按 dl_binder_design 上游源码核对过，和网上多数教程的写法不同：

1. -relax_cycles > 0 与 -seqs_per_struct > 1 不能同时用，源码里会直接抛异常。
   所以固定 relax_cycles=0 + seqs_per_struct=2。
2. 这一步不需要 GPU。上游说明 MPNN 上 GPU 只快一点点，FastRelax 根本用不上 GPU。
   把它放 CPU 上能省 GPU 计时。
3. 走 -pdbdir / -outpdbdir 而不是 silent 文件。上游支持直接用 pdb 目录，
   这样能避开 silent_tools 的依赖问题，而且下游 AF2 可以用 -runlist 分片。
4. -temperature 默认是 1e-6（近似 argmax），不是独立版 ProteinMPNN 教程里的 0.1。
   -omit_AAs 默认 'CX'，即默认就不生成半胱氨酸。两者都在 config 里显式写出来。

脚本自带断点续跑：上游用 -checkpoint_name 记录已完成的 tag，
每组一个独立的 checkpoint 文件，避免组间串台。

用法（proteinmpnn_binder_design 环境）:
    python scripts/03_design_seqs.py --campaign config/campaign_pilot.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

from common import OUTPUTS, die, load_tools, load_yaml, record_throughput, run


def group_names(camp: dict) -> list[str]:
    names = [g["name"] for g in camp["diffusion"]["groups"]]
    if camp.get("negative_control", {}).get("enabled"):
        names.append("negctrl")
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="config/campaign_pilot.yaml")
    parser.add_argument("--only-group")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    camp = load_yaml(args.campaign)
    tools = load_tools(require=not args.dry_run)
    seq_cfg = camp["sequence"]

    diff_root = OUTPUTS / "diffusion"
    if not diff_root.exists() and not args.dry_run:
        die(f"没有骨架目录 {diff_root}，先跑 02_diffuse.py")

    names = group_names(camp)     # noise00 、noise05 和 阴性对照 negctrl 三组组名列表
    if args.only_group:
        names = [n for n in names if n == args.only_group]
        if not names:
            die(f"配置里没有组 {args.only_group}")

    # Python 拿到的一份环境变量表，内容来自启动这个脚本之前在终端里执行的 export
    env = dict(os.environ)
    if seq_cfg.get("device") == "cpu":
        # 让 torch 看不到 GPU，把显存留给 AF2
        env["CUDA_VISIBLE_DEVICES"] = ""

    for name in names:
        pdb_dir = diff_root / name
        # 把各组内pdb骨架文件按 文件名的字符串排序成列表，noise00_10.pdb 会排在 noise00_2.pdb 前面
        backbones = sorted(pdb_dir.glob("*.pdb")) if pdb_dir.exists() else []  
        if not backbones:
            print(f"[skip] 组 {name} 没有骨架")
            continue

        out_dir = OUTPUTS / "sequences" / name        # 序列文件保存路径
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = out_dir / "mpnn.check.point"      # 检查点文件，记录已完成的骨架名称

        cmd = [
            sys.executable,                    # sys模块属性，是当前 Python 解释器。在云主机上它属于已经激活的 proteinmpnn_binder_design 环境。
            tools.dl_interface_design,         # 是 DL_BINDER_DIR/mpnn_fr/dl_interface_design.py，也就是 ProteinMPNN 的界面设计脚本。
            "-pdbdir",
            pdb_dir,
            "-outpdbdir",
            out_dir,
            "-checkpoint_name",                
            checkpoint,                        # 检查点文件，记录已完成的骨架名称，断点续跑时跳过已完成的骨架
            "-relax_cycles",
            str(seq_cfg["relax_cycles"]),      # Rosetta FastRelax 松弛循环轮数，大于 0 时会在选定序列后轻微调整骨架和侧链坐标
            "-seqs_per_struct",
            str(seq_cfg["seqs_per_struct"]),   # 每条骨架采样几条序列。每条骨架多条序列 和 Rosetta FastRelax 松弛循环轮数 不能同时用
            "-temperature",
            str(seq_cfg.get("temperature", 1e-6)), # ProteinMPNN 采样温度，默认 1e-6（近似 argmax）
            "-omit_AAs",
            str(seq_cfg.get("omit_aas", "CX")),    # -omit_AAs 是禁止使用的氨基酸，默认 "CX"：C 是半胱氨酸，容易形成计划外的二硫键；X 是非标准残基。
        ]
        if seq_cfg.get("use_soluble_weights"):
            cmd += ["-checkpoint_path", tools.soluble_weights]   # 使用溶解度权重文件，提高亲水性氨基酸的生成概率

        n_todo = len(backbones)
        if checkpoint.exists():
            done = len(     # 检查点文件里已经完成的骨架条数。它等于 mpnn.check.point 中非空行的个数
                # read_text(encoding="utf-8") 把整个文件读成一个字符串。encoding="utf-8" 指定按 UTF-8 解码。
                # splitlines() 按换行把这个字符串切成多行，换行符本身不留在每一行里。
                [ln for ln in checkpoint.read_text(encoding="utf-8").splitlines() if ln.strip()]
            )
            n_todo = max(len(backbones) - done, 0)
            print(f"[resume] 组 {name}: 已完成 {done}/{len(backbones)} 个骨架")
            if n_todo == 0:
                print(f"[skip] 组 {name} 已全部完成")
                continue

        # 启动拼好的 ProteinMPNN 命令，等到它结束，再把花费的秒数存进 elapsed。如果 dry_run 为真，只打印命令不实际运行。
        elapsed = run(cmd, env=env, dry_run=args.dry_run)
        if not args.dry_run:
            record_throughput(   # 记录吞吐量，写入 output/sequence/throughput.json
                f"sequence.{name}",
                n_todo * int(seq_cfg["seqs_per_struct"]),
                elapsed,
                note=(
                    f"device={seq_cfg.get('device')} "
                    f"relax_cycles={seq_cfg['relax_cycles']} "
                    f"soluble={bool(seq_cfg.get('use_soluble_weights'))}"
                ),
            )

    print("\n序列设计完毕，下一步 04_predict_af2.py")


if __name__ == "__main__":
    main()
