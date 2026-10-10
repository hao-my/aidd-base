#!/usr/bin/env python
"""阶段 4：AF2 initial guess 预测复合物，产出过滤用的置信度分数。

这是整条流程的瓶颈，也是把成功率提高约一个数量级的那一步
（Bennett et al., Nat Commun 2023）。它与普通 AF2 的区别是：
把设计的复合物坐标灌进 pair representation 的初始 recycle，
并把靶点链当模板固定，只从单序列预测 binder。

两个必须遵守的输入约束（predict.py 源码里会直接报错）：
- binder 必须是第一条链
- binder 和靶点的残基编号不能重叠，否则 AF2 输出是垃圾
  （源码检测到重复编号会自动重编号，除非加 -maintain_res_numbering）

分片：-runlist 只在 -pdbdir 模式下生效，所以这里走 pdb 目录而非 silent。
每片一个独立的 scorefilename 和 checkpoint，中断后重跑只补未完成的片。

用法（af2_binder_design 环境，需要 GPU）:
    export BD_GPU_LABEL="RTX4090-24G"
    python scripts/04_predict_af2.py --campaign config/campaign_pilot.yaml
"""

from __future__ import annotations

import argparse
import sys

from common import OUTPUTS, die, load_tools, load_yaml, record_throughput, run


def group_names(camp: dict) -> list[str]:
    names = [g["name"] for g in camp["diffusion"]["groups"]]
    if camp.get("negative_control", {}).get("enabled"):
        names.append("negctrl")
    return names


def chunk(items: list, size: int) -> list[list]:
    '''将序列文件分片成列表，每个元素包含250个文件'''
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="config/campaign_pilot.yaml")
    parser.add_argument("--only-group")
    parser.add_argument(
        "--recycle",         # 回收机制轮数
        type=int,
        default=3,
        help="AF2 recycle 次数，上游默认 3。调低能省时间但会影响置信度分数，"
        "改了就不能和论文阈值直接对比。",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    camp = load_yaml(args.campaign)
    tools = load_tools(require=not args.dry_run)
    chunk_size = int(camp.get("predict", {}).get("chunk_size", 0))    # 分片提交数量。方便中断续跑，避免一个实例跑几十小时中途掉线全丢

    names = group_names(camp)      # 三组名称：noise00、noise05、negctrl
    if args.only_group:
        names = [n for n in names if n == args.only_group]
        if not names:
            die(f"配置里没有组 {args.only_group}")

    for name in names:
        seq_dir = OUTPUTS / "sequences" / name
        # glob 是 pathlib.Path 的方法，用来按文件名模式列出目录里的文件，返回的是匹配到的路径。
        designs = sorted(seq_dir.glob("*_dldesign_*.pdb")) if seq_dir.exists() else []
        if not designs:
            print(f"[skip] 组 {name} 没有序列设计结果")
            continue

        out_dir = OUTPUTS / "af2" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        # stem 是 pathlib.Path 的一个属性。返回文件名去掉扩展名之后的那一段
        # 例如 outputs/sequences/noise00/noise00_249_dldesign_0.pdb。对这个路径：
        # p.name 是 noise00_249_dldesign_0.pdb，含扩展名的文件名
        # p.suffix 是 .pdb
        # p.stem 是 noise00_249_dldesign_0
        tags = [p.stem for p in designs]
        print(f"[info] 组 {name}: {len(tags)} 个设计待预测")

        for idx, part in enumerate(chunk(tags, chunk_size)):
            runlist = out_dir / f"runlist_{idx:03d}.txt"            # 已完成序列名单
            score_file = out_dir / f"scores_{idx:03d}.sc"           # 存储分数
            checkpoint = out_dir / f"af2_{idx:03d}.check.point"     # 记录当前片已完成序列路径

            if score_file.exists() and checkpoint.exists():
                done = len(               # 已完成序列数量，读取checkpoint计算非空行数
                    [
                        ln
                        for ln in checkpoint.read_text(encoding="utf-8").splitlines()
                        if ln.strip()
                    ]
                )
                if done >= len(part):
                    print(f"[skip] 组 {name} 第 {idx} 片已完成 {done}/{len(part)}")
                    continue
                print(f"[resume] 组 {name} 第 {idx} 片: {done}/{len(part)} 已完成")

            runlist.write_text("\n".join(part) + "\n", encoding="utf-8")

            cmd = [                     # 交给 AF2 initial guess 的一条命令，预测这一片里的序列复合物
                sys.executable,         # 当前环境的python
                tools.af2_predict,      # af2_initial_guess/predict.py  它读设计好的复合物，把设计坐标放进第一轮 recycle，固定 PD-L1，再预测结合剂
                "-pdbdir",
                seq_dir,                # 这一组的序列目录，例如 outputs/sequences/noise00
                "-runlist",
                runlist,                # 这一片要预测的名单
                "-outpdbdir",
                out_dir / f"pred_{idx:03d}",    # 这一片预测结构的输出目录。这里的 PDB 链序和输入相反：链 A 变成结合剂，链 B 变成靶点。
                "-scorefilename",
                score_file,             # 这一片的分数表。
                "-checkpoint_name", 
                checkpoint,
                "-recycle",
                str(args.recycle),      # 来回修正的次数，默认 3。最后一轮的坐标和置信度才写入上面的 PDB 和 .sc
            ]
            elapsed = run(cmd, dry_run=args.dry_run)
            if not args.dry_run:
                record_throughput(
                    f"predict.{name}",
                    len(part),
                    elapsed,
                    note=f"chunk={idx} recycle={args.recycle}",
                )

    print("\nAF2 预测完毕，下一步 05_filter_rank.py")


if __name__ == "__main__":
    main()
