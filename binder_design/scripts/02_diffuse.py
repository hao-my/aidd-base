#!/usr/bin/env python
"""阶段 2：RFdiffusion 生成 binder 骨架。

按 campaign 配置逐组调用 run_inference.py。组的划分就是消融轴：
noise00 / noise05 是同一 hotspot 下的两档推理噪声，negative_control
是换到背面 hotspot 的对照组。

每组耗时写入 outputs/throughput.json，这是试点阶段的核心产物。

需要 GPU。用法（在租的机器上，SE3nv 环境里）:
    export RFDIFFUSION_DIR=~/program/RFdiffusion
    export DL_BINDER_DIR=~/program/dl_binder_design
    export BD_GPU_LABEL="RTX4090-24G"
    python scripts/02_diffuse.py --campaign config/campaign_pilot.yaml
"""

from __future__ import annotations

import argparse
import sys

from common import (
    OUTPUTS,
    die,
    load_tools,
    load_yaml,
    record_throughput,
    resolve,
    run,
)


def build_contigs(tgt: dict, binder: dict) -> str:
    """拼 contigmap。

    格式 [A18-132/0 80-120]:
      A18-132  固定的靶点链段，扩散过程中坐标不变
      /0       链断点，连接段长度为0，两侧是物理上分开的两条链（PPI 而非融合蛋白）
      80-120   从头生成的 binder，长度在区间内随机取
    """
    return (
        f"[{tgt['chain']}{tgt['res_start']}-{tgt['res_end']}/0 "
        f"{binder['length_min']}-{binder['length_max']}]"
    )


def build_hotspot_arg(chain: str, hotspots: list) -> str:
    """hotspot 可以写成 56 或 B56，统一补上链 ID。"""
    tags = []
    for h in hotspots:
        text = str(h).strip()
        # isalpha() 判断字符串是否全为字母，且不能为空
        tags.append(text if text[:1].isalpha() else f"{chain}{text}")
    return "[" + ",".join(tags) + "]"


def count_existing(out_dir) -> int:
    """已经生成的 pdb 数，用于断点续跑。"""
    if not out_dir.exists():
        return 0
    return len(list(out_dir.glob("*.pdb")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="config/campaign_pilot.yaml")
    parser.add_argument("--target", default="config/target_pdl1.yaml")
    parser.add_argument("--only-group", help="只跑指定组名，便于分批提交")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    args = parser.parse_args()

    # 加载配置文件  yaml格式文件读取后同josn一样变成字典和列表，都用键一层层往下取。
    # 但 yaml 可用 # 写注释，字符串常常不用引号，层级靠缩进表示
    camp = load_yaml(args.campaign)
    tcfg = load_yaml(args.target)
    # 寻找环境变量中的 RFdiffusion 和 dl_binder_design 中的三个脚本：run_inference.py, dl_interface_design.py, predict.py
    tools = load_tools(require=not args.dry_run)   

    # 获取靶点链/起始位置/clean结构信息、粘合剂binder的长度采样区间、hotspot候选
    tgt, binder, hotspots = tcfg["target"], tcfg["binder"], tcfg["hotspots"]
    chain = tgt["chain"]

    # 验证 clean_pdb 和 hotspots 是否存在
    input_pdb = resolve(tgt["clean_pdb"])
    if not input_pdb.exists() and not args.dry_run:
        die(f"靶点结构不存在: {input_pdb}，先跑 01_prepare_target.py")

    if not hotspots.get("design"):
        die(
            "config 里 hotspots.design 还是空的。先跑 01_prepare_target.py，"
            "再从 data/target/hotspots.json 挑 3-6 个填进去。"
        )

    # 拼 contigmap  [A18-132/0 80-120]
    contigs = build_contigs(tgt, binder)
    print(f"[info] contigs = {contigs}")

    # 组装任务列表：消融组 + 可选的阴性对照组
    groups = list(camp["diffusion"]["groups"])       # 一个有 noise00 和 noise0 两项的 字典列表
    neg = camp.get("negative_control", {})
    if neg.get("enabled"):
        if not hotspots.get("negative_control"):
            die("负对照已开启但 hotspots.negative_control 为空")
        groups.append(                                    # 添加阴性对照组
            {
                "name": "negctrl",
                "hotspot_set": "negative_control",
                "num_designs": neg["num_designs"],
                "noise_scale_ca": neg.get("noise_scale_ca", 0.0),
                "noise_scale_frame": neg.get("noise_scale_frame", 0.0),
            }
        )
    
    # 只跑指定组名，便于分批提交
    if args.only_group:
        groups = [g for g in groups if g["name"] == args.only_group]
        if not groups:
            die(f"配置里没有组 {args.only_group}")

    for group in groups:
        name = group["name"]
        out_dir = OUTPUTS / "diffusion" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        # 前缀用组名而不是统一的 "design"。否则三组都产出 design_0.pdb，
        # 到了 AF2 阶段 tag 会跨组重名，合并分数时会互相覆盖，
        # 消融对比直接失效。
        prefix = name
        want = int(group["num_designs"])     # 每个组要生成的骨架数量
        have = count_existing(out_dir)       # 已经生成的骨架数量
        if have >= want:
            print(f"[skip] 组 {name} 已有 {have}/{want} 个骨架")
            continue
        if have:
            print(f"[resume] 组 {name} 已有 {have} 个，继续生成 {want - have} 个")

        # 拼 hotspot 参数 [A56, A115, A121, A123] 或 [A31, A32, A33, A105]
        hs = build_hotspot_arg(chain, hotspots[group["hotspot_set"]])
        cmd = [
            sys.executable,    #当前这个 Python 解释器，在云主机上它应该是已经激活的 SE3nv 环境里的 python
            tools.run_inference,    #是 RFDIFFUSION_DIR/scripts/run_inference.py，真正做扩散的程序
            f"inference.output_prefix={out_dir / prefix}",
            f"inference.input_pdb={input_pdb}",       # 靶点结构：指定区间内，去除水和杂原子后的标准氨基酸残基链
            f"contigmap.contigs={contigs}",           # contigmap结构： [A18-132/0 80-120]   连接段长度为0，在热点表面折叠生成长度在80-120之间的粘合剂binder
            f"ppi.hotspot_res={hs}",                  # hotspot候选： [A56, A115, A121, A123] 或 [A31, A32, A33, A105]
            f"inference.num_designs={want - have}",                     # RFdiffusion 生成骨架时，每个残基用一个刚体框架表示，里面有两样东西：
            f"denoiser.noise_scale_ca={group['noise_scale_ca']}",       # Cα 的位置，残基在空间里的落点。noise_scale_ca 缩放加在这个位置上的扰动。
            f"denoiser.noise_scale_frame={group['noise_scale_frame']}", # 主链的朝向，N–Cα–C 这三原子构成的局部坐标系。noise_scale_frame 缩放加在这个朝向上的扰动。
            *camp["diffusion"].get("extra_args", []),
        ]
        elapsed = run(cmd, dry_run=args.dry_run)      # 运行扩散程序，返回运行时间。上述列表中的参数会覆盖run_inference.py配置文件中的默认值
        if not args.dry_run:
            record_throughput(   # 记录吞吐量，写入 outputs/diffusion/throughput.json
                f"diffusion.{name}",
                want - have,
                elapsed,
                note=f"hotspots={hs} noise_ca={group['noise_scale_ca']}",
            )

    print("\n骨架生成完毕，下一步 03_design_seqs.py")


if __name__ == "__main__":
    main()
