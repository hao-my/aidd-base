#!/usr/bin/env python
"""阶段 6：对通过 AF2 主过滤的复合物计算结合能和结合剂 SAP。

每条结构：ref2015 上 1 轮 FastRelax，InterfaceAnalyzer 取 dG_separated，
再删掉靶点链 B，对结合剂计算 SapScoreMetric。松弛坐标不保存。
pass_ddg、pass_sap 只作标记，不删行。不算 dSASA。

默认读 outputs/passing.csv，PDB 在 outputs/af2/<组名>/pred_*/ 下查找。
换 --scores 时，总表和检查点写到该分数表所在目录，避免和全量结果混在一起。
检查点每完成一条就追加。全部成功后才写出总表。

用法（已安装 PyRosetta 的环境）:
    python scripts/07_interface_metrics.py
    python scripts/07_interface_metrics.py --scores outputs/refine150_rank/passing.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUTPUTS, resolve

WORKERS = 6
TARGET_LEN = 115
DG_CUTOFF = -30.0
SAP_CUTOFF = 45.0

FIELDS = [
    "description",
    "group",
    "backbone",
    "pae_interaction",
    "plddt_binder",
    "binder_aligned_rmsd",
    "dG_separated",
    "sap",
    "pass_ddg",
    "pass_sap",
    "pdb",
]
CKPT_FIELDS = ["description", "dG_separated", "sap"]
NEEDED = [
    "description",
    "group",
    "backbone",
    "pae_interaction",
    "plddt_binder",
    "binder_aligned_rmsd",
]


def default_output(scores: Path) -> Path:
    """全量通过表的结果放在 outputs/ 下，其余跟分数表放在同一目录。"""
    if scores.resolve() == (OUTPUTS / "passing.csv").resolve():
        return OUTPUTS / "rosetta_interface.csv"
    return scores.resolve().parent / "rosetta_interface.csv"


def find_pdb(group: str, description: str) -> Path:
    hits = sorted((OUTPUTS / "af2" / group).glob(f"pred_*/{description}.pdb"))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"{group}/{description} 匹配到 {len(hits)} 个 PDB，期望 1 个"
        )
    return hits[0]


def score_pose(pdb: str) -> tuple[float, float]:
    """返回 (dG_separated, sap)。dG 在删链之前算，SAP 只看结合剂。"""
    # os.environ 是本进程能看到的环境变量。
    # setdefault("OMP_NUM_THREADS", "1") 的意思是：如果 OMP_NUM_THREADS 还不存在，就把它写成 "1"；已经有人在外面设过，就保持原值。
    # OMP_NUM_THREADS 告诉使用 OpenMP 的程序用几条线程。PyRosetta 的 FastRelax 和界面打包会读它。
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import pyrosetta
    from pyrosetta.rosetta.core.pack.guidance_scoreterms.sap import SapScoreMetric
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.protocols.relax import FastRelax


    # 启动这个进程里的 Rosetta，并关掉它的全部日志。
    # init 每个进程都要先调用一次。它读取 Rosetta 自带的氨基酸、旋转异构体和能量函数数据。后面的 pose_from_pdb、FastRelax 和 InterfaceAnalyzer 都依赖这次初始化。
    # -mute all 是 Rosetta 的命令行选项，意思是把所有日志通道都关掉。Rosetta 的输出按通道分开，例如松弛、打分、打包各写各的。-mute 后面跟通道名，就不再打印那个通道。
    pyrosetta.init("-mute all")
    # 把一个 PDB 文件读进内存，变成 Rosetta 能直接操作的结构对象 pose
    pose = pyrosetta.pose_from_pdb(pdb)
    # 当前任务要求复合物有两条链，A和B。pose本身可以有任意条链。
    if pose.num_chains() != 2:  
        raise RuntimeError(f"{pdb} 链数是 {pose.num_chains()}，期望 2")
    # PDB 里原来的链字母和残基号另外存在 pdb_info 里
    info = pose.pdb_info()
    # 整条 Pose 内部从 1 连续编号。链 A 的结合剂在前，链 B 的靶点在后。
    # chain_begin(2) 是 2 号链（B）的第一个残基的编号。链 A 的结合剂在 1 号链，从 1 开始编号。
    chain_b_begin = pose.chain_begin(2)
    # 检查 PDB 里 A 和 B 链的标识是否正确
    if info.chain(1) != "A" or info.chain(chain_b_begin) != "B":
        raise RuntimeError(
            f"{pdb} 链标识是 {info.chain(1)}/{info.chain(chain_b_begin)}，期望 A/B"
        )
    target_len = pose.chain_end(2) - chain_b_begin + 1
    if target_len != TARGET_LEN:
        raise RuntimeError(f"{pdb} 靶点链长度是 {target_len}，期望 {TARGET_LEN}")

    # 创建一个打分函数对象，并选用 Rosetta 的 ref2015 权重。
    sfxn = pyrosetta.create_score_function("ref2015")
    # 用 ref2015 对这个复合物做 1 轮松弛，并直接改掉内存里 pose 的坐标。
    FastRelax(sfxn, 1).apply(pose)

    bound = pose.clone()
    #创建一个界面分析器，此时还没有结构和能量。 InterfaceAnalyzerMover 是 Rosetta 里专门算两条链界面能量的一个操作器。
    analyzer = InterfaceAnalyzerMover()
    # "A_B" 指定界面两侧：链 A 是结合剂，链 B 是靶点。能量只比较这两侧之间的变化。
    analyzer.set_interface("A_B")
    # set_scorefunction 是把一个打分函数交给界面分析器。sfxn 就是被交过去的那个对象。
    analyzer.set_scorefunction(sfxn)
    # set_pack_input 决定打结合态能量之前，要不要先重排侧链。False 表示不要。直接用松弛留下的坐标计算 E_结合
    # pack 指侧链打包：氨基酸种类不变，只从旋转异构体库里为每个位点再选一个侧链构象。
    analyzer.set_pack_input(False)
    # set_pack_separated 决定两条链被拉开之后，要不要重排刚暴露出来的界面侧链。True 表示要。
    analyzer.set_pack_separated(True)
    # set_calc_dSASA 决定要不要计算界面埋藏表面积。False 表示不算，结果里不会有 dSASA。
    analyzer.set_calc_dSASA(False)
    # set_compute_packstat 决定要不要计算界面原子堆得紧不紧。False 表示不算。
    # packstat 来自 RosettaHoles。它在界面附近放许多小探针，看原子之间留下的空隙有多大。空隙越少，分数越接近 1
    analyzer.set_compute_packstat(False)
    # 让已经配置好的界面分析器在副本 bound 上真正算一遍：先给结合态打分，再拉开链、重排界面侧链、给分开状态打分。原来的 pose 不被改动。
    # 算完的差值存在分析器内部，下一行 get_interface_dG() 才把它取出来。
    analyzer.apply(bound)
    dg = float(analyzer.get_interface_dG())

    # delete_residue_range_slow 按残基编号删掉一段
    pose.delete_residue_range_slow(pose.chain_begin(2), pose.chain_end(2))
    if pose.num_chains() != 1:
        raise RuntimeError(f"{pdb} 删除靶点后还剩 {pose.num_chains()} 条链")
    # 在剩下的结合剂上计算 SAP：表面疏水斑块越大，分数越高。
    sap = float(SapScoreMetric().calculate(pose))
    return dg, sap


def score_one(task: tuple[str, str]) -> dict:
    description, pdb = task
    try:
        dg, sap = score_pose(pdb)
    except Exception as exc:
        return {"description": description, "error": f"{type(exc).__name__}: {exc}"}
    return {"description": description, "dG_separated": dg, "sap": sap}


def load_rows(scores: Path) -> list[dict]:
    with scores.open(encoding="utf-8") as fh:
        # DictReader是 Python 标准库里的 CSV 阅读器。
        # 它把第一行当作列名，后面每一行读成一个字典，键是列名，值是该格的文字。
        # reader 不是列表，也不是字典。它是一个 csv.DictReader 对象，作用是按行往外吐数据：每次给出下一行的字典
        reader = csv.DictReader(fh)
        # fieldnames 是 DictReader 读到的表头，一个列名列表。寻找scores文件中不存在的需求列，然后报错
        missing = [name for name in NEEDED if reader.fieldnames is None or name not in reader.fieldnames]
        if missing:
            raise SystemExit(f"{scores} 缺少列: {', '.join(missing)}")
        # 把这个对象从头到尾读完，收成真正的字典列表。列表里每个元素都是一行，一行的字典里每个键值对都是一列。
        rows = list(reader)
    if not rows:
        raise SystemExit(f"{scores} 没有数据行")
    return rows


def load_done(checkpoint: Path) -> dict[str, dict[str, str]]:
    if not checkpoint.exists():
        return {}
    with checkpoint.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        if fields != CKPT_FIELDS:
            raise SystemExit(
                f"{checkpoint} 表头是 {fields}，期望 {CKPT_FIELDS}。"
                "这不是本脚本的检查点，请换一个 --output。"
            )
        return {row["description"]: row for row in reader}


def finalize(rows: list[dict], done: dict[str, dict[str, str]], out: Path) -> None:
    missing = [row["description"] for row in rows if row["description"] not in done]
    if missing:
        raise SystemExit(f"还有 {len(missing)} 条没有 SAP 和 dG_separated，不写总表")
    tmp = out.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            scored = done[row["description"]]
            dg = float(scored["dG_separated"])
            sap = float(scored["sap"])
            writer.writerow(
                {
                    "description": row["description"],
                    "group": row["group"],
                    "backbone": row["backbone"],
                    "pae_interaction": row["pae_interaction"],
                    "plddt_binder": row["plddt_binder"],
                    "binder_aligned_rmsd": row["binder_aligned_rmsd"],
                    "dG_separated": f"{dg:.4f}",
                    "sap": f"{sap:.4f}",
                    "pass_ddg": dg < DG_CUTOFF,
                    "pass_sap": sap < SAP_CUTOFF,
                    "pdb": row["pdb"],
                }
            )
    os.replace(tmp, out)
    print(f"[write] {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default=str(OUTPUTS / "passing.csv"))
    parser.add_argument(
        "--output",
        default=None,
        help="总表路径。默认：全量通过表写到 outputs/rosetta_interface.csv，"
        "其他 --scores 写到该分数表同目录。",
    )
    args = parser.parse_args()

    # 将rosetta结果和检查点文件的路径 与通过主筛选的scores文件 设置到同一目录下
    scores = resolve(args.scores)
    if not scores.is_file():
        raise SystemExit(f"找不到分数表 {scores}")
    out = resolve(args.output) if args.output else default_output(scores)
    checkpoint = out.with_suffix(".checkpoint.csv")

    # 加载scores文件为一个字典列表。每行是一个字典，键是列名，值是该格的文字。
    rows = load_rows(scores)
    for row in rows:
        try:    # 在每行加入一列pdb文件的路径
            row["pdb"] = str(find_pdb(row["group"], row["description"]))
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc

    # 加载检查点文件，返回一个字典。文件不存在则返回空字典
    # 键是每个样本ID，值是一个字典-三个键值对对应checkpoint文件的三列
    done = load_done(checkpoint)
    # 同样方式，把scores文件所需列也转成一个字典，内层字典是NEEDED六列键值对
    by_desc = {row["description"]: row for row in rows}
    # 将还未完成样本的description和pdb文件路径 组成一个列表，供后续计算使用
    todo = [(row["description"], row["pdb"]) for row in rows if row["description"] not in done]
    print(f"[info] 已完成 {len(done)}，待评分 {len(todo)}", flush=True)
    if not todo:
        # 在全部设计都算完之后，把检查点里的分数和输入表里的原有列拼成总表 rosetta_interface.csv
        finalize(rows, done, out)
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    # 分析还未开始时，检查点文件不存在，则新建一个
    new_file = not checkpoint.exists()
    failed = 0
    # checkpoint.open("a") 用追加模式打开。文件还不存在时，这个模式会新建一个空文件。
    with checkpoint.open("a", encoding="utf-8", newline="") as fh:
        # csv.DictWriter 是 DictReader 的反向操作。把一个字典写成一行 CSV，fieldnames 规定列名和顺序
        writer = csv.DictWriter(fh, fieldnames=CKPT_FIELDS)
        if new_file:
            # writeheader() 把这三个列名写成第一行，作为表头
            writer.writeheader()
        with Pool(WORKERS) as pool:   # 多进程计算
            # pool.imap_unordered 是进程池的一种派发方式。把 todo 列表中的每一项交给 score_one 去算，算完一条就交回一条，
            for i, scored in enumerate(pool.imap_unordered(score_one, todo), start=1):
                description = scored["description"]
                if description not in by_desc:
                    raise SystemExit(f"未知 description {description}")
                if "error" in scored:
                    failed += 1
                    print(f"[error] {description}: {scored['error']}", flush=True)
                else:
                    record = {
                        "description": description,
                        "dG_separated": f"{scored['dG_separated']:.4f}",
                        "sap": f"{scored['sap']:.4f}",
                    }
                    writer.writerow(record)
                    fh.flush()
                    done[description] = record
                if i % 25 == 0 or i == len(todo):
                    print(f"[progress] {i}/{len(todo)}", flush=True)
    if failed:
        raise SystemExit(f"{failed} 条失败，未写入总表。修好后重跑会只补这些条。")
    finalize(rows, done, out)


if __name__ == "__main__":
    main()
