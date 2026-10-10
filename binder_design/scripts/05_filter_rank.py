#!/usr/bin/env python
"""阶段 5：三级置信度过滤、去冗余、出 top-N 与漏斗统计。

三条主过滤直接对应 RFdiffusion 论文 (Watson et al., Nature 2023) 的
in silico success 定义，该定义又来自 Bennett et al., Nat Commun 2023 的
实验校准：plddt_binder > 80、pae_interaction < 10、binder_aligned_rmsd < 1 Å。

过滤之后再做序列去冗余，避免 top-N 里全是同一折叠的近亲。这一层不是论文标准，
纯粹是为了让最终候选有多样性。

不需要 GPU。可以把 outputs/af2 下的 .sc 文件拉回本机跑。

用法:
    python scripts/05_filter_rank.py --campaign config/campaign_pilot.yaml
"""

from __future__ import annotations

import argparse
import json
import re

import pandas as pd

from pathlib import Path

from common import OUTPUTS, die, load_yaml, resolve

# out.sc 里的 tag 形如 noise00_12_dldesign_1_af2pred
# 前半段是 02 生成的骨架名（前缀为组名，见 02_diffuse.py 里的说明）
TAG_RE = re.compile(r"^(?P<backbone>.+?)_dldesign_(?P<seq_idx>\d+)_af2pred$")

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

OPS = {
    ">": lambda s, v: s > v,
    ">=": lambda s, v: s >= v,
    "<": lambda s, v: s < v,
    "<=": lambda s, v: s <= v,
}


# --------------------------------------------------------------------------
# 读分数
# --------------------------------------------------------------------------

def parse_score_file(path) -> pd.DataFrame:
    """解析 af2_initial_guess 写出的 .sc。

    格式是 Rosetta 风格：每行以 SCORE: 开头，其中一行是表头（最后一列 description）。
    """
    header: list[str] | None = None    # header 用来存列名。一开始还没读到表头，所以是 None
    rows: list[list[str]] = []         # rows 用来存数据行。每一行本身又是一个字符串列表，一开始是空列表
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("SCORE:"):
            continue
        tokens = line.split()[1:]
        if not tokens:
            continue
        if tokens[-1] == "description" and header is None:
            header = tokens
            continue
        if header and len(tokens) == len(header):
            rows.append(tokens)
    if header is None:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=header)
    for col in df.columns:
        if col != "description":
            df[col] = pd.to_numeric(df[col], errors="coerce")   # 其余列都转成数字。某个格子转不成数字时，errors="coerce" 把它写成缺失值 NaN，而不是让整个文件报错。
    return df


def load_all_scores(camp: dict) -> pd.DataFrame:
    names = [g["name"] for g in camp["diffusion"]["groups"]]
    if camp.get("negative_control", {}).get("enabled"):
        names.append("negctrl")     # 三个组的name

    frames = []
    for name in names:
        score_files = sorted((OUTPUTS / "af2" / name).glob("scores_*.sc"))   # 每个分片的score结果文件路径列表
        if not score_files:
            print(f"[warn] 组 {name} 没有分数文件")
            continue
        for sf in score_files:
            part = parse_score_file(sf)     # 解析分片score结果文件，返回DataFrame
            if part.empty:
                continue
            part["group"] = name   # 给这一片分数表加上一列，列名是 group，每一行都写成当前组的名字，例如 noise00
            frames.append(part)
    if not frames:
        die("没读到任何分数，先确认 04_predict_af2.py 跑完了")
    df = pd.concat(frames, ignore_index=True)       # 三组拼成一张大表
    # 按 (组, tag) 去重而不是只按 tag：同一片重跑会产生重复行，
    # 但不同组的 tag 本来就应该各自独立。
    df = df.drop_duplicates(subset=["group", "description"], keep="last")

    # str.extract 用这个正则去匹配 description 列的每一行。匹配成功时得到两列：backbone 和 seq_idx
    # 名字会拆成骨架 noise00_170、序列序号 0
    parsed = df["description"].str.extract(TAG_RE)
    df["backbone"] = parsed["backbone"]     # 把拆出来的骨架名写回大表，新列就叫 backbone
    df["seq_idx"] = pd.to_numeric(parsed["seq_idx"], errors="coerce")
    unparsed = df["backbone"].isna().sum()  # 骨架名仍为空的行数，也就是名字不符合这套规则的设计有多少条
    if unparsed:
        print(f"[warn] {unparsed} 条 tag 不符合预期命名，已保留但 backbone 为空")
    return df


# --------------------------------------------------------------------------
# 序列（用于去冗余）
# --------------------------------------------------------------------------

def binder_sequence(pdb_path) -> str:
    """从 MPNN 输出的 PDB 里取第一条链的序列。

    第一条链就是 binder：RFdiffusion 把扩散出的链放在前面（B链），
    AF2 脚本也要求 binder 是第一条链（A链）。
    """
    seq: list[str] = []
    current_chain = None
    seen = set()    # 同一个残基在 PDB 里可能有多行 Cα（例如两套交替坐标），用它避免把同一个残基数两次
    for line in pdb_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        chain = line[21]
        if current_chain is None:
            current_chain = chain
        if chain != current_chain:
            break
        # 用链 ID 和残基编号组成一对。line[22:27] 是残基序号那几列，例如 " 1 "。同一条链、同一个序号视为同一个残基
        key = (chain, line[22:27])
        if key in seen:
            continue
        seen.add(key)
        seq.append(THREE_TO_ONE.get(line[17:20].strip(), "X"))
    return "".join(seq)


def attach_sequences(df: pd.DataFrame) -> pd.DataFrame:
    seqs = []
    # 遍历一张表的每一行。iterrows() 每次给出(行号, 这一行)
    # row 是这一行的各列，所以可以用 row["description"]、row["group"] 按列名取值
    # removesuffix("_af2pred") 的参数是要去掉的结尾；名字确实以这段结尾时才去掉
    for _, row in df.iterrows():    
        tag = str(row["description"]).removesuffix("_af2pred")
        path = OUTPUTS / "sequences" / str(row["group"]) / f"{tag}.pdb"
        # 取 MPNN 生成的结合剂序列（pdb文件中的第一条链，B链）
        seqs.append(binder_sequence(path) if path.exists() else "")
    df = df.copy()
    df["binder_seq"] = seqs
    df["binder_len"] = df["binder_seq"].str.len()
    return df


def identity(a: str, b: str) -> float:
    """按位比对的序列一致度，长度差超过 10% 直接判为不同。

    这是个廉价近似。如果装了 TMalign，按结构 TM-score 聚类更准，
    把 filters.yaml 里 cluster.method 改成 tmscore 并自行接上即可。
    """
    if not a or not b:
        return 0.0
    short, long_ = sorted((len(a), len(b)))
    if short / long_ < 0.9:
        return 0.0
    n = short
    same = sum(1 for i in range(n) if a[i] == b[i])
    return same / n


def greedy_cluster(df: pd.DataFrame, threshold: float, max_per_cluster: int) -> pd.DataFrame:
    """按已排好的顺序贪心聚类，每簇最多保留 max_per_cluster 个。"""
    reps: list[tuple[str, int]] = []  # (代表序列, 该簇已保留数)
    keep_flags = []                   # 记录这一行要不要留下来
    cluster_ids = []                  # 记录这一行属于哪个簇
    for seq in df["binder_seq"]:
        assigned = None
        for cid, (rep_seq, count) in enumerate(reps):
            # 用上面定义的 identity 函数判断这一行和哪个代表最像。
            if identity(seq, rep_seq) >= threshold:    
                assigned = cid
                break
        if assigned is None:            # 所有代表都不够像，就新开一簇
            reps.append((seq, 1))
            cluster_ids.append(len(reps) - 1)
            keep_flags.append(True)
        else:     # 如果当前序列和已保存的对比序列相似度大于阈值，就加入该簇，并记录不保存（升序，靠前的PAE低）
            rep_seq, count = reps[assigned]   
            keep = count < max_per_cluster
            reps[assigned] = (rep_seq, count + (1 if keep else 0))
            cluster_ids.append(assigned)
            keep_flags.append(keep)
    out = df.copy()
    out["cluster"] = cluster_ids
    out["cluster_representative"] = keep_flags
    return out


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def apply_primary(df: pd.DataFrame, primary: dict) -> pd.DataFrame:
    #  复制一张表再改。不改原表，避免误伤。
    df = df.copy() 
    # pd.Series：新建一列，带行标签；True 是要填进这一列的数据；
    # index=df.index 指定这一列的行标签用分数表 df 的行标签：表左侧那一列行号
    mask = pd.Series(True, index=df.index)
    for col, rule in primary.items():
        if col not in df.columns:
            die(f"分数文件里没有列 {col}，实际列: {list(df.columns)}")
        # 把规则里的 op 转成一个函数。例如 "<" 转成 lambda s, v: s < v
        op = OPS.get(rule["op"])        
        if op is None:
            die(f"不支持的比较符 {rule['op']}")
        # 用这条规则比较整列。this 的每个元素是 True 或 False。
        this = op(df[col], float(rule["value"]))
        # df[列名] = 一列值 是 pandas 新增/替换 列的写法
        df[f"pass_{col}"] = this
        # fillna(False)是 Series 的方法：把this中缺失值换成False，返回一列新结果，不改原来this。
        # &= 是按行做逻辑与，再写回 mask。同一行上，mask 和这一条结果都是 True，才仍然是 True
        mask &= this.fillna(False)
    # 三条都走完后，mask 里仍为 True 的才是三条同时通过。这一列名叫 pass_primary。
    df["pass_primary"] = mask
    return df


def resolve_output_dir(raw: str | None) -> Path:
    out = OUTPUTS if raw is None else resolve(raw)
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="config/campaign_pilot.yaml")
    parser.add_argument("--filters", default="config/filters.yaml")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="分数表写出目录。默认 outputs/，跟进实验请另给目录，避免覆盖全量 scores.csv。",
    )
    args = parser.parse_args()
    out_dir = resolve_output_dir(args.output_dir)

    camp = load_yaml(args.campaign)
    flt = load_yaml(args.filters)

    # 合并三个组AF2生成的打分结果文件为一张大表，添加 group、backbone 和 seq_idx三列
    df = load_all_scores(camp)
    print(f"[info] 读到 {len(df)} 条 AF2 预测")

    # 对结果中的三条主过滤，逐条判断。添加 pass_plddt_binder 3列 和 pass_primary 列，是否全通过看 pass_primary 列。
    df = apply_primary(df, flt["primary"])
    # 添加 结合剂序列 binder_seq 和 长度 binder_len 两列
    df = attach_sequences(df)

    scores_csv = out_dir / "scores.csv"
    # to_csv(输出路径, index=False) 是 DataFrame 的方法。
    # index=False 表示不要把行号写进文件。不加这个参数时，CSV 最左边会多一列 0,1,2,...pandas 的行标签
    df.to_csv(scores_csv, index=False)
    print(f"[write] 全部分数 -> {scores_csv}")

    rank = flt["ranking"]
    # df[一列布尔值] 是 pandas 按行筛选的写法。你给它一列和表同样长的 True/False，它就只留下 True 所在的那些行。
    # 界面误差最小的排在最前。sort_values(排序键, ascending=True/False) 是 pandas 排序的写法。True 表示从小到大。
    passing = df[df["pass_primary"]].sort_values(
        rank["sort_by"], ascending=bool(rank["ascending"])
    )
    print(f"[info] 通过三级过滤 {len(passing)} 条")

    # 把已经按界面 pAE 排好序的通过设计分成簇。
    # 序列太像的算同一簇，每簇只留 pAE 最好的那一条，避免 top 20 里全是近亲。
    # 添加了 cluster 和 cluster_representative 两列，分别记录这一行属于哪个簇，以及是否是该簇的代表。
    clustered = greedy_cluster(
        passing,
        float(rank["cluster"]["identity_threshold"]),
        int(rank["cluster"]["max_per_cluster"]),
    )
    clustered.to_csv(out_dir / "passing.csv", index=False)

    # 只取20簇的代表。仍是按PAE升序排列，靠前的是PAE最小的
    reps = clustered[clustered["cluster_representative"]]
    topn = reps.head(int(rank["top_n"]))
    cols = [     # 决定 topn.csv 要哪些列。* 把这几个名字拆开，放进列表，而不是嵌套成另一个列表
        "description", "group", "backbone", "binder_len", "cluster",
        *flt["primary"].keys(), *flt.get("report_only", []), "binder_seq",  
    ]
    # 再滤一遍列名：只保留 topn 里真正存在的列。某一列没读进来时，写文件不会因为缺列中断。
    cols = [c for c in cols if c in topn.columns]
    topn[cols].to_csv(out_dir / "topn.csv", index=False)
    print(f"[write] top-{len(topn)} -> {out_dir / 'topn.csv'}")

    # ------------------ 漏斗 ------------------
    funnel = {"campaign": camp["name"], "by_group": {}}
    # 按 group 列把大表拆开。name 是组名，sub 是该组的所有行，一行对应一次 AF2 预测。
    for name, sub in df.groupby("group"):
        n_backbones = len(list((OUTPUTS / "diffusion" / str(name)).glob("*.pdb")))  # RFdiffusion 写出的骨架数
        sub_pass = sub[sub["pass_primary"]]   # 该组通过三条主过滤的设计
        stage = {
            "backbones_generated": n_backbones,
            "af2_predicted": int(len(sub)),   # 该组AF2预测的设计数
            "passed_primary": int(len(sub_pass)),
            "pass_rate": round(len(sub_pass) / len(sub), 4) if len(sub) else None,
        }

        # 给三条过滤各自算一个通过率
        for col in flt["primary"]:   # for col in 字典 只取出键；值要另外用 .items() 或 .values() 才拿得到。
            flag = f"pass_{col}"
            if flag in sub.columns:
                stage[f"rate_{col}"] = round(float(sub[flag].mean()), 4)   # 单个条件的通过率
        funnel["by_group"][str(name)] = stage   # 把这一组的结果字典写进漏斗字典中的 by_group 子字典里

    funnel["total"] = {
        "af2_predicted": int(len(df)),
        "passed_primary": int(len(passing)),
        "clusters": int(clustered["cluster"].nunique()) if len(clustered) else 0,
        "top_n": int(len(topn)),
    }
    funnel_path = out_dir / "funnel.json"
    funnel_path.write_text(
        json.dumps(funnel, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[write] 漏斗 -> {funnel_path}")
    print(json.dumps(funnel, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
