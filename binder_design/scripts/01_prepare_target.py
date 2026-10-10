#!/usr/bin/env python
"""阶段 1：靶点准备与 hotspot 候选推导。

做三件事：
1. 下载 PD-1/PD-L1 复合物，裁出干净的 PD-L1 靶点链，存成 RFdiffusion 的输入。
2. 用 PD-1 链实测界面残基（Cβ 距离阈值内），作为 design hotspot 候选。
3. 用 SASA 找暴露且离界面最远的表面残基，作为阴性对照 hotspot 候选。

hotspot 不由本脚本自动定稿。它输出候选清单，你从中挑 3-6 个填回
config/target_pdl1.yaml 的 hotspots.design。这样做的理由：官方建议 3-6 个，
且模型训练时只见过真实 hotspot 的 0-20%，推理时会自己补更多接触，
给太多会过度约束界面。

本脚本不需要 GPU，可以在本机跑。

用法:
    python scripts/01_prepare_target.py --target config/target_pdl1.yaml
"""

from __future__ import annotations

import argparse
import json
import urllib.request

from Bio.PDB import PDBIO, PDBParser, Select   #Select 是用于 PDB 文件输出筛选的基类
from Bio.PDB.SASA import ShrakeRupley

from common import REPO_ROOT, die, load_yaml, resolve

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


# pdb_id: str 表示编号是字符串，-> None 表示函数不返回值，结果直接写进 dest 这个文件。
def download_pdb(pdb_id: str, dest) -> None:
    if dest.exists():
        print(f"[skip] 已存在 {dest}")
        return
    url = RCSB_URL.format(pdb_id=pdb_id.upper())     # pdb_id.upper() 将pdb_id转换为大写
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get] {url}")
    urllib.request.urlretrieve(url, dest)    # 用标准库把该地址的内容下载并写入 dest


def pseudo_cb(residue):
    """取 Cβ 坐标；甘氨酸没有 Cβ，退回 Cα。返回 None 表示该残基不可用。"""
    # 在 Biopython 里，一个残基是「原子名字 → 原子」的字典。主链和侧链上的每个原子都有固定名字，所以可以用 "CB"、"CA" 直接查有没有这个原子。
    # 原子名   是什么       哪些残基有
    # N       主链氮         都有
    # CA      主链 Cα        都有
    # C       主链羰基碳     都有
    # O       羰基氧         都有
    # CB      侧链 Cβ       除甘氨酸外都有
    for name in ("CB", "CA"):
        if name in residue:
            return residue[name].coord
    return None


class ChainRangeSelect(Select):    # 它继承 Biopython 的 Select。父类默认全部接受。这个子类只改了三个判断。
    """只保留指定链里指定残基区间的标准氨基酸，丢掉水和杂原子。"""

    def __init__(self, chain_id: str, res_start: int, res_end: int):
        self.chain_id = chain_id
        self.res_start = res_start
        self.res_end = res_end

    def accept_chain(self, chain):
        # 比较链 ID。链 A 留下，链 B（PD-1）整条不写。
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        hetflag, resseq, _icode = residue.id
        # 在已经通过的链上再筛残基。hetflag 去掉空格后还有字符，说明是水或配体
        if hetflag.strip():
            return False
        # 标准氨基酸再看序号是否在 18 到 132 之间
        return self.res_start <= resseq <= self.res_end

    def accept_atom(self, atom):
        # 在留下的残基里再筛原子。两个条件都要成立：
        # atom.element 是这个原子的化学元素：去掉氢。
        # atom.get_altloc() 是空格或 "A"。有的晶体里同一个原子有两套坐标，PDB 用占位符 A、B 区分。空格表示只有一套坐标；"A" 表示取第一套。
        return atom.element != "H" and atom.get_altloc() in (" ", "A")


def dist(a, b) -> float:
    return float(sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5)


def pick_compact_set(interface, coord_by_resi, *, n: int, max_spread: float):
    """从界面残基里贪心挑 n 个空间上互相靠近的。

    以离 partner 最近的残基为种子，依次加入「与已选集合平均距离最小」
    且不超过 max_spread 的残基。这样得到的 hotspot 聚成一片，
    而不是散落在整个界面上。
    """
    if not interface:
        return []
    chosen = [interface[0]]   #interface已排序，索引 0 是离受体链 PD-1 最近的 残基字典
    while len(chosen) < n:
        best = None
        best_d = None
        chosen_ids = {c["resi"] for c in chosen}    # resi 代表残基编号
        for cand in interface:                      # 遍历一圈，选一个候选残基
            if cand["resi"] in chosen_ids:
                continue
            # coord_by_resi: {残基序号: Cβ坐标}，计算所有候选残基与已选最近残基之间的Cβ距离
            ds = [dist(coord_by_resi[cand["resi"]], coord_by_resi[c["resi"]]) for c in chosen]
            # 要求与已选残基的最远距离得小于 max_spread阈值 14 A
            if max(ds) > max_spread:
                continue
            mean_d = sum(ds) / len(ds)     # 计算平均距离
            if best_d is None or mean_d < best_d:
                best, best_d = cand, mean_d     # 更新 残基字典best 和 平均距离best_d
        if best is None:
            break
        chosen.append(best)
    return chosen


def protein_residues(chain):
    """链上的标准残基（丢掉水、配体和没有 Cα/Cβ 的残基）。"""
    out = []
    # 链上当前这一个残基，里面有原子坐标和残基编号
    # Biopython 读入后，一个残基有两套名字：
    # 三字母名称在 residue.resname 里，例如 "TYR"、"ALA"。这是化学上的氨基酸种类。
    # 编号在 residue.id 里，是一个三元组 (hetflag, resseq, icode)：
    for residue in chain:
        # residue.id 拆成三项。Biopython 里这个编号固定是三元组：hetflag, resseq, icode
        # 三元组里前两项是字符串，icode 是单字符。icode 通常是空格，表示该残基没有插入码
        # hetflag 残基类型标记，标准氨基酸上是一个空格' '。水的 hetflag 是 'W'。配体、离子这类非标准残基的 hetflag 以 'H_' 开头。
        # resseq 残基序号，如 56，标准氨基酸上用不倒，所以前面加_
        # icode 插入码，少数结构用来区分同一序号的两个残基，通常是空格' '，表示该残基没有插入码,这里也用不到
        hetflag, _resseq, _icode = residue.id
        # 标准氨基酸去掉后变成空字符串，条件为假，会留下来。水或配体去掉空格后仍有字符，条件为真，continue 跳过这个残基。
        if hetflag.strip():
            continue
        # pseudo_cb(residue) 取 Cβ 坐标；甘氨酸没有 Cβ，退回 Cα。返回 None 表示该残基不可用。
        if pseudo_cb(residue) is None:
            continue
        out.append(residue)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="config/target_pdl1.yaml")
    parser.add_argument(
        "--n-hotspots",
        type=int,
        default=4,
        help="推荐 hotspot 个数。官方建议 3-6 个：模型训练时只见过真实 hotspot "
        "的 0-20%%，推理时会自己补更多接触，给太多会过度约束界面。",
    )
    parser.add_argument(
        "--hotspot-spread",
        type=float,
        default=14.0,
        help="推荐集合内任意两个 hotspot 的 Cβ 距离上限（Å），控制聚集程度",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.target)
    src, tgt, prt = cfg["source"], cfg["target"], cfg["partner"]

    local_pdb = resolve(src["local_pdb"])     # 本地储存pdb文件的绝对路径
    download_pdb(src["pdb_id"], local_pdb)    # 下载pdb文件到本地

    # PDBParser 是 Biopython 的 PDB 文件解析器。QUIET 是它的构造参数：是否抑制解析过程中的警告信息。
    # 缺原子、残基编号不连续、非标准残基，Biopython 默认都会打印一行警告。4ZQK 裁链前这类提示很多，关掉后终端只留下脚本自己的 [info]。
    # 这里没有改另外两个构造参数，所以仍用默认值：PERMISSIVE=True（格式不完全标准也尽量读完），get_header=False（不单独解析文件头里的实验信息）。
    # get_structure(id, file) 负责真正读文件，返回一个 Structure。 
    # src["pdb_id"] 是给这个结构起的名字，str(local_pdb) 是本地文件路径
    # Structure 是 Biopython 的 PDB 文件解析器返回的结构体，包含一个或多个 Model，每个 Model 包含一个或多个 Chain，每个 Chain 包含一个或多个 Residue，每个 Residue 包含一个或多个 Atom。
    structure = PDBParser(QUIET=True).get_structure(src["pdb_id"], str(local_pdb))
    # structure[0]，取第 0 号模型。一次 X 射线实验通常只给出一个坐标模型，4ZQK 就是这样，所以只有 structure[0]
    # model内按chain_id索引，chain_id是字符串，model[chain_id]返回一个Chain对象
    model = structure[0]

    for label, chain_id in (("靶点", tgt["chain"]), ("partner", prt["chain"])):
        if chain_id not in model:
            die(f"{label}链 {chain_id} 不在 {src['pdb_id']} 里，实际链: {[c.id for c in model]}")

    target_chain = model[tgt["chain"]]    #链A：PD-L1    许多细胞表面，肿瘤细胞上尤其多，是配体。贴到PD-1上后，T细胞收到「不要攻击」的信号，免疫反应被压下去
    partner_chain = model[prt["chain"]]   #链B：PD-1     T细胞表面，是受体

    target_res = protein_residues(target_chain)   # 返回一个列表，列表里每个元素是一个 标准氨基酸残基 Residue
    observed = [r.id[1] for r in target_res]      # 得到每个标准氨基酸残基的序号。residue.id = hetflag, resseq, icode 
    print(
        f"[info] 靶点链 {tgt['chain']} 实际残基范围 {min(observed)}-{max(observed)}"
        f"（共 {len(observed)} 个，可能有缺失编号）"
    )

    # 校验配置里的区间确实落在观测范围内，避免静默裁错
    if tgt["res_start"] < min(observed) or tgt["res_end"] > max(observed):
        die(
            f"配置区间 {tgt['res_start']}-{tgt['res_end']} 超出实测范围 "
            f"{min(observed)}-{max(observed)}，请修正 config 后重跑"
        )

    # ---------------- 1. 写出干净靶点，只包含靶点链A：PD-L1 ----------------
    clean_pdb = resolve(tgt["clean_pdb"])      # resolve把配置里的相对路径解析成绝对路径
    clean_pdb.parent.mkdir(parents=True, exist_ok=True)
    # PDBIO() 没有参数。它是 Biopython 的写出工具，此时还不知道要写哪个结构。
    io = PDBIO()
    # set_structure是PDBIO的实例方法，用于设置要写入的结构
    io.set_structure(structure)
    io.save(
        str(clean_pdb),
        # ChainRangeSelect筛选器：只保留指定链里指定残基区间的标准氨基酸，丢掉水和杂原子
        ChainRangeSelect(tgt["chain"], tgt["res_start"], tgt["res_end"]),
    )
    print(f"[write] 干净靶点 -> {clean_pdb}")

    # ---------------- 2. 界面 hotspot 候选 ----------------
    cutoff = float(prt["contact_cutoff"])
    # 计算 受体链 PD-1 上每个标准氨基酸残基的 Cβ 坐标
    partner_cb = [pseudo_cb(r) for r in protein_residues(partner_chain)]
    # 筛选 靶点链（配体） PD-L1 上在配置区间内的标准氨基酸残基
    in_range = [
        r for r in target_res if tgt["res_start"] <= r.id[1] <= tgt["res_end"]
    ]

    # 界面残基列表：保留靶点链上 与 受体链 PD-1 空间距离小于 10A 的氨基酸残基
    interface = []
    for residue in in_range:
        cb = pseudo_cb(residue)
        # 计算靶点链 PD-L1 上每个标准氨基酸残基与 受体链 PD-1 上每个标准氨基酸残基的 Cβ 距离
        dists = [float(((cb - p) ** 2).sum() ** 0.5) for p in partner_cb]
        dmin = min(dists)
        # 如果距离小于配置里的 Cβ 距离阈值，说明这个靶点残基与 PD-1 残基有接触
        if dmin <= cutoff:
            interface.append(
                {
                    "resi": residue.id[1],                        # 靶点残基序号
                    "resn": residue.get_resname(),                # 三字母靶点残基名称
                    "tag": f"{tgt['chain']}{residue.id[1]}",      # 靶点残基链ID+序号
                    "min_cb_dist": round(dmin, 2),                # 最小Cβ距离
                }
            )
    interface.sort(key=lambda d: d["min_cb_dist"])     # 按最小Cβ距离排序
    print(
        f"[info] Cβ ≤ {cutoff} Å 的界面残基 {len(interface)} 个，"
        f"最近 8 个: {[d['tag'] for d in interface[:8]]}"
    )

    # 从界面残基里挑一组空间上聚在一起的，作为推荐 hotspot。散落在界面各处的 hotspot 会让 RFdiffusion 难以同时满足，聚成一片的更接近训练时见到的样子。
    # 
    # 对靶点链 PD-L1 上每个标准氨基酸残基，计算其 Cβ 坐标，得到一个字典：{残基序号: Cβ坐标}
    coord_by_resi = {r.id[1]: pseudo_cb(r) for r in in_range}
    # 从界面残基里挑一组空间上聚在一起的，作为推荐 hotspot。
    recommended = pick_compact_set(
        interface, coord_by_resi, n=args.n_hotspots, max_spread=args.hotspot_spread
    )
    print(f"[info] 推荐 hotspot（空间聚集、离界面最近）: {[d['tag'] for d in recommended]}")

    # ---------------- 3. 阴性对照候选 ----------------
    # 先算 SASA 只保留暴露残基，再取离界面质心最远的几个。
    # 埋在内部的残基当 hotspot 没有意义，binder 到不了。
    # ShrakeRupley类，给 PD-L1 链上每个残基算溶剂可及表面积，结果写在残基自己的 .sasa 上，单位是 Å²。
    # compute 是这个类的计算方法。第一个参数 target_chain 是链 A，也就是 PD-L1。第二个参数 level="R" 表示把面积汇总到残基（Residue）
    ShrakeRupley().compute(target_chain, level="R")
    if interface:                                                               
        ifc_ids = {d["resi"] for d in interface}                                # 残基序号
        ifc_coords = [pseudo_cb(r) for r in in_range if r.id[1] in ifc_ids]     # 残基CB坐标
        cx = sum(c[0] for c in ifc_coords) / len(ifc_coords)
        cy = sum(c[1] for c in ifc_coords) / len(ifc_coords)                    #(cx, cy, cz) 是 PD-1 结合面的质心，不是整条 PD-L1 链的中心。
        cz = sum(c[2] for c in ifc_coords) / len(ifc_coords)
    else:
        die("没找到界面残基，检查 partner 链和 contact_cutoff")

    # 排除链两端各 5 个残基：末端柔性大，当 hotspot 没有意义，也会让阴性对照变成「无效 hotspot」而不是「错误位点」。
    lo, hi = min(observed), max(observed)     # 靶点链A的残基序号范围
    term_margin = 5

    exposed = []
    for residue in in_range:
        resi = residue.id[1]
        if resi - lo < term_margin or hi - resi < term_margin:
            continue
        # getattr 是 Python 内置函数，按名字读对象上的属性。三个参数分别是：对象 residue、属性名 "sasa"、找不到时的默认值 0.0。
        sasa = getattr(residue, "sasa", 0.0)
        if sasa < 40.0:  # Å²，粗筛掉埋藏残基
            continue
        cb = pseudo_cb(residue)
        d = ((cb[0] - cx) ** 2 + (cb[1] - cy) ** 2 + (cb[2] - cz) ** 2) ** 0.5
        exposed.append(
            {
                "resi": residue.id[1],
                "resn": residue.get_resname(),
                "tag": f"{tgt['chain']}{residue.id[1]}",
                "sasa": round(float(sasa), 1),
                "dist_to_interface_centroid": round(float(d), 2),
            }
        )
    exposed.sort(key=lambda d: -d["dist_to_interface_centroid"])    # 从大到小排序，将距离PD-1结合面最远的残基排在前面
    # 阴性对照也要聚成一片，否则它和「hotspot 位置错」之外还多了
    # 「hotspot 不自洽」这个混淆因素，对照就不干净了。
    negative = pick_compact_set(
        exposed, coord_by_resi, n=args.n_hotspots, max_spread=args.hotspot_spread
    )
    print(f"[info] 阴性对照候选（背面、空间聚集）: {[d['tag'] for d in negative]}")

    out = {
        "pdb_id": src["pdb_id"],
        "target_chain": tgt["chain"],
        "partner_chain": prt["chain"],
        "contact_cutoff": cutoff,
        "observed_res_range": [min(observed), max(observed)],
        "recommended_design": [d["tag"] for d in recommended],
        "recommended_negative_control": [d["tag"] for d in negative],
        "interface_candidates": interface,
        "negative_control_candidates": negative,
        "note": (
            "recommended_design / recommended_negative_control 可以直接填进 "
            "config/target_pdl1.yaml 的 hotspots 两个字段。填之前务必在 PyMOL 里"
            "打开 clean_pdb 看一眼：design 组应聚在 PD-1 结合面上，"
            "negative_control 组应在背面且不与界面重叠。"
            "interface_candidates 给出全部界面残基，想换 hotspot 从这里挑。"
        ),
    }
    dest = REPO_ROOT / "data" / "target" / "hotspots.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[write] hotspot 候选 -> {dest}")
    print("\n下一步：编辑 config/target_pdl1.yaml 的 hotspots.design 和 hotspots.negative_control")


if __name__ == "__main__":
    main()
