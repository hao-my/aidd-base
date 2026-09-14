"""
评测空壳（Phase 1）

当前：只打印占位说明，不计算指标。
Phase 2 计划：
  - RDKit 有效率 / 唯一率 / 相对训练集新颖性
  - QED 分布
  - （可选）调用 08 的 BACE pIC50 打分器或 09 的 logS 模型做筛选
"""

from pathlib import Path

import config


def evaluate(generated_csv=None, checkpoint=None):
    generated_csv = Path(generated_csv or (config.OUTPUT_DIR / "generated_molecules.csv"))
    checkpoint = Path(checkpoint or config.CHECKPOINT_FINAL)

    print("=" * 60)
    print("evaluate.py 是 Phase 1 空壳，尚未实现指标计算。")
    print()
    print("预期输入:")
    print(f"  - 生成 CSV : {generated_csv}  exists={generated_csv.exists()}")
    print(f"  - 权重     : {checkpoint}  exists={checkpoint.exists()}")
    print(f"  - 训练集   : {config.DATASET_FILE}  exists={config.DATASET_FILE.exists()}")
    print()
    print("Phase 2 将实现:")
    print("  1. 有效率 / 唯一率 / 新颖性")
    print("  2. QED 统计")
    print("  3. 可选：接 08 scorer 或 09 multimodal 做性质筛选")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
