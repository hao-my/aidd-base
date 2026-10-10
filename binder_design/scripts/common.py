"""配置加载、外部工具定位、计时与吞吐记录。

所有阶段脚本共用这里的东西。设计原则：
- 路径全部相对仓库根，脚本在哪个 cwd 调用都不影响结果。
- 外部工具（RFdiffusion、dl_binder_design）位置从环境变量读，
  因为它们装在租的 GPU 机上，路径每次租都可能不同。
- 每个阶段把耗时写进 outputs/throughput.json，这是试点阶段的主要产物。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = REPO_ROOT / "outputs"
THROUGHPUT_FILE = OUTPUTS / "throughput.json"

# Windows 控制台默认 GBK，打印 Å 之类的字符会抛 UnicodeEncodeError。
# 脚本主要在 Linux 上跑，但靶点准备和分析在本机跑，所以统一改成 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

def load_yaml(rel_or_abs: str | Path) -> dict:
    path = Path(rel_or_abs)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"配置不存在: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(rel: str | Path) -> Path:
    """把配置里的相对路径解析成绝对路径。"""
    path = Path(rel)
    return path if path.is_absolute() else REPO_ROOT / path


# --------------------------------------------------------------------------
# 外部工具
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Tools:
    rfdiffusion: Path
    dl_binder: Path

    @property
    def run_inference(self) -> Path:
        return self.rfdiffusion / "scripts" / "run_inference.py"

    @property
    def dl_interface_design(self) -> Path:
        return self.dl_binder / "mpnn_fr" / "dl_interface_design.py"

    @property
    def af2_predict(self) -> Path:
        return self.dl_binder / "af2_initial_guess" / "predict.py"

    @property
    def soluble_weights(self) -> Path:
        return (
            self.dl_binder
            / "mpnn_fr"
            / "ProteinMPNN"
            / "soluble_model_weights"
            / "v_48_020.pt"
        )


def load_tools(require: bool = True) -> Tools:
    """从 RFDIFFUSION_DIR / DL_BINDER_DIR 环境变量定位外部工具。

    require=False 时只返回路径不做存在性检查，便于在本机（无 GPU）
    做 dry-run 和参数拼接的单元检查。
    """
    rf = os.environ.get("RFDIFFUSION_DIR", "")
    dl = os.environ.get("DL_BINDER_DIR", "")
    if require:
        missing = [name for name, val in (("RFDIFFUSION_DIR", rf), ("DL_BINDER_DIR", dl)) if not val]
        if missing:
            raise EnvironmentError(
                "未设置环境变量 " + ", ".join(missing) + "。参见 env/SETUP.md。"
            )
    tools = Tools(rfdiffusion=Path(rf), dl_binder=Path(dl))
    if require:
        for label, path in (
            ("RFdiffusion run_inference.py", tools.run_inference),
            ("dl_interface_design.py", tools.dl_interface_design),
            ("af2_initial_guess/predict.py", tools.af2_predict),
        ):
            if not path.exists():
                raise FileNotFoundError(f"找不到 {label}: {path}")
    return tools


# --------------------------------------------------------------------------
# 子进程
# --------------------------------------------------------------------------

def run(
    cmd: list,
    *,
    cwd: Path | None = None,
    env: dict | None = None,
    dry_run: bool = False,
) -> float:
    """执行命令并返回墙钟秒数。dry_run 只打印不执行。"""
    printable = " ".join(str(c) for c in cmd)
    print(f"[run] {printable}", flush=True)
    if dry_run:
        return 0.0
    start = time.perf_counter()
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"命令失败（退出码 {proc.returncode}）: {printable}")
    print(f"[done] {elapsed:.1f}s", flush=True)
    return elapsed


# --------------------------------------------------------------------------
# 吞吐记录
# --------------------------------------------------------------------------

def record_throughput(stage: str, n_items: int, seconds: float, note: str = "") -> None:
    """累积写入 outputs/throughput.json。

    stage 例如 "diffusion.noise00"、"sequence"、"predict"。
    n_items 是该阶段处理的设计数，用来算单位耗时。
    """
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if THROUGHPUT_FILE.exists():
        data = json.loads(THROUGHPUT_FILE.read_text(encoding="utf-8"))
    entry = {
        "n_items": n_items,
        "seconds": round(seconds, 1),
        "seconds_per_item": round(seconds / n_items, 2) if n_items else None,
        "gpu": os.environ.get("BD_GPU_LABEL", "unknown"),
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if note:
        entry["note"] = note
    data.setdefault(stage, []).append(entry)
    THROUGHPUT_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[throughput] {stage}: {entry['seconds_per_item']}s/design -> {THROUGHPUT_FILE}")


def die(msg: str) -> None:
    print(f"错误: {msg}", file=sys.stderr)
    raise SystemExit(1)
