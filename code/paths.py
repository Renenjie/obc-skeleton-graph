# -*- coding: utf-8 -*-
"""
paths.py — 项目统一路径配置（相对定位，可在任意目录下运行）
================================================================================

本文件把所有实验脚本中原本写死的 Windows 绝对路径（``E:\\skeleton_v1\\...``）
收敛到一处，改为**相对本仓库根目录**定位，从而保证：

* 仓库 clone / 解压到任意路径（Windows、Linux、macOS）都能直接运行；
* 移动数据或结果目录时，只需改环境变量，无需改代码；
* 所有输出位置集中可见，便于复现与清理。

目录约定
--------
::

    <repo>/                          # 仓库根目录（本例为 skeleton_v1/file）
    ├── data/
    │   ├── skeleton_3780/           # 输入骨架图（*.png）
    │   ├── Image_3780/              # 原始字形图（备用）
    │   └── Matrix_3780/             # 真值邻接矩阵（*.csv）
    ├── Experiment_results/          # 所有实验输出（自动创建）
    └── code/
        ├── paths.py                 # 本文件
        ├── method_A/src/            # 本文方法
        ├── baselines/method_B..F/   # 对比方法
        └── compare_code/            # 参数敏感性 / 消融 / 指标评估

环境变量覆盖（可选）
--------------------
数据或结果不在默认位置时，设置以下变量即可，无需改代码：

===========================  ==========================================
变量                          作用
===========================  ==========================================
SKELETON_DATASET             切换数据集，如 ``skeleton_756``；同时决定
                             输入目录与真值目录（skeleton_N ↔ Matrix_N）
SKELETON_DATA_DIR            覆盖 ``<repo>/data``
SKELETON_INPUT_DIR           覆盖输入骨架图目录（默认 data/<DATASET>）
SKELETON_GT_DIR              覆盖真值矩阵目录（默认 data/Matrix_<N>）
SKELETON_RESULT_DIR          覆盖结果根目录（默认 Experiment_results）
===========================  ==========================================

切换数据集（推荐做法：直接改文件，不涉及绝对路径）
---------------------------------------------------
两种粒度，都在文件里改，不用环境变量、不用命令行参数：

1. **全局切换** —— 改本文件里的 ``DEFAULT_DATASET``，所有脚本跟随::

       DEFAULT_DATASET = "skeleton_756"

2. **只改某个脚本** —— 改该脚本顶部自己的一行（默认继承全局）::

       DATASET = "skeleton_756"
       input_dir, gt_dir, DATASET_TAG = dataset_paths(DATASET)

无论哪种，输入骨架图与真值矩阵都会一起切换（``skeleton_N`` ↔ ``Matrix_N``），
结果目录名也会自动带上规模标签（如 ``A_matrix_756_t1``）。

用法
----
在子目录脚本中先定位本文件所在目录，再导入::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                                if (p / "paths.py").is_file())))
    from paths import SKELETON_DIR, GT_DIR, RESULT_ROOT

    print(SKELETON_DIR)   # <repo>/data/skeleton_3780
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "CODE_ROOT",
    "REPO_ROOT",
    "DATA_ROOT",
    "RESULT_ROOT",
    "DEFAULT_DATASET",
    "DATASET",
    "DATASET_TAG",
    "SKELETON_DIR",
    "GT_DIR",
    "IMAGE_EXTENSIONS",
    "env_path",
    "dataset_paths",
    "ensure_dir",
    "list_images",
    "baseline_dir",
    "print_config",
]

# ---------------------------------------------------------------------------
# 1. 仓库根目录：本文件位于 <repo>/code/paths.py
# ---------------------------------------------------------------------------
CODE_ROOT: Path = Path(__file__).resolve().parent          # <repo>/code
REPO_ROOT: Path = CODE_ROOT.parent                          # <repo>

# 输入骨架图支持的图片后缀（各脚本原先各自定义，此处统一）
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def env_path(var: str, default: Path) -> Path:
    """读取环境变量 ``var`` 作为路径；未设置则返回 ``default``。

    ``~`` 会被展开，返回值一律为绝对路径，避免受当前工作目录影响。
    """
    raw = os.environ.get(var)
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(default).resolve()


# ---------------------------------------------------------------------------
# 2. 数据与结果根目录
# ---------------------------------------------------------------------------
DATA_ROOT: Path = env_path("SKELETON_DATA_DIR", REPO_ROOT / "data")
RESULT_ROOT: Path = env_path("SKELETON_RESULT_DIR", REPO_ROOT / "Experiment_results")

# 数据集选择：默认 3780 张。
# 数据目录约定为 <DATA_ROOT>/skeleton_<N> 与 <DATA_ROOT>/Matrix_<N>，
# 因此只需切换数据集名，输入与真值会一起换，不会张冠李戴。
#
# 【全局开关】改下面这一行即可，所有脚本默认继承该值：
DEFAULT_DATASET: str = "skeleton_3780"
#
# 若只想改某一个脚本，去那个脚本里改它自己的 ``DATASET = "skeleton_756"``，
# 不必动本文件，也不需要设置环境变量或在命令行传参。
DATASET: str = (os.environ.get("SKELETON_DATASET") or DEFAULT_DATASET).strip()

# 数据集规模标签，如 "3780" / "756"，用于拼结果目录名
DATASET_TAG: str = DATASET.replace("skeleton_", "").replace("Matrix_", "")

# 输入骨架图 / 真值邻接矩阵
SKELETON_DIR: Path = env_path("SKELETON_INPUT_DIR", DATA_ROOT / f"skeleton_{DATASET_TAG}")
GT_DIR: Path = env_path("SKELETON_GT_DIR", DATA_ROOT / f"Matrix_{DATASET_TAG}")


def dataset_paths(name: str | None = None) -> tuple[Path, Path, str]:
    """按数据集名解析 ``(输入骨架图目录, 真值矩阵目录, 规模标签)``。

    只依赖 ``DATA_ROOT``（由本文件位置相对推导），不接受绝对路径，也不会
    受当前工作目录影响。各脚本因此可以只写一行::

        DATASET = "skeleton_756"
        input_dir, gt_dir, DATASET_TAG = dataset_paths(DATASET)
        # -> <repo>/data/skeleton_756, <repo>/data/Matrix_756, "756"

    参数
    ----
    name : ``"skeleton_756"`` / ``"Matrix_756"`` / ``"756"`` 均可；
           传 ``None`` 表示沿用全局 ``DATASET``（含环境变量覆盖）。
    """
    if name is None or not str(name).strip():
        return SKELETON_DIR, GT_DIR, DATASET_TAG

    tag = str(name).strip().replace("skeleton_", "").replace("Matrix_", "")
    if not tag:
        raise ValueError(f"无法从数据集名 {name!r} 解析出规模标签")
    return DATA_ROOT / f"skeleton_{tag}", DATA_ROOT / f"Matrix_{tag}", tag


# ---------------------------------------------------------------------------
# 3. 常用辅助函数
# ---------------------------------------------------------------------------
def ensure_dir(path: os.PathLike | str) -> Path:
    """创建目录（含父目录）并返回 ``Path`` 对象。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(directory: os.PathLike | str) -> list[Path]:
    """返回 ``directory`` 下的图片文件，**按文件名排序**以保证运行顺序可复现。

    排序很关键：``os.listdir`` 的返回顺序依赖文件系统，不同机器/不同拷贝
    顺序可能不同，会导致日志、进度编号乃至部分后处理的结果不一致。
    """
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(
        (p for p in d.iterdir()
         if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name,
    )


def baseline_dir(method: str, kind: str = "matrix") -> Path:
    """对比方法的结果目录，例如 ``baseline_dir("B") -> Experiment_results/baseline/B_matrix_3780``。

    参数
    ----
    method : {"B", "C", "D", "E", "F"}
    kind   : ``"matrix"`` 存放 ``*adj.npy`` / ``*nodes.npy``；
             ``"visualization"`` 存放可视化结果。
    """
    m = method.upper()
    if kind == "matrix":
        return RESULT_ROOT / "baseline" / f"{m}_matrix_{DATASET_TAG}"
    if kind == "visualization":
        return RESULT_ROOT / "baseline" / f"{m}_matrix_{DATASET_TAG}_visualization"
    raise ValueError(f"未知的 kind: {kind!r}（应为 'matrix' 或 'visualization'）")


def print_config() -> None:
    """打印当前生效的路径配置，便于在实验日志中留存复现环境信息。"""
    print("=" * 78)
    print("路径配置（code/paths.py）")
    for name, value in [
        ("DATASET", DATASET),
        ("REPO_ROOT", REPO_ROOT),
        ("DATA_ROOT", DATA_ROOT),
        ("SKELETON_DIR", SKELETON_DIR),
        ("GT_DIR", GT_DIR),
        ("RESULT_ROOT", RESULT_ROOT),
    ]:
        print(f"  {name:<14}: {value}")
    print("=" * 78)


if __name__ == "__main__":
    print_config()
