# -*- coding: utf-8 -*-
"""
run_sensitivity_analysis.py
---------------------------------------------------------------
参数映射说明：
  1a: endpoint_detector.min_branch_len          ∈ {3,4,5,6,7,8}
  1b: circular_detector.track_steps             ∈ {4,6,8,10,12,14}   ← 修正
  2a: circular_detector.min_distance            ∈ {4,6,8,10,12,14}
  2b: GraphPostProcessor.merge_dist             ∈ {4,5,6,7,8,9}
  3a: GlobalMidpointDetector.min_chord_length   ∈ {40,50,60,70,80,90}
  3b: GlobalMidpointDetector.min_spacing        ∈ {10,15,20,25,30,35}
  4a: 首次建图 firs min_edge_len（原值 6）           ∈ {3,4,5,6,7,8}
  4b: 临时/补边建图 sub min_edge_len（原值 3）      ∈ {1,2,3,4,5,6}
---------------------------------------------------------------
"""
# -*- coding: utf-8 -*-
"""
run_sensitivity_analysis.py
以 main.py 为基准的敏感性分析（无可视化，逻辑与 main.py 完全一致）
"""
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

# --- 定位仓库 code/ 目录（含 paths.py），保证脚本可在任意工作目录下运行 ---
_CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").is_file())
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from paths import (CODE_ROOT, RESULT_ROOT, DEFAULT_DATASET, dataset_paths,
                   list_images, print_config)

# ==================================================================
# 路径配置（相对仓库根目录，见 code/paths.py）
# ==================================================================
CODE_DIR    = CODE_ROOT / "method_A" / "src"                   # 本文方法源码
# 数据集设置：只改下一行（相对 <repo>/data，无需绝对路径）；DEFAULT_DATASET = 跟随全局
DATASET     = DEFAULT_DATASET                                  # 例："skeleton_756"
INPUT_DIR, _, DATASET_TAG = dataset_paths(DATASET)             # <repo>/data/skeleton_<N>
BASE_OUTPUT = RESULT_ROOT / "parameter_sensitivity_analysis"   # 各参数组合的输出根目录
BASE_NAME   = f"A_matrix_{DATASET_TAG}_t5"                      # 各组合输出目录的前缀

CODE_DIR_STR = str(CODE_DIR)

# 确保 src 优先级最高
if CODE_DIR_STR in sys.path:
    sys.path.remove(CODE_DIR_STR)
sys.path.insert(0, CODE_DIR_STR)

# 清理旧模块缓存
for _m in ['endpoint_detector', 'junction_detector', 'midpoint_detector',
           'graph_postprocessor', 'global_midpoint_detector',
           'corner_remover', 'degree_edge_builder', 'graph_utils']:
    sys.modules.pop(_m, None)

from endpoint_detector import EndpointDetector
from junction_detector import JunctionDetector
from midpoint_detector import CircularMidpointDetector
from graph_utils import build_graph_from_skeleton, snap_to_skeleton
from graph_postprocessor import GraphPostProcessor
from global_midpoint_detector import GlobalMidpointDetector
from corner_remover import RedundantCornerRemover
from degree_edge_builder import DegreeLimitedEdgeBuilder

# ==================================================================
# 模块加载验证
# ==================================================================
import endpoint_detector as _ed
import junction_detector as _jd
import midpoint_detector as _md
import graph_postprocessor as _gp
import degree_edge_builder as _deb

print("=" * 78)
print("模块加载验证")
_ok = True
for _mod in [_ed, _jd, _md, _gp, _deb]:
    _mod_file = str(Path(_mod.__file__).resolve())
    flag = "✅" if _mod_file.startswith(CODE_DIR_STR) else "❌"
    if flag == "❌":
        _ok = False
    print(f"  {flag} {_mod.__name__:30s}: {_mod.__file__}")
print("=" * 78 + "\n")
if not _ok:
    print("❌ 模块加载路径不正确，请检查 sys.path")
    sys.exit(1)


# ==================================================================
# 默认配置（= main.py 当前值）
# ==================================================================
DEFAULT_CONFIG = {
    'endpoint_min_branch_len': 2,
    'circular_min_distance':   12,
    'circular_track_steps':    8,
    'merge_dist':              7,    
    'min_chord_length':        50,
    'min_spacing':             15,
    'min_edge_len_initial':    4,
    'min_edge_len_temp':       4,
}


def format_duration(seconds):
    if seconds < 0:
        seconds = 0
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


# ==================================================================
# 单次实验（完全复制 main.py 的 main()，仅去掉可视化）
# ==================================================================
def run_once(overrides, output_suffix):
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(overrides)

    adj_output_dir = os.path.join(BASE_OUTPUT, f"{BASE_NAME}_{output_suffix}")
    os.makedirs(adj_output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"[实验] {output_suffix}  覆盖: {overrides}")
    print(f"[输出] {adj_output_dir}")
    print("=" * 70)

    t0 = time.time()

    # ---------- 开关 ----------
    USE_ENDPOINT_VERIFY     = True
    USE_MIDPOINT_VERIFY     = True
    USE_MIDPOINT_DEDUP      = True
    USE_MERGE_CLOSE         = True
    USE_MERGE_PRIORITY      = True
    USE_GLOBAL_MIDPOINT     = True
    USE_ANGLE_CORNER_REMOVE = True
    USE_DEGREE_LIMITED_EDGE = True

    # ---------- 构造各模块 ----------
    endpoint_detector = EndpointDetector(
        min_branch_len=cfg['endpoint_min_branch_len'] if USE_ENDPOINT_VERIFY else 1
    )
    junction_detector = JunctionDetector()   # 无去重，与 main.py 一致
    circular_detector = CircularMidpointDetector(
        min_distance=cfg['circular_min_distance'] if USE_MIDPOINT_DEDUP else 3,
        track_steps=cfg['circular_track_steps'] if USE_MIDPOINT_VERIFY else 3,
    )
    post_processor = GraphPostProcessor(
        merge_dist=cfg['merge_dist'],
        type_priority={0: 0, 1: 1, 2: 2},
        enable_merge=USE_MERGE_CLOSE,
        use_type_priority=USE_MERGE_PRIORITY,
    )
    corner_remover = RedundantCornerRemover()
    global_mid_detector = GlobalMidpointDetector(
        min_chord_length=cfg['min_chord_length'],
        use_curvature=False,
        straightness_threshold=0.9,
        min_spacing=cfg['min_spacing'],
    )

    min_edge_len_initial = cfg['min_edge_len_initial']
    min_edge_len_temp    = cfg['min_edge_len_temp']

    node_counts = []
    file_counter = 1

    for file_path in list_images(INPUT_DIR):
        filename = file_path.name
        print(f"  处理 {file_counter}: {filename}")

        img_color = cv2.imread(str(file_path))
        if img_color is None:
            print("    警告: 无法读取")
            file_counter += 1
            continue

        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # ---------- 关键点检测 ----------
        endpoints = endpoint_detector.find_endpoints(gray)
        junctions = junction_detector.find_junctions(gray, endpoints=endpoints)
        circular_midpoints = circular_detector.find_circular_midpoints(
            gray, endpoints=endpoints, junctions=junctions
        )

        # ---------- 构建 node_list + branch_count ----------
        node_list = []
        node_branch_counts = []
        for (x, y) in endpoints:
            node_list.append((x, y, 0))
            node_branch_counts.append(None)
        for j in junctions:
            node_list.append((j['position'][0], j['position'][1], 1))
            node_branch_counts.append(j.get('branch_count', 4))
        for m in circular_midpoints:
            if isinstance(m, (tuple, list)) and len(m) == 2:
                node_list.append((m[0], m[1], 2))
            elif isinstance(m, dict):
                node_list.append((m['position'][0], m['position'][1], 2))
            node_branch_counts.append(None)

        coords = [(x, y) for (x, y, _) in node_list]
        snapped = snap_to_skeleton(binary, coords)
        node_list_snapped = [(sx, sy, t) for (sx, sy), (_, _, t) in zip(snapped, node_list)]

        # ---------- snap 后重建 junction_branch_map ----------
        junction_branch_map = {}
        for (sx, sy, t), bc in zip(node_list_snapped, node_branch_counts):
            if t == 1 and bc is not None:
                junction_branch_map[(sx, sy)] = bc

        proc_coords = [(x, y) for (x, y, _) in node_list_snapped]
        adj = build_graph_from_skeleton(binary, proc_coords,
                                        min_edge_len=min_edge_len_initial, snap=False)
        edges = []
        n = len(node_list_snapped)
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append((proc_coords[i], proc_coords[j]))

        # ---------- ① 邻近合并 ----------
        final_nodes, _ = post_processor.process(node_list_snapped, edges, skeleton=binary)

        # 删除孤立节点
        node_degrees = {n[:2]: 0 for n in final_nodes}
        for p1, p2 in edges:
            if p1 in node_degrees:
                node_degrees[p1] += 1
            if p2 in node_degrees:
                node_degrees[p2] += 1
        final_nodes = [n for n in final_nodes if node_degrees.get(n[:2], 0) > 0]

        if len(final_nodes) == 0:
            print("    警告: 无有效节点")
            file_counter += 1
            continue

        # ---------- ⑥ 拐点删除 ----------
        if USE_ANGLE_CORNER_REMOVE:
            temp_coords = [(x, y) for (x, y, _) in final_nodes]
            temp_adj = build_graph_from_skeleton(binary, temp_coords,
                                                 min_edge_len=min_edge_len_temp, snap=False)
            temp_edges = []
            n_temp = len(final_nodes)
            for i in range(n_temp):
                for j in range(i + 1, n_temp):
                    if temp_adj[i, j]:
                        temp_edges.append((temp_coords[i], temp_coords[j]))

            final_nodes, temp_edges = corner_remover.remove(
                final_nodes, temp_edges, angle_min=150, angle_max=180
            )

            node_degrees = {n[:2]: 0 for n in final_nodes}
            for p1, p2 in temp_edges:
                if p1 in node_degrees:
                    node_degrees[p1] += 1
                if p2 in node_degrees:
                    node_degrees[p2] += 1
            final_nodes = [n for n in final_nodes if node_degrees.get(n[:2], 0) > 0]

        if len(final_nodes) == 0:
            print("    警告: 拐点删除后无有效节点")
            file_counter += 1
            continue

        # ---------- ⑤ 全局中点 ----------
        if USE_GLOBAL_MIDPOINT:
            temp_coords = [(x, y) for (x, y, _) in final_nodes]
            temp_adj = build_graph_from_skeleton(binary, temp_coords,
                                                 min_edge_len=min_edge_len_temp, snap=False)
            global_midpoints = global_mid_detector.find_global_midpoints(
                binary, final_nodes, temp_adj
            )
            existing_coords = set(temp_coords)
            for gx, gy, gtype in global_midpoints:
                if (gx, gy) not in existing_coords:
                    final_nodes.append((gx, gy, gtype))
                    existing_coords.add((gx, gy))

        # ---------- ⑦ 最终建边（传 junction_branch_map）----------
        if USE_DEGREE_LIMITED_EDGE:
            degree_edge_builder = DegreeLimitedEdgeBuilder(
                max_degree={0: 1, 1: 4, 2: 2, 3: 2},
                junction_branch_map=junction_branch_map
            )
            final_edges, final_adj = degree_edge_builder.build(binary, final_nodes)
        else:
            temp_coords = [(x, y) for (x, y, _) in final_nodes]
            final_adj = build_graph_from_skeleton(binary, temp_coords,
                                                  min_edge_len=min_edge_len_temp, snap=False)
            final_edges = []
            n_final = len(final_nodes)
            for i in range(n_final):
                for j in range(i + 1, n_final):
                    if final_adj[i, j]:
                        final_edges.append((temp_coords[i], temp_coords[j]))

        node_counts.append(len(final_nodes))
        print(f"    节点={len(final_nodes)}, 边={len(final_edges)}")

        # ---------- 保存（只存 .npy）----------
        base = os.path.splitext(filename)[0]
        np.save(os.path.join(adj_output_dir, f"{base}nodes.npy"),
                np.array(final_nodes, dtype=np.int32))
        np.save(os.path.join(adj_output_dir, f"{base}adj.npy"), final_adj)

        file_counter += 1

    total = time.time() - t0
    avg_nodes = (sum(node_counts) / len(node_counts)) if node_counts else 0.0
    print(f"  完成！{len(node_counts)} 张，平均节点 {avg_nodes:.2f}，耗时 {format_duration(total)}")

    return {
        'output_suffix': output_suffix,
        'overrides': overrides,
        'avg_nodes': avg_nodes,
        'num_images': len(node_counts),
        'elapsed': total,
    }


# ==================================================================
# 实验定义
#   （去掉 circular_min_distance=2，会产生大量空图）
# ==================================================================
EXPERIMENTS = [
    ('1a', 'endpoint_min_branch_len', [1, 2, 3, 4, 5, 6, 7, 8],       'min_branch_len'),
    ('1b', 'circular_track_steps',    [3, 4, 6, 8, 10, 12, 14],       'track_steps'),
    ('2a', 'circular_min_distance',   [2, 3, 4, 6, 8, 10, 12, 14],    'circular_min_distance'),
    ('2b', 'merge_dist',              [4, 5, 6, 7, 8, 9],          'merge_dist'),
    ('3a', 'min_chord_length',        [30, 40, 50, 60, 70, 80, 90],    'min_chord_length'),
    ('3b', 'min_spacing',             [10, 15, 20, 25, 30, 35],    'min_spacing'),
    ('4a', 'min_edge_len_initial',    [3, 4, 5, 6, 7, 8],          'min_edge_len'),
    ('4b', 'min_edge_len_temp',       [1, 2, 3, 4, 5, 6],          'temp_min_edge_len'),
]


# ==================================================================
# 主程序
# ==================================================================
def main():
    overall_start = time.time()
    all_results = []

    print_config()
    if not INPUT_DIR.is_dir():
        print(f"❌ 输入目录不存在: {INPUT_DIR}")
        sys.exit(1)

    for group_id, key, values, suffix_name in EXPERIMENTS:
        print("\n" + "#" * 70)
        print(f"# 实验组 {group_id}  |  {key}  |  {values}")
        print("#" * 70)

        for v in values:
            overrides = {key: v}
            suffix = f"{suffix_name}={v}"
            try:
                res = run_once(overrides, suffix)
                res['group'] = group_id
                res['param'] = key
                res['value'] = v
                all_results.append(res)
            except Exception as e:
                traceback.print_exc()
                print(f"  [错误] {suffix}: {e}")
                all_results.append({
                    'group': group_id, 'param': key, 'value': v,
                    'output_suffix': suffix, 'overrides': overrides,
                    'avg_nodes': None, 'num_images': 0, 'elapsed': 0.0,
                    'error': str(e),
                })

    total_elapsed = time.time() - overall_start
    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    print(f"{'组':<5}{'参数':<28}{'取值':<8}{'平均节点数':<14}{'图片数':<8}{'耗时(s)':<10}")
    print("-" * 78)
    for r in all_results:
        avg = r.get('avg_nodes')
        avg_str = f"{avg:.2f}" if isinstance(avg, (int, float)) else "ERROR"
        print(f"{r.get('group',''):<5}"
              f"{str(r.get('param','')):<28}"
              f"{str(r.get('value','')):<8}"
              f"{avg_str:<14}"
              f"{r.get('num_images',0):<8}"
              f"{r.get('elapsed',0):<10.3f}")
    print("-" * 78)
    print(f"总耗时: {format_duration(total_elapsed)}")
    print("=" * 78)


if __name__ == "__main__":
    main()