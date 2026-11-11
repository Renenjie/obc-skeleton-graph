# main.py
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter

# --- 定位仓库 code/ 目录（含 paths.py），保证脚本可在任意工作目录下运行 ---
_CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").is_file())
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from paths import (RESULT_ROOT, dataset_paths,
                   ensure_dir, list_images, print_config)

from endpoint_detector import EndpointDetector
from junction_detector import JunctionDetector
from midpoint_detector import CircularMidpointDetector
from visualizer import GraphVisualizer
from graph_utils import build_graph_from_skeleton, snap_to_skeleton
from graph_postprocessor import GraphPostProcessor
from global_midpoint_detector import GlobalMidpointDetector
from corner_remover import RedundantCornerRemover
from degree_edge_builder import DegreeLimitedEdgeBuilder


# ---------- 主程序 ----------
# ============================================================================
# 数据集设置：只改下面这一行即可（相对 <repo>/data，不需要绝对路径）
#   "skeleton_3780" -> data/skeleton_3780 + data/Matrix_3780 -> A_matrix_3780_t1
#   "skeleton_756"  -> data/skeleton_756  + data/Matrix_756  -> A_matrix_756_t1
# 输入骨架图与真值矩阵会一起切换，避免"图用 756、真值还用 3780"对不上。
# ============================================================================
DATASET = "skeleton_3780"

input_dir, gt_dir, DATASET_TAG = dataset_paths(DATASET)   # <repo>/data/skeleton_<N>, <repo>/data/Matrix_<N>
adj_output_dir = RESULT_ROOT / f"A_matrix_{DATASET_TAG}"
vis_output_dir = RESULT_ROOT / f"A_matrix_{DATASET_TAG}_visualization"


def format_duration(seconds):
    if seconds < 0:
        seconds = 0
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"


def main():
    task_start_time = time.time()

    print_config()
    ensure_dir(adj_output_dir)
    ensure_dir(vis_output_dir)

    image_files = list_images(input_dir)
    if not image_files:
        print(f"错误：输入目录中没有找到图像文件 - {input_dir}")
        return
    print(f"共找到 {len(image_files)} 张图像，开始处理...")

    # ===== 开关 =====
    USE_ENDPOINT_VERIFY     = True     # 端点分支长度验证
    USE_MIDPOINT_VERIFY     = True     # 局部中点追踪验证
    USE_MIDPOINT_DEDUP      = True     # 局部中点切比雪夫去重
    
    USE_MERGE_CLOSE         = True     # ① 邻近节点合并
    USE_MERGE_PRIORITY      = True  # ①-1 合并时是否使用类型优先级
    USE_GLOBAL_MIDPOINT     = True     # ⑤ 全局中点添加
    USE_ANGLE_CORNER_REMOVE = True     # ⑥ 夹角冗余拐点删除
    USE_DEGREE_LIMITED_EDGE = True     # ⑦ 度数限制建边

    endpoint_detector = EndpointDetector(
        min_branch_len=2 if USE_ENDPOINT_VERIFY else 1
    )
    junction_detector = JunctionDetector()
    circular_detector = CircularMidpointDetector(
        min_distance=12 if USE_MIDPOINT_DEDUP else 1,
        track_steps=8 if not USE_MIDPOINT_VERIFY else 8
    )
    post_processor = GraphPostProcessor(
        merge_dist=7,
        type_priority={0: 0, 1: 1, 2: 2},
        enable_merge=USE_MERGE_CLOSE,
        use_type_priority=USE_MERGE_PRIORITY
    )
    corner_remover = RedundantCornerRemover()
    global_mid_detector = GlobalMidpointDetector(
        min_chord_length=50,
        use_curvature=False,
        straightness_threshold=0.9,
        min_spacing=15
    )
    visualizer = GraphVisualizer(save_visualizations=True)

    node_counts = []
    file_counter = 1

    for file_path in image_files:
        filename = file_path.name
        print(f"处理 {file_counter}: {filename}")

        img_color = cv2.imread(str(file_path))
        if img_color is None:
            print("  警告: 无法读取")
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

        # 构建 node_list，同步记录每个节点的 branch_count（仅交叉点非 None）
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

        # snap 后重建 junction_branch_map（键为 snap 后的坐标）
        junction_branch_map = {}
        for (sx, sy, t), bc in zip(node_list_snapped, node_branch_counts):
            if t == 1 and bc is not None:
                junction_branch_map[(sx, sy)] = bc
        proc_coords = [(x, y) for (x, y, _) in node_list_snapped]
        
        min_edge_len=4 #L_first
        
        adj = build_graph_from_skeleton(binary, proc_coords, min_edge_len=4, snap=False) # L_first
        edges = []
        n = len(node_list_snapped)
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append((proc_coords[i], proc_coords[j]))

        # ---------- ① 邻近节点合并 ----------
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
            print("  警告: 无有效节点，跳过")
            file_counter += 1
            continue
        
        
        min_edge_len=4 #L_sub
        
        # ---------- ⑥ 夹角拐点删除 ----------
        if USE_ANGLE_CORNER_REMOVE:
            temp_coords = [(x, y) for (x, y, _) in final_nodes]
            temp_adj = build_graph_from_skeleton(binary, temp_coords, min_edge_len, snap=False) #L_sub
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
            print("  警告: 拐点删除后无有效节点，跳过")
            file_counter += 1
            continue

        # ---------- ⑤ 全局中点添加 ----------
        if USE_GLOBAL_MIDPOINT:
            temp_coords = [(x, y) for (x, y, _) in final_nodes]
            temp_adj = build_graph_from_skeleton(binary, temp_coords, min_edge_len, snap=False) #L_sub
            global_midpoints = global_mid_detector.find_global_midpoints(binary, final_nodes, temp_adj)
            print(f"  检测到全局中点数量: {len(global_midpoints)}")
            existing_coords = set(temp_coords)
            for gx, gy, gtype in global_midpoints:
                if (gx, gy) not in existing_coords:
                    final_nodes.append((gx, gy, gtype))
                    existing_coords.add((gx, gy))

        # ---------- ⑦ 最终建边（交叉点度数 = 实际 branch_count） ----------
        if USE_DEGREE_LIMITED_EDGE:
            degree_edge_builder = DegreeLimitedEdgeBuilder(
                max_degree={0: 1, 1: 4, 2: 2, 3: 2},
                junction_branch_map=junction_branch_map
            )
            final_edges, final_adj = degree_edge_builder.build(binary, final_nodes)
        else:
            temp_coords = [(x, y) for (x, y, _) in final_nodes]
            final_adj = build_graph_from_skeleton(binary, temp_coords, min_edge_len, snap=False) #L_sub
            final_edges = []
            n_final = len(final_nodes)
            for i in range(n_final):
                for j in range(i + 1, n_final):
                    if final_adj[i, j]:
                        final_edges.append((temp_coords[i], temp_coords[j]))

        # ---------- 诊断打印（前 5 张） ----------
        if file_counter <= 5:
            type_counts = Counter(t for _, _, t in final_nodes)
            G_tmp = nx.from_numpy_array(final_adj)
            n_edges = G_tmp.number_of_edges()
            n_comp = nx.number_connected_components(G_tmp) if len(final_nodes) > 0 else 0
            cyclo = n_edges - len(final_nodes) + n_comp

            gt_name = os.path.splitext(filename)[0] + '.csv'
            gt_path = os.path.join(gt_dir, gt_name)
            cyclo_gt = -1
            if os.path.exists(gt_path):
                gt_adj = pd.read_csv(gt_path, header=None).values
                G_gt = nx.from_numpy_array(((gt_adj + gt_adj.T) > 0).astype(int))
                cyclo_gt = (G_gt.number_of_edges()
                            - G_gt.number_of_nodes()
                            + nx.number_connected_components(G_gt))

            junc_detail = []
            for i, (x, y, t) in enumerate(final_nodes):
                if t == 1:
                    actual_deg = int(final_adj[i].sum())
                    allowed_deg = junction_branch_map.get((x, y), 4)
                    junc_detail.append(f"{actual_deg}/{allowed_deg}")

            print(f"  [诊断] 端点={type_counts.get(0,0)} 交叉={type_counts.get(1,0)} "
                  f"中点={type_counts.get(2,0)} 环点={type_counts.get(3,0)} | "
                  f"节点={len(final_nodes)} 边={n_edges} 分量={n_comp} "
                  f"环={cyclo} GT环={cyclo_gt} "
                  f"误差={abs(cyclo-cyclo_gt) if cyclo_gt >= 0 else 'NA'}")
            if junc_detail:
                print(f"  [诊断] 交叉点度数(实际/允许): {junc_detail}")

        node_counts.append(len(final_nodes))
        print(f"  最终节点数量: {len(final_nodes)}, 边数量: {len(final_edges)}")

        # ---------- 保存结果 ----------
        base = os.path.splitext(filename)[0]
        np.save(os.path.join(adj_output_dir, f"{base}nodes.npy"),
                np.array(final_nodes, dtype=np.int32))
        np.save(os.path.join(adj_output_dir, f"{base}adj.npy"), final_adj)

        # ---------- 可视化 ----------
        ep = [(x, y) for (x, y, t) in final_nodes if t == 0]
        jn = [{'position': (x, y), 'branch_count': 3}
              for (x, y, t) in final_nodes if t == 1]
        mp = [(x, y) for (x, y, t) in final_nodes if t in (2, 3)]
        paths = [[p1, p2] for p1, p2 in final_edges]

        vis_path = os.path.join(vis_output_dir, f"detected_{filename}")
        visualizer.visualize_keypoints(skeleton_img=binary, endpoints=ep, junctions=jn,
                                       circular_midpoints=mp, paths=paths,
                                       save_path=vis_path)
        file_counter += 1

    if node_counts:
        avg_nodes = sum(node_counts) / len(node_counts)
        print(f"完成！共处理 {len(node_counts)} 张图像，平均最终节点数量: {avg_nodes:.2f}")
    else:
        print("完成！没有成功处理的图像。")

    total_elapsed = time.time() - task_start_time
    print(f"总耗时: {format_duration(total_elapsed)}")


if __name__ == "__main__":
    main()