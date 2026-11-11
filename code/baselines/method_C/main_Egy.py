import os
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage import graph
from scipy.ndimage import distance_transform_edt, label

# --- 定位仓库 code/ 目录（含 paths.py），保证脚本可在任意工作目录下运行 ---
_CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").is_file())
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from paths import (DEFAULT_DATASET, RESULT_ROOT, dataset_paths,
                   ensure_dir, list_images, print_config)

from endpoint_detector_Egy import EndpointDetector
from junctionpoint_detector_Egy import Junctionpoint_detector
from midpoint_detector_Egy import InflexionDetector
from visualizer_Egy import GraphVisualizer

"""A Fuzzy Hierarchical Attributed Graph Approach for Handwritten Egyptian
Hieroglyphs description and matching."""

# ---------- 路径（相对仓库根目录，见 code/paths.py）----------
# 数据集设置：只改下一行（相对 <repo>/data，无需绝对路径）；DEFAULT_DATASET = 跟随全局
DATASET = DEFAULT_DATASET              # 例："skeleton_756"
input_dir, _, DATASET_TAG = dataset_paths(DATASET)         # <repo>/data/skeleton_<N>
adj_output_dir = RESULT_ROOT / "baseline" / f"C_matrix_{DATASET_TAG}"
vis_output_dir = RESULT_ROOT / "baseline" / f"C_matrix_{DATASET_TAG}_visualization"
# ---------- 节点合并函数 ----------
def merge_singular_points(endpoints, junctions_raw, merge_dist=8):
    """
    合并端点和交叉点，消除距离过近的冗余节点。
    优先级：交叉点 > 端点（簇内若有交叉点，则保留交叉点质心）。
    """
    ep_list = [(x, y, 0) for (x, y) in endpoints]
    jp_list = [(x, y, 1) for (x, y) in junctions_raw]
    all_pts = ep_list + jp_list
    if not all_pts:
        return []

    pts_arr = np.array([(p[0], p[1]) for p in all_pts])
    n = len(pts_arr)
    used = np.zeros(n, dtype=bool)
    merged = []

    for i in range(n):
        if used[i]:
            continue
        dists = np.linalg.norm(pts_arr - pts_arr[i], axis=1)
        cluster_idx = np.where(dists < merge_dist)[0]
        used[cluster_idx] = True

        cluster_types = [all_pts[idx][2] for idx in cluster_idx]
        if 1 in cluster_types:  # 包含交叉点
            jp_idx = [idx for idx in cluster_idx if all_pts[idx][2] == 1]
            jp_pts = pts_arr[jp_idx]
            centroid = np.mean(jp_pts, axis=0)
            merged.append((int(round(centroid[0])), int(round(centroid[1])), 1))
        else:  # 纯端点
            ep_pts = pts_arr[cluster_idx]
            centroid = np.mean(ep_pts, axis=0)
            merged.append((int(round(centroid[0])), int(round(centroid[1])), 0))
    return merged


# ---------- 骨架吸附与邻接矩阵 ----------
def snap_to_skeleton(skeleton_img, nodes):
    skeleton = skeleton_img > 0
    if not np.any(skeleton):
        return nodes
    dist, indices = distance_transform_edt(~skeleton, return_indices=True)
    snapped = []
    for x, y in nodes:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < skeleton.shape[0] and 0 <= xi < skeleton.shape[1]:
            if skeleton[yi, xi]:
                snapped.append((x, y))
            else:
                near_y, near_x = indices[:, yi, xi]
                snapped.append((near_x, near_y))
        else:
            snapped.append((x, y))
    return snapped


def build_adjacency(skeleton_img, nodes, max_distance=400):
    if not nodes:
        return np.zeros((0, 0), dtype=int)

    snapped_nodes = snap_to_skeleton(skeleton_img, nodes)
    n = len(snapped_nodes)
    if n < 2:
        return np.zeros((n, n), dtype=int)

    labeled, _ = label(skeleton_img > 0)
    node_labels = []
    for x, y in snapped_nodes:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < labeled.shape[0] and 0 <= xi < labeled.shape[1]:
            node_labels.append(labeled[yi, xi])
        else:
            node_labels.append(0)

    cost = np.where(skeleton_img > 0, 1, np.inf)
    node_pos = [(int(round(y)), int(round(x))) for x, y in snapped_nodes]

    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if node_labels[i] != node_labels[j] or node_labels[i] == 0:
                continue
            if abs(node_pos[i][0] - node_pos[j][0]) + abs(node_pos[i][1] - node_pos[j][1]) > max_distance:
                continue
            try:
                path, cost_val = graph.route_through_array(
                    cost, node_pos[i], node_pos[j], fully_connected=True
                )
                if path is not None and cost_val < np.inf:
                    adj[i, j] = adj[j, i] = 1
            except:
                pass
    return adj


# ------------------ 主流程 ------------------
def main():

    print_config()
    ensure_dir(adj_output_dir)
    ensure_dir(vis_output_dir)

    # ==================== 初始化检测器 ====================
    ep_detector = EndpointDetector(prune_length=4)
    jp_detector = Junctionpoint_detector(merge_distance=4)
    in_detector = InflexionDetector(curvature_thresh=0.03, t_suppress=0.15)
    visualizer = GraphVisualizer(save_visualizations=True)

    # ==================== 遍历图像 ====================
    image_files = list_images(input_dir)
    if not image_files:
        print(f"错误：输入目录中没有找到图像文件 - {input_dir}")
        return
    print(f"共找到 {len(image_files)} 张图像，开始处理...")

    file_counter = 1
    for file_path in image_files:
        filename = file_path.name
        print(f"正在处理第 {file_counter} 个文件: {filename}")

        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  警告: 无法读取图像 {filename}，跳过")
            file_counter += 1
            continue

        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # ---------- 关键点检测 ----------
        # 检测器内部用 cv2.imread 读取，统一传字符串路径
        image_path = str(file_path)
        endpoints = ep_detector.detect(image_path)        
        junctions_raw = jp_detector.detect(image_path)   
        inflexions = in_detector.detect(image_path)        

        # ---------- 合并端点和交叉点 ----------
        merged_singular = merge_singular_points(
            endpoints, junctions_raw, merge_dist=12
        )

        # ---------- 吸附所有节点到骨架 ----------
        merged_coords = [(x, y) for (x, y, _) in merged_singular]
        snapped_merged = snap_to_skeleton(binary, merged_coords)

        snapped_inflex = snap_to_skeleton(binary, inflexions)

        # ---------- 构建节点列表与邻接矩阵 ----------
        node_list = []
        node_coords = []
        for (_, _, typ), (sx, sy) in zip(merged_singular, snapped_merged):
            node_list.append((sx, sy, typ))
            node_coords.append((sx, sy))
        for (sx, sy) in snapped_inflex:
            node_list.append((sx, sy, 2))
            node_coords.append((sx, sy))

        # 构建邻接矩阵
        adj_matrix = build_adjacency(binary, node_coords, max_distance=400)
        nodes_npy = np.array(node_list, dtype=np.int32)

        # ---------- 保存文件 ----------
        base_name = os.path.splitext(filename)[0]
        np.save(os.path.join(adj_output_dir, f"{base_name}adj.npy"), adj_matrix)
        np.save(os.path.join(adj_output_dir, f"{base_name}nodes.npy"), nodes_npy)

        # ---------- 可视化 ----------
        # 提取吸附后的端点和交叉点用于绘图
        vis_endpoints = [(sx, sy) for (_, _, typ), (sx, sy)
                         in zip(merged_singular, snapped_merged) if typ == 0]
        vis_junctions = [{'position': (sx, sy), 'branch_count': 3}
                         for (_, _, typ), (sx, sy)
                         in zip(merged_singular, snapped_merged) if typ == 1]

        vis_path = os.path.join(vis_output_dir, f"keypoints_{filename}")
        visualizer.visualize_keypoints(
            skeleton_img=binary,
            endpoints=vis_endpoints,
            junctions=vis_junctions,
            inflexions=snapped_inflex,
            paths=None,
            save_path=vis_path
        )

        # 打印统计信息
        ep_cnt = sum(1 for _, _, t in merged_singular if t == 0)
        jp_cnt = sum(1 for _, _, t in merged_singular if t == 1)
        print(f"  节点总数: {len(node_list)} (端点: {ep_cnt}, 交叉点: {jp_cnt}, 拐点: {len(inflexions)})")
        file_counter += 1

    print("所有图像处理完成！")


if __name__ == "__main__":
    main()