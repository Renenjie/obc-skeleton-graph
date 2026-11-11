import csv
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

from endpoint_junctionpoint_detector_h import KeypointDetector
from post_processor_h import PostProcessor
from visualizer_h import GraphVisualizer

"""Glyph graph isomorphism network for structure recognition of oracle bone
inscription"""

# ---------- 路径（相对仓库根目录，见 code/paths.py）----------
# 数据集设置：只改下一行（相对 <repo>/data，无需绝对路径）；DEFAULT_DATASET = 跟随全局
DATASET = DEFAULT_DATASET                # 例："skeleton_756"
skeleton_dir, _, DATASET_TAG = dataset_paths(DATASET)      # <repo>/data/skeleton_<N>
vis_dir = RESULT_ROOT / f"B_matrix_{DATASET_TAG}_visualization"
adj_save_dir = RESULT_ROOT / "baseline" / f"B_matrix_{DATASET_TAG}"
node_save_dir = RESULT_ROOT / "baseline" / f"B_matrix_{DATASET_TAG}"


# ---------- 辅助函数（保持不变）----------
def snap_to_skeleton(skeleton_img, nodes):
    """将节点坐标吸附到最近骨架点"""
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
    """基于骨架图像构建节点间的邻接矩阵"""
    nodes = snap_to_skeleton(skeleton_img, nodes)
    n = len(nodes)
    if n < 2:
        return np.zeros((n, n), dtype=int)

    labeled, num_features = label(skeleton_img > 0)
    node_labels = []
    for x, y in nodes:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < labeled.shape[0] and 0 <= xi < labeled.shape[1]:
            node_labels.append(labeled[yi, xi])
        else:
            node_labels.append(0)

    cost = np.where(skeleton_img > 0, 1, np.inf)
    node_pos = [(int(round(y)), int(round(x))) for x, y in nodes]

    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
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


# ---------- 主处理流程 ----------
def main():
    spur_threshold = 5.0
    merge_radius = 0
    print_config()
    ensure_dir(vis_dir)
    ensure_dir(adj_save_dir)
    ensure_dir(node_save_dir)

    kp_detector = KeypointDetector()
    post_processor = PostProcessor(spur_threshold, merge_radius = 5.0)

    visualizer = GraphVisualizer(save_visualizations=True)   # 使用导入的类

    if not os.path.isdir(skeleton_dir):
        print(f"错误：骨架图目录不存在 - {skeleton_dir}")
        return
    png_files = list_images(skeleton_dir)
    if not png_files:
        print("骨架图目录下没有 PNG 文件")
        return

    print(f"找到 {len(png_files)} 个骨架图文件，开始处理...")
    summary = []

    for idx, skeleton_path in enumerate(png_files, 1):
        filename = skeleton_path.name
        print(f"[{idx}/{len(png_files)}] 处理: {filename}")

        try:
            skeleton_img = cv2.imread(str(skeleton_path), cv2.IMREAD_GRAYSCALE)
            if skeleton_img is None:
                raise ValueError(f"无法读取骨架图: {skeleton_path}")
            skeleton = (skeleton_img > 0).astype('uint8')

            # 1. 首次检测
            endpoints, intersections, _ = kp_detector.detect(skeleton)
            # 2. 去毛刺
            skeleton, endpoints = post_processor.remove_spurs(skeleton, endpoints, intersections)
            # 3. 重新检测
            endpoints, intersections, _ = kp_detector.detect(skeleton)
            # 4. 合并交叉点
            skeleton, intersections = post_processor.merge_intersections(skeleton, intersections)
            # 5. 结构验证
            skeleton, endpoints, intersections = post_processor.structural_validation(
                skeleton, endpoints, intersections
            )

            # ----- 重要：坐标转换 -----
            # 将 (row, col) 转为 (col, row)，与 snap_to_skeleton 的预期一致
            endpoints = [(x, y) for x, y in endpoints]
            intersections = [(x, y) for x, y in intersections]

            nodes = endpoints + intersections
            adj = build_adjacency(skeleton, nodes, max_distance=400)

            base_name = os.path.splitext(filename)[0]

            adj_path = os.path.join(adj_save_dir, f"{base_name}adj.npy")
            np.save(adj_path, adj)
            node_path = os.path.join(node_save_dir, f"{base_name}nodes.npy")
            np.save(node_path, np.array(nodes))


            junctions = [{'position': (x, y), 'branch_count': 3} for (x, y) in intersections]
            vis_path = os.path.join(vis_dir, f"{base_name}_visualization.png")
            visualizer.visualize_keypoints(
                skeleton_img=skeleton,
                endpoints=endpoints,
                junctions=junctions,
                save_path=vis_path
            )

            summary.append((filename, len(endpoints), len(intersections)))

        except Exception as e:
            print(f"    处理失败: {e}")
            summary.append((filename, -1, -1))

    print("处理完成：")


if __name__ == '__main__':
    main()