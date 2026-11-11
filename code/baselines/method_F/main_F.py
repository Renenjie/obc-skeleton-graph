import os
import sys
from pathlib import Path
from typing import List, Tuple, Set
import time

import cv2
import numpy as np

# --- 定位仓库 code/ 目录（含 paths.py），保证脚本可在任意工作目录下运行 ---
_CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").is_file())
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from paths import (DEFAULT_DATASET, RESULT_ROOT, dataset_paths,
                   ensure_dir, list_images, print_config)

from endpoint_branch_detector import EndpointBranchDetector
from pruner import Pruner
from corner_detector import CornerDetector
from visualizer import SkeletonVisualizer

"""Graph-matching-based character recognition for
Chinese seal images"""


# ---------- 路径（相对仓库根目录，见 code/paths.py）----------
# 数据集设置：只改下一行（相对 <repo>/data，无需绝对路径）；DEFAULT_DATASET = 跟随全局
DATASET = DEFAULT_DATASET              # 例："skeleton_756"
input_dir, _, DATASET_TAG = dataset_paths(DATASET)          # <repo>/data/skeleton_<N>
output_dir = RESULT_ROOT / "baseline" / f"F_matrix_{DATASET_TAG}"   # adj 与 nodes 保存到同一目录
vis_output_dir = RESULT_ROOT / "baseline" / f"F_matrix_{DATASET_TAG}_visualization"


def build_initial_edges(skeleton: np.ndarray, nodes: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    根据骨架图和关键点列表，构建初始边（连接关系）。
    对每个关键点，沿骨架向每个方向追踪，直到遇到另一个关键点或死胡同。
    使用 visited 集合防止环形路径导致无限循环。
    """
    h, w = skeleton.shape
    node_set = set(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    def get_neighbors(x, y):
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx] == 1:
                    neighbors.append((nx, ny))
        return neighbors

    edges = set()
    for idx, (sx, sy) in enumerate(nodes):
        neighbors = get_neighbors(sx, sy)
        for (nx, ny) in neighbors:
            if (nx, ny) in node_set:
                end_idx = node_to_idx[(nx, ny)]
                if idx != end_idx:
                    edges.add((min(idx, end_idx), max(idx, end_idx)))
                continue

            prev = (sx, sy)
            curr = (nx, ny)
            visited = {(sx, sy), curr}
            while True:
                curr_neighbors = get_neighbors(curr[0], curr[1])
                next_point = None
                for nbr in curr_neighbors:
                    if nbr != prev and nbr not in visited:
                        next_point = nbr
                        break
                if next_point is None:
                    break
                prev, curr = curr, next_point
                visited.add(curr)
                if curr in node_set and curr != (sx, sy):
                    end_idx = node_to_idx[curr]
                    edges.add((min(idx, end_idx), max(idx, end_idx)))
                    break
    return list(edges)


def load_skeleton_image(image_path: str) -> np.ndarray:
    """
    读取图像并转换为二值骨架图，像素值为 0 或 1。
    严格按照论文 Algorithm 1 第 1 步：缩放到 100x100 固定大小。
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 关键步骤：缩放到 100x100（使用 INTER_NEAREST 保留二值边缘，避免插值产生中间值）
    img_resized = cv2.resize(img, (100, 100), interpolation=cv2.INTER_NEAREST)
    
    # 二值化：大于阈值 127 设为 1，否则为 0
    _, binary = cv2.threshold(img_resized, 127, 1, cv2.THRESH_BINARY)
    return binary.astype(np.uint8)

def main():
    print_config()
    ensure_dir(output_dir)
    ensure_dir(vis_output_dir)

    MIN_BRANCH_LENGTH = 32    # 去除短枝
    D_MAX = 12.0              # 距离容忍度
    THETA_MIN = 1.2          # 角度阈值
    
    detector = EndpointBranchDetector()
    pruner = Pruner()
    corner_detector = CornerDetector()
    visualizer = SkeletonVisualizer(save_visualizations=True)

    image_files = list_images(input_dir)
    if not image_files:
        print(f"错误：输入目录中没有找到图像文件 - {input_dir}")
        return

    print(f"共找到 {len(image_files)} 张图像，开始处理...")
    total_start = time.time()

    # ---------- 新增：统计节点数 ----------
    node_counts = []
    # ------------------------------------

    for idx, img_path in enumerate(image_files, 1):
        try:
            filename = img_path.name
            base_name = os.path.splitext(filename)[0]

            skeleton = load_skeleton_image(str(img_path))

            endpoints, branches = detector.detect(skeleton)
            pruned_skeleton = pruner.prune(skeleton, endpoints, branches, min_length=MIN_BRANCH_LENGTH)
            new_endpoints, new_branches = detector.detect(pruned_skeleton)

            initial_nodes = new_endpoints + new_branches
            if len(initial_nodes) < 2:
                print(f"[{idx}/{len(image_files)}] {filename}: 节点数少于2，跳过")
                adj_matrix = np.zeros((len(initial_nodes), len(initial_nodes)), dtype=np.int8)
                np.save(os.path.join(output_dir, f"{base_name}adj.npy"), adj_matrix)
                np.save(os.path.join(output_dir, f"{base_name}nodes.npy"), np.array(initial_nodes, dtype=np.int32))
                visualizer.visualize(pruned_skeleton,
                                     endpoints=new_endpoints,
                                     branch_points=new_branches,
                                     save_path=os.path.join(vis_output_dir, f"{base_name}.png"),
                                     show=False,
                                     draw_edges=False)
                # 跳过时也记录（可选），此处记录为 0 以免影响平均
                node_counts.append(0)
                continue

            initial_edges = build_initial_edges(pruned_skeleton, initial_nodes)
            final_nodes, final_edges = corner_detector.detect(
                pruned_skeleton, initial_nodes, initial_edges,
                d_max=D_MAX, theta_min=THETA_MIN
            )

            num_nodes = len(final_nodes)
            # ---------- 记录节点数 ----------
            node_counts.append(num_nodes)
            # --------------------------------

            adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.int8)
            for (i, j) in final_edges:
                if i < num_nodes and j < num_nodes:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1

            np.save(os.path.join(output_dir, f"{base_name}adj.npy"), adj_matrix)
            nodes_array = np.array(final_nodes, dtype=np.int32)
            np.save(os.path.join(output_dir, f"{base_name}nodes.npy"), nodes_array)

            endpoints_final = []
            branches_final = []
            corners_final = []
            for (x, y) in final_nodes:
                conn = visualizer.compute_connectivity(pruned_skeleton, x, y)
                if conn == 1:
                    endpoints_final.append((x, y))
                elif conn == 2:
                    corners_final.append((x, y))
                else:
                    branches_final.append((x, y))

            edge_paths = [[final_nodes[i], final_nodes[j]] for (i, j) in final_edges
                          if i < len(final_nodes) and j < len(final_nodes)]

            visualizer.visualize(
                pruned_skeleton,
                endpoints=endpoints_final,
                branch_points=branches_final,
                corners=corners_final,
                edges=edge_paths,
                save_path=os.path.join(vis_output_dir, f"{base_name}.png"),
                show=False,
                draw_edges=True,
                point_radius=3,
                corner_radius=4
            )

            print(f"[{idx}/{len(image_files)}] {filename}: 处理完成，节点数={num_nodes}")

        except Exception as e:
            print(f"[{idx}/{len(image_files)}] {filename}: 处理失败，错误: {e}")
            continue

    total_end = time.time()
    # ---------- 输出平均节点数 ----------
    if node_counts:
        valid_counts = [c for c in node_counts if c > 0]  # 排除跳过的0节点样本
        avg_nodes = np.mean(valid_counts) if valid_counts else 0
        print(f"平均节点数（成功处理样本）: {avg_nodes:.2f}  (总处理样本数: {len(node_counts)})")
    # ------------------------------------
    print(f"全部处理完成！总耗时: {total_end - total_start:.2f} 秒")

if __name__ == "__main__":
    main()