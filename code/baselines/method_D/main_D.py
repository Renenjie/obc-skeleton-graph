import os
import sys
from pathlib import Path
from typing import List, Tuple
from collections import Counter

import cv2
import numpy as np

# --- 定位仓库 code/ 目录（含 paths.py），保证脚本可在任意工作目录下运行 ---
_CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").is_file())
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from paths import (DEFAULT_DATASET, RESULT_ROOT, dataset_paths,
                   ensure_dir, list_images, print_config)

from endpoint_junctionpoint_midpoint_detector_tang import KeypointDetector
from GraphBuilder_tang import GraphBuilder
from GraphOptimizer_tang import GraphOptimizer
from visualizer_tang import GraphVisualizer   # 使用新版本（支持 all_keypoints）


"""Graph Feature Extraction Method in Chinese Character Recognition"""

# ------------------- 配置路径（相对仓库根目录，见 code/paths.py）--------------------
# 数据集设置：只改下一行（相对 <repo>/data，无需绝对路径）；DEFAULT_DATASET = 跟随全局
DATASET = DEFAULT_DATASET              # 例："skeleton_756"
input_dir, _, DATASET_TAG = dataset_paths(DATASET)         # <repo>/data/skeleton_<N>
adj_output_dir = RESULT_ROOT / "baseline" / f"D_matrix_{DATASET_TAG}"
vis_output_dir = RESULT_ROOT / "baseline" / f"D_matrix_{DATASET_TAG}_visualization"


# ---------- 辅助函数 ----------
def build_adjacency_from_edges(node_coords: List[Tuple[int, int]], 
                               edges: List[Tuple[Tuple[int, int], Tuple[int, int], float]]) -> np.ndarray:
    n = len(node_coords)
    adj = np.zeros((n, n), dtype=np.uint8)
    coord_to_idx = {coord: i for i, coord in enumerate(node_coords)}
    for (p1, p2, _) in edges:
        if p1 in coord_to_idx and p2 in coord_to_idx:
            i, j = coord_to_idx[p1], coord_to_idx[p2]
            adj[i, j] = 1
            adj[j, i] = 1
    return adj

def load_skeleton(image_path: str) -> np.ndarray:
    """加载骨架图像，前景设为1，背景0"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return binary.astype(np.uint8)

def get_image_files(folder: str) -> List[str]:
    """返回文件夹下所有图像路径（按文件名排序，保证处理顺序可复现）。"""
    return [str(p) for p in list_images(folder)]


# ---------- 主处理函数 ----------
def process_single_image(skeleton_path, adj_out, vis_out, visualizer):
    """
    处理单张骨架图像，返回优化后的节点数量
    """
    base_name = os.path.splitext(os.path.basename(skeleton_path))[0]
    skeleton = load_skeleton(skeleton_path)
    h, w = skeleton.shape
    print(f"处理: {base_name}")

    build_thresh = 1          # 最短边长度
    merge_thresh = 6          # 关键点合并窗口

    # 1. 关键点检测
    detector = KeypointDetector(skeleton)
    keypoints = detector.detect_all()


    # 2. 图构建
    builder = GraphBuilder(skeleton, keypoints, stroke_len_thresh=build_thresh)
    keypoints, edges = builder.build()

    # 3. 图优化
    optimizer = GraphOptimizer(
        keypoints, edges, skeleton,
        stroke_len_thresh=merge_thresh,
        angle_low=0.0,
        angle_high=10.0,
        remove_straight=True,
        straight_angle=175.0
    )
    keypoints_opt, edges_opt = optimizer.optimize()


    # 4. 分类节点（用于可视化和保存）
    type_map = {
        'endpoint': 0, 'corner': 1, 'trifurcation': 2, 'quadfurcation': 3,
        'pentafurcation': 4, 'hexafurcation': 5, 'heptafurcation': 6, 'octafurcation': 7,
    }
    type_counts = Counter([typ for (_,_,typ) in keypoints_opt])


    node_coords = []
    node_types = []
    endpoints = []
    corners = []
    junctions = []
    for (x, y, typ) in keypoints_opt:
        node_coords.append((x, y))
        node_types.append(type_map.get(typ, -1))
        if typ == 'endpoint':
            endpoints.append((x, y))
        elif typ == 'corner':
            corners.append((x, y))
        elif typ in ('trifurcation', 'quadfurcation', 'pentafurcation',
                     'hexafurcation', 'heptafurcation', 'octafurcation'):
            branch = {'trifurcation':3, 'quadfurcation':4, 'pentafurcation':5,
                      'hexafurcation':6, 'heptafurcation':7, 'octafurcation':8}.get(typ, 3)
            junctions.append({'position': (x, y), 'branch_count': branch})

    # 5. 保存邻接矩阵和节点信息
    nodes_array = np.array([[x, y, t] for (x, y), t in zip(node_coords, node_types)], dtype=np.int32)
    adj_matrix = build_adjacency_from_edges(node_coords, edges_opt)
    ensure_dir(adj_out)
    np.save(os.path.join(adj_out, f"{base_name}nodes.npy"), nodes_array)
    np.save(os.path.join(adj_out, f"{base_name}adj.npy"), adj_matrix)

    # 6. 生成可视化路径（Bresenham 直线）
    paths = []
    for (p1, p2, _) in edges_opt:
        x1, y1 = p1
        x2, y2 = p2
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        x, y = x1, y1
        while True:
            points.append((x, y))
            if x == x2 and y == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        paths.append(points)

    # 7. 可视化并保存
    skeleton_vis = (skeleton * 255).astype(np.uint8)
    ensure_dir(vis_out)
    save_path = os.path.join(vis_out, f"{base_name}_vis.png")

    # 收集所有关键点坐标（无论类型），用于在图中以蓝色小点显示
    all_keypoints = [(x, y) for (x, y, _) in keypoints_opt]

    visualizer.visualize_keypoints(
        skeleton_img=skeleton_vis,
        endpoints=endpoints,
        junctions=junctions,
        corners=corners,
        all_keypoints=all_keypoints,   # 新增参数
        paths=paths,
        save_path=save_path
    )

    # 返回最终的节点数量
    return len(node_coords)


# ---------- 主入口 ----------
def main():
    print_config()
    visualizer = GraphVisualizer(save_visualizations=True)

    # 用于统计节点数量
    node_counts = []

    if os.path.isfile(input_dir):
        # 单张图片处理
        node_count = process_single_image(str(input_dir), str(adj_output_dir),
                                          str(vis_output_dir), visualizer)
        node_counts.append(node_count)
        print(f"节点数量: {node_count}")

    elif os.path.isdir(input_dir):
        image_files = get_image_files(input_dir)
        if not image_files:
            print(f"文件夹 {input_dir} 中没有找到图像文件。")
            return
        print(f"找到 {len(image_files)} 个图像文件，开始批量处理...")
        for img_path in image_files:
            try:
                node_count = process_single_image(img_path, str(adj_output_dir),
                                                  str(vis_output_dir), visualizer)
                node_counts.append(node_count)
                print(f"  -> 节点数量: {node_count}")
            except Exception as e:
                print(f"处理 {img_path} 时出错: {e}")

    else:
        print(f"输入路径不存在: {input_dir}")
        return

    # 输出平均节点数量
    if node_counts:
        avg_nodes = sum(node_counts) / len(node_counts)
        print(f"\n===== 处理完成，平均节点数量: {avg_nodes:.2f} (共处理 {len(node_counts)} 张图像) =====")
    else:
        print("\n没有成功处理任何图像，无法计算平均节点数量。")


if __name__ == "__main__":
    main()