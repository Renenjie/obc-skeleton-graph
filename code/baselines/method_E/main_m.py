import os
import sys
from pathlib import Path

import cv2
import numpy as np

# --- 定位仓库 code/ 目录（含 paths.py），保证脚本可在任意工作目录下运行 ---
_CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").is_file())
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from paths import (DEFAULT_DATASET, RESULT_ROOT, dataset_paths,
                   ensure_dir, list_images, print_config)

from Keypoint_Detector import KeypointDetector
from Subcomponent_Sampler import SubcomponentSampler
from Edge_Extractor import EdgeExtractor
from GraphVisualizer import GraphVisualizer

"""Graph-Based Keyword Spotting in Historical
Handwritten Documents"""

# ---------- 路径（相对仓库根目录，见 code/paths.py）----------
# 数据集设置：只改下一行（相对 <repo>/data，无需绝对路径）；DEFAULT_DATASET = 跟随全局
DATASET = DEFAULT_DATASET              # 例："skeleton_756"
input_dir, _, DATASET_TAG = dataset_paths(DATASET)         # <repo>/data/skeleton_<N>
adj_output_dir = RESULT_ROOT / "baseline" / f"E_matrix_{DATASET_TAG}"
vis_output_dir = RESULT_ROOT / "baseline" / f"E_matrix_{DATASET_TAG}_visualization"


def simplify_graph(nodes, edges, angle_threshold=160):
    """
    简化图：删除度数为2且共线的中间节点
    nodes: list of (x, y, type)
    edges: list of (u, v) 无向边，u<v
    angle_threshold: 夹角阈值（度），大于此值视为共线（直线上）
    return: (new_nodes, new_edges)
    """
    # 构建邻接表
    adj = [[] for _ in range(len(nodes))]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # 计算节点的度数
    degrees = [len(adj[i]) for i in range(len(nodes))]
    
    to_remove = set()
    # 需要重复扫描直到稳定
    changed = True
    while changed:
        changed = False
        # 建立当前邻接（因为节点可能被标记删除，但未实际移除）
        # 简化策略：只处理当前度数为2且未被标记的节点
        for i in range(len(nodes)):
            if i in to_remove:
                continue
            if degrees[i] == 2:
                # 获取两个邻居
                neighbors = [n for n in adj[i] if n not in to_remove]
                if len(neighbors) != 2:
                    continue
                a, b = neighbors[0], neighbors[1]
                # 计算向量
                xi, yi = nodes[i][0], nodes[i][1]
                xa, ya = nodes[a][0], nodes[a][1]
                xb, yb = nodes[b][0], nodes[b][1]
                v1 = (xa - xi, ya - yi)
                v2 = (xb - xi, yb - yi)
                # 计算夹角（度）
                dot = v1[0]*v2[0] + v1[1]*v2[1]
                norm1 = np.hypot(v1[0], v1[1])
                norm2 = np.hypot(v2[0], v2[1])
                if norm1 == 0 or norm2 == 0:
                    continue
                cos_angle = dot / (norm1 * norm2)
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                if angle > angle_threshold:  # 接近180° 共线
                    # 删除节点i，添加边(a,b)
                    to_remove.add(i)
                    # 更新度数
                    degrees[i] = 0
                    degrees[a] -= 1
                    degrees[b] -= 1
                    # 删除旧边
                    adj[a].remove(i)
                    adj[b].remove(i)
                    # 添加新边（如果尚未存在）
                    if a != b and b not in adj[a]:
                        adj[a].append(b)
                        adj[b].append(a)
                        degrees[a] += 1
                        degrees[b] += 1
                    changed = True
                    break  # 重新扫描
    # 构建新节点列表和边列表
    new_nodes = [node for idx, node in enumerate(nodes) if idx not in to_remove]
    # 重新编号
    old_to_new = {}
    new_idx = 0
    for idx in range(len(nodes)):
        if idx not in to_remove:
            old_to_new[idx] = new_idx
            new_idx += 1
    new_edges = []
    for u, v in edges:
        if u in to_remove or v in to_remove:
            continue
        new_edges.append((old_to_new[u], old_to_new[v]))
    # 去重并排序
    new_edges = list(set((min(a,b), max(a,b)) for a,b in new_edges))
    return new_nodes, new_edges

def main():
    # 输入输出目录配置


    # ========= 参数调整 =========
    sample_distance = 7      # 等距采样间隔（原2→4，减少采样点）
    merge_distance = 13       # 邻近关键点合并距离（原1→2）
    enable_simplify = True   # 是否启用后处理简化
    simplify_angle = 130     # 共线夹角阈值（大于此值删除中间点）
    save_visualizations = True
    # ===========================

    print_config()
    ensure_dir(adj_output_dir)
    ensure_dir(vis_output_dir)

    detector = KeypointDetector(merge_distance=merge_distance)
    sampler = SubcomponentSampler(sample_distance=sample_distance)
    extractor = EdgeExtractor()
    visualizer = GraphVisualizer(save_visualizations=save_visualizations)

    image_files = list_images(input_dir)
    if not image_files:
        print(f"错误：输入目录中没有找到图像文件 - {input_dir}")
        return
    print(f"共找到 {len(image_files)} 张图像，开始处理...")

    for img_path in image_files:
        filename = img_path.name
        print(f"处理: {img_path}")

        skeleton = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if skeleton is None:
            print("  无法读取，跳过")
            continue

        _, skeleton_bin = cv2.threshold(skeleton, 127, 1, cv2.THRESH_BINARY)

        endpoints, junctions = detector.detect(skeleton_bin)
        all_nodes = sampler.sample(skeleton_bin, endpoints, junctions)

        # 构建带类型节点列表
        endpoint_set = set(endpoints)
        junction_set = set(junctions)
        node_list = []
        for pt in all_nodes:
            if pt in endpoint_set:
                node_list.append((pt[0], pt[1], 0))
            elif pt in junction_set:
                node_list.append((pt[0], pt[1], 1))
            else:
                node_list.append((pt[0], pt[1], 2))

        # 提取原始边
        edges = extractor.extract_edges(skeleton_bin, all_nodes)

        # 后处理简化（可选项）
        if enable_simplify and len(node_list) > 0:
            node_list, edges = simplify_graph(node_list, edges, angle_threshold=simplify_angle)

        # 构建邻接矩阵
        n = len(node_list)
        adj_matrix = np.zeros((n, n), dtype=np.int32)
        for u, v in edges:
            adj_matrix[u, v] = 1
            adj_matrix[v, u] = 1

        base = os.path.splitext(filename)[0]
        np.save(os.path.join(adj_output_dir, f"{base}nodes.npy"), np.array(node_list, dtype=np.int32))
        np.save(os.path.join(adj_output_dir, f"{base}adj.npy"), adj_matrix)

        if save_visualizations:
            # 提取各类点用于可视化
            vis_ep = [(x,y) for (x,y,t) in node_list if t==0]
            vis_jn = [(x,y) for (x,y,t) in node_list if t==1]
            vis_mp = [(x,y) for (x,y,t) in node_list if t==2]
            skeleton_vis = skeleton_bin.copy() * 255
            vis_path = os.path.join(vis_output_dir, f"{base}_graph.png")
            visualizer.visualize_graph(
                skeleton_vis, vis_ep, vis_jn, vis_mp, edges,
                save_path=vis_path
            )

        print(f"  最终节点数: {n}, 边数: {len(edges)}")
    print("完成！")

if __name__ == "__main__":
    main()