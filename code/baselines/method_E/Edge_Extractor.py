import numpy as np
import cv2
from collections import deque

class EdgeExtractor:
    """基于局部行走的快速边提取，直接连接相邻节点（路径上无其他节点）"""
    def __init__(self):
        pass

    def extract_edges(self, skeleton, nodes):
        """
        skeleton: 二值图，前景=1，背景=0
        nodes: list of (x,y)
        return: list of (i, j) 无向边
        """
        skel = (skeleton > 0).astype(np.uint8)
        h, w = skel.shape

        # 建立节点坐标到索引的映射
        node_index = {pt: idx for idx, pt in enumerate(nodes)}

        # 用于记录已经处理过的节点对（无向）
        edges_set = set()

        # 对每个节点，尝试从8个方向出发行走
        for idx, (sx, sy) in enumerate(nodes):
            # 找到起始点在骨架上的相邻像素（8邻域）
            neighbors = self._get_neighbors(skel, sx, sy)
            for nx, ny in neighbors:
                # 避免重复：只从较小索引节点出发的行走方向
                # 简单起见，允许双向，但用 set 去重
                self._walk_and_connect(
                    skel, (sx, sy), (nx, ny), node_index, idx, edges_set
                )
        # 转换为列表
        edges = [(u, v) for (u, v) in edges_set]
        return edges

    def _get_neighbors(self, skel, x, y):
        """返回 (x,y) 的8邻域中前景像素坐标列表"""
        h, w = skel.shape
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and skel[ny, nx] == 1:
                    neighbors.append((nx, ny))
        return neighbors

    def _walk_and_connect(self, skel, start_pt, first_step, node_index, start_idx, edges_set):
        """
        从 start_pt 开始，沿着 first_step 方向行走，直到遇到另一个节点或死胡同。
        如果遇到另一个节点且路径上无其他节点，则添加边 (start_idx, end_idx)。
        """
        h, w = skel.shape
        path_points = [start_pt, first_step]
        cur = first_step
        prev = start_pt

        while True:
            # 如果当前点是节点且不是起点
            if cur in node_index and cur != start_pt:
                # 检查路径中间是否有其他节点（path_points[1:-1] 中除了 cur 外）
                # 注意 path_points 包含起点、...、cur
                has_other_node = False
                for pt in path_points[1:-1]:  # 不包括起点和终点
                    if pt in node_index:
                        has_other_node = True
                        break
                if not has_other_node:
                    end_idx = node_index[cur]
                    if start_idx < end_idx:
                        edges_set.add((start_idx, end_idx))
                return

            # 获取当前点的未经过的邻居（不考虑前一个点）
            neighbors = self._get_neighbors(skel, cur[0], cur[1])
            # 排除前一点
            next_candidates = [p for p in neighbors if p != prev]
            if len(next_candidates) != 1:
                # 出现分支或死胡同，终止
                return
            nxt = next_candidates[0]

            # 防止循环（理论上不会出现，但安全）
            if nxt in path_points:
                return

            path_points.append(nxt)
            prev = cur
            cur = nxt