import numpy as np
import math
from collections import deque

class DensityController:
    """
    合并距离过近的关键点，优先保留：
        - 端点 (type=0) > 交叉点 (type=1) > 拐点 (type=2)
    对于两个拐点，比较它们周围（沿骨架10像素内）的其他关键点数量，数量少的优先保留。
    距离阈值根据节点类型组合动态调整：
        - 相同类型：使用对应类型的系数（endpoint_factor, junction_factor, corner_factor）
        - 不同类型：使用两个系数的平均值
    """
    def __init__(self, merge_dist=5, trace_radius=10, use_chebyshev=False,
                 endpoint_factor=1.0, junction_factor=1.0, corner_factor=1.5):
        """
        merge_dist         : 基准合并距离阈值（像素）
        trace_radius       : 沿骨架追踪的半径（像素），用于统计拐点附近的关键点数量
        use_chebyshev      : 是否使用切比雪夫距离（max(|dx|,|dy|)），否则使用欧氏距离
        endpoint_factor    : 端点-端点合并距离的放大系数（默认1.0）
        junction_factor    : 交叉点-交叉点合并距离的放大系数（默认1.0）
        corner_factor      : 拐点-拐点合并距离的放大系数（默认1.5）
        """
        self.merge_dist = merge_dist
        self.trace_radius = trace_radius
        self.use_chebyshev = use_chebyshev
        self.endpoint_factor = endpoint_factor
        self.junction_factor = junction_factor
        self.corner_factor = corner_factor
        self.directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                           (1, 0), (1, -1), (0, -1), (-1, -1)]
        # 类型到系数的映射
        self.type_factor = {0: endpoint_factor, 1: junction_factor, 2: corner_factor}

    def _distance(self, p1, p2):
        if self.use_chebyshev:
            return max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
        else:
            return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    def _get_merge_threshold(self, node_a, node_b):
        """
        根据两个节点的类型计算实际的合并距离阈值。
        """
        typ_a = node_a[2]
        typ_b = node_b[2]
        if typ_a == typ_b:
            factor = self.type_factor[typ_a]
        else:
            # 不同类型，取平均系数
            factor = (self.type_factor[typ_a] + self.type_factor[typ_b]) / 2.0
        return self.merge_dist * factor

    def _count_keypoints_in_radius(self, skeleton, start_node, nodes, adj, radius=10):
        """
        沿骨架 BFS，统计以 start_node 为起点，路径距离 ≤ radius 像素范围内
        出现的其他关键点数量（不包括 start_node 自身）。
        返回: 关键点数量, 以及从 start_node 出发的不同骨架分支数（即度数）
        """
        h, w = skeleton.shape
        start_coord = nodes[start_node][:2]
        queue = deque()
        queue.append((start_coord[0], start_coord[1], 0.0, -1, -1))
        visited = {(start_coord[0], start_coord[1])}
        keypoint_count = 0
        reached_keypoints = set()
        start_neighbors = []

        while queue:
            r, c, dist, pr, pc = queue.popleft()
            if (r, c) != start_coord:
                for i, (x, y, typ) in enumerate(nodes):
                    if (x, y) == (r, c):
                        if i != start_node and i not in reached_keypoints:
                            reached_keypoints.add(i)
                            keypoint_count += 1
                        break
            for dr, dc in self.directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and skeleton[nr, nc] > 0:
                    if (nr, nc) == (pr, pc):
                        continue
                    if (nr, nc) in visited:
                        continue
                    ndist = dist + math.hypot(dr, dc)
                    if ndist > radius:
                        continue
                    visited.add((nr, nc))
                    queue.append((nr, nc, ndist, r, c))
                    if dist == 0:
                        start_neighbors.append((nr, nc))

        # 估算分支数
        if start_neighbors:
            dir_angles = []
            for nr, nc in start_neighbors:
                ang = math.degrees(math.atan2(nr - start_coord[0], nc - start_coord[1])) % 360
                dir_angles.append(ang)
            dir_angles.sort()
            merged = []
            for ang in dir_angles:
                if not merged or min(abs(ang - merged[-1]), 360 - abs(ang - merged[-1])) > 45:
                    merged.append(ang)
            branch_count = len(merged)
        else:
            branch_count = 0
        return keypoint_count, branch_count

    def _should_keep(self, node_a, node_b, skeleton, nodes, adj):
        typ_a = nodes[node_a][2]
        typ_b = nodes[node_b][2]
        if typ_a != typ_b:
            return node_a if typ_a < typ_b else node_b

        if typ_a == 2:   # 都是拐点
            cnt_a, _ = self._count_keypoints_in_radius(skeleton, node_a, nodes, adj, self.trace_radius)
            cnt_b, _ = self._count_keypoints_in_radius(skeleton, node_b, nodes, adj, self.trace_radius)
            return node_a if cnt_a <= cnt_b else node_b

        return node_a if node_a < node_b else node_b

    def merge_nodes(self, nodes, edges, skeleton):
        """
        合并距离过近的关键点，使用动态合并距离（根据节点类型）。
        输入：
            nodes   : list of (x, y, type)
            edges   : list of ((x1,y1), (x2,y2))
            skeleton: 二值骨架图 (0/1 或 0/255)
        返回：
            new_nodes, new_edges
        """
        if len(nodes) < 2:
            return nodes, edges

        coord_to_idx = {n[:2]: i for i, n in enumerate(nodes)}
        adj = {i: set() for i in range(len(nodes))}
        for p1, p2 in edges:
            if p1 in coord_to_idx and p2 in coord_to_idx:
                i1, i2 = coord_to_idx[p1], coord_to_idx[p2]
                adj[i1].add(i2)
                adj[i2].add(i1)

        changed = True
        while changed:
            changed = False
            n_nodes = len(nodes)
            to_remove = set()
            replace_map = {}
            for i in range(n_nodes):
                if i in to_remove:
                    continue
                xi, yi, _ = nodes[i]
                for j in range(i+1, n_nodes):
                    if j in to_remove:
                        continue
                    xj, yj, _ = nodes[j]
                    # 计算距离
                    dist = self._distance((xi, yi), (xj, yj))
                    # 动态阈值
                    thresh = self._get_merge_threshold(nodes[i], nodes[j])
                    if dist <= thresh:
                        keep = self._should_keep(i, j, skeleton, nodes, adj)
                        remove = j if keep == i else i
                        if remove not in to_remove:
                            to_remove.add(remove)
                            replace_map[nodes[remove][:2]] = nodes[keep][:2]
                            changed = True

            if not to_remove:
                break

            new_nodes = [n for idx, n in enumerate(nodes) if idx not in to_remove]
            new_edges = []
            seen = set()
            for p1, p2 in edges:
                np1 = replace_map.get(p1, p1)
                np2 = replace_map.get(p2, p2)
                if np1 == np2:
                    continue
                if (np1, np2) in seen or (np2, np1) in seen:
                    continue
                seen.add((np1, np2))
                new_edges.append((np1, np2))
            nodes = new_nodes
            edges = new_edges
            coord_to_idx = {n[:2]: i for i, n in enumerate(nodes)}
            adj = {i: set() for i in range(len(nodes))}
            for p1, p2 in edges:
                if p1 in coord_to_idx and p2 in coord_to_idx:
                    i1, i2 = coord_to_idx[p1], coord_to_idx[p2]
                    adj[i1].add(i2)
                    adj[i2].add(i1)

        return nodes, edges