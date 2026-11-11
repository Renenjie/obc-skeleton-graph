import numpy as np
import math
from typing import List, Tuple

class GraphOptimizer:
    """
    图优化器：
    1. 滑动窗口合并邻近关键点（自下向上）
    2. 删除度数为2且夹角在 [angle_low, angle_high] 或 [straight_angle, 180] 的冗余节点
    3. 每次节点集变化后，基于原始骨架重新计算所有关键点的类型
    """
    def __init__(self, 
                 keypoints: List[Tuple[int, int, str]], 
                 edges: List[Tuple[Tuple[int, int], Tuple[int, int], float]],
                 skeleton: np.ndarray,
                 stroke_len_thresh: int = 3,      # 合并窗口（切比雪夫距离）
                 angle_low: float = 0.0,
                 angle_high: float = 10.0,
                 remove_straight: bool = True,
                 straight_angle: float = 170.0):
        self.keypoints = keypoints
        self.edges = edges
        self.skeleton = skeleton.astype(np.uint8)
        self.thresh = stroke_len_thresh
        self.h, self.w = skeleton.shape
        self.angle_low = angle_low
        self.angle_high = angle_high
        self.remove_straight = remove_straight
        self.straight_angle = straight_angle

    def optimize(self):
        self._merge_nearby_keypoints()
        # 循环删除冗余节点，直到稳定
        prev_count = 0
        while len(self.keypoints) != prev_count:
            prev_count = len(self.keypoints)
            self._remove_redundant_degree2()
        return self.keypoints, self.edges

    # --------------------------------------------------------------------------
    # 辅助：基于当前位置在骨架上重新计算关键点类型
    # --------------------------------------------------------------------------
    def _update_keypoint_types(self):
        for i, (x, y, _) in enumerate(self.keypoints):
            if self.skeleton[y, x] == 0:
                continue
            N = self._count_foreground_neighbors(x, y)
            M = self._count_changes(x, y)
            new_type = self._classify(N, M)
            self.keypoints[i] = (x, y, new_type)

    def _count_foreground_neighbors(self, x: int, y: int) -> int:
        count = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.w and 0 <= ny < self.h:
                    if self.skeleton[ny, nx] == 1:
                        count += 1
        return count

    def _count_changes(self, x: int, y: int) -> int:
        neigh = [
            self._get_pixel(x, y-1),
            self._get_pixel(x+1, y-1),
            self._get_pixel(x+1, y),
            self._get_pixel(x+1, y+1),
            self._get_pixel(x, y+1),
            self._get_pixel(x-1, y+1),
            self._get_pixel(x-1, y),
            self._get_pixel(x-1, y-1)
        ]
        extended = neigh + [neigh[0]]
        changes = 0
        for i in range(8):
            if extended[i] == 0 and extended[i+1] == 1:
                changes += 1
        return changes

    def _get_pixel(self, x: int, y: int) -> int:
        if 0 <= x < self.w and 0 <= y < self.h:
            return self.skeleton[y, x]
        return 0

    @staticmethod
    def _classify(N: int, M: int) -> str:
        if N == 1 and M == 2:
            return 'endpoint'
        if N == 2 and M == 4:
            return 'corner'
        if N == 3 and M == 6:
            return 'trifurcation'
        if N == 4 and M == 8:
            return 'quadfurcation'
        if N == 5 and M == 6:
            return 'pentafurcation'
        if N == 6 and M == 4:
            return 'hexafurcation'
        if N == 7 and M == 2:
            return 'heptafurcation'
        if N == 8 and M == 4:
            return 'octafurcation'
        return ''

    # --------------------------------------------------------------------------
    # 1. 滑动窗口合并邻近关键点
    # --------------------------------------------------------------------------
    def _merge_nearby_keypoints(self):
        if len(self.keypoints) < 2:
            return

        n = len(self.keypoints)
        to_remove = set()
        used = set()

        for i in range(n):
            if i in used:
                continue
            x1, y1, _ = self.keypoints[i]
            group = [i]
            for j in range(i + 1, n):
                if j in used:
                    continue
                x2, y2, _ = self.keypoints[j]
                if abs(x1 - x2) <= self.thresh and abs(y1 - y2) <= self.thresh:
                    group.append(j)
            if len(group) > 1:
                best_idx = max(group, key=lambda idx: (self.keypoints[idx][1], self.keypoints[idx][0]))
                for idx in group:
                    if idx != best_idx:
                        to_remove.add(idx)
                used.update(group)

        if not to_remove:
            return

        coord_map = {}
        for i in range(n):
            if i in to_remove:
                xi, yi, _ = self.keypoints[i]
                for j in range(n):
                    if j not in to_remove and abs(xi - self.keypoints[j][0]) <= self.thresh and abs(yi - self.keypoints[j][1]) <= self.thresh:
                        coord_map[(xi, yi)] = (self.keypoints[j][0], self.keypoints[j][1])
                        break

        new_keypoints = [self.keypoints[i] for i in range(n) if i not in to_remove]
        new_edges = []
        for (p1, p2, length) in self.edges:
            new_p1 = coord_map.get(p1, p1)
            new_p2 = coord_map.get(p2, p2)
            if new_p1 == new_p2:
                continue
            p1_exists = any(new_p1 == (x, y) for (x, y, _) in new_keypoints)
            p2_exists = any(new_p2 == (x, y) for (x, y, _) in new_keypoints)
            if not p1_exists or not p2_exists:
                continue
            new_edges.append((new_p1, new_p2, length))

        unique_edges = []
        for e in new_edges:
            rev = (e[1], e[0], e[2])
            if e not in unique_edges and rev not in unique_edges:
                unique_edges.append(e)

        self.keypoints = new_keypoints
        self.edges = unique_edges
        self._update_keypoint_types()

    # --------------------------------------------------------------------------
    # 2. 删除度数为 2 的冗余拐点（基于夹角）
    # --------------------------------------------------------------------------
    def _remove_redundant_degree2(self):
        coord_to_idx = {(x, y): i for i, (x, y, _) in enumerate(self.keypoints)}
        adj = {i: set() for i in range(len(self.keypoints))}
        for (p1, p2, _) in self.edges:
            if p1 in coord_to_idx and p2 in coord_to_idx:
                i1, i2 = coord_to_idx[p1], coord_to_idx[p2]
                adj[i1].add(i2)
                adj[i2].add(i1)

        changed = True
        while changed:
            changed = False
            to_remove = []
            for i, neighbors in adj.items():
                if len(neighbors) == 2:
                    a, b = list(neighbors)
                    angle = self._compute_angle(i, a, b)
                    if angle is None:
                        continue
                    # 严格按论文：删除 [10, 70] 或 [110, 180] 的节点
                    if (10.0 <= angle <= 70.0) or (110.0 <= angle <= 180.0):
                        to_remove.append(i)

            if not to_remove:
                break

            i = to_remove[0]
            neighbors = list(adj[i])
            if len(neighbors) != 2:
                continue
            a, b = neighbors[0], neighbors[1]

            coord_i = (self.keypoints[i][0], self.keypoints[i][1])
            coord_a = (self.keypoints[a][0], self.keypoints[a][1])
            coord_b = (self.keypoints[b][0], self.keypoints[b][1])

            new_edges = []
            for (p1, p2, length) in self.edges:
                if p1 == coord_i or p2 == coord_i:
                    continue
                new_edges.append((p1, p2, length))
            new_len = np.hypot(coord_a[0] - coord_b[0], coord_a[1] - coord_b[1])
            new_edges.append((coord_a, coord_b, new_len))

            new_keypoints = [kp for idx, kp in enumerate(self.keypoints) if idx != i]

            self.keypoints = new_keypoints
            self.edges = new_edges
            self._update_keypoint_types()

            coord_to_idx = {(x, y): idx for idx, (x, y, _) in enumerate(self.keypoints)}
            adj = {idx: set() for idx in range(len(self.keypoints))}
            for (p1, p2, _) in self.edges:
                if p1 in coord_to_idx and p2 in coord_to_idx:
                    i1, i2 = coord_to_idx[p1], coord_to_idx[p2]
                    adj[i1].add(i2)
                    adj[i2].add(i1)
            changed = True

    def _compute_angle(self, i: int, a: int, b: int) -> float:
        xi, yi, _ = self.keypoints[i]
        xa, ya, _ = self.keypoints[a]
        xb, yb, _ = self.keypoints[b]
        vec_a = (xa - xi, ya - yi)
        vec_b = (xb - xi, yb - yi)
        norm_a = math.hypot(vec_a[0], vec_a[1])
        norm_b = math.hypot(vec_b[0], vec_b[1])
        if norm_a == 0 or norm_b == 0:
            return None
        dot = vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1]
        cos = dot / (norm_a * norm_b)
        cos = max(-1.0, min(1.0, cos))
        angle_rad = math.acos(cos)
        return math.degrees(angle_rad)