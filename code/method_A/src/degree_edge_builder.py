# degree_edge_builder.py
import numpy as np
import math
from collections import deque


class DegreeLimitedEdgeBuilder:
    def __init__(self, max_degree=None, junction_branch_map=None):
        if max_degree is None:
            max_degree = {0: 1, 1: 4, 2: 2, 3: 2}
        self.max_degree = max_degree
        self.junction_branch_map = junction_branch_map or {}
        self.directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                           (1, 0), (1, -1), (0, -1), (-1, -1)]

    def build(self, skeleton, nodes):
        coords = [(x, y) for (x, y, _) in nodes]
        n = len(nodes)
        coord_to_idx = {c: i for i, c in enumerate(coords)}
        adj = np.zeros((n, n), dtype=int)
        edges = []
        degree = [0] * n

        def limit_of(idx):
            node_type = nodes[idx][2]
            if node_type == 1:
                bc = self.junction_branch_map.get(coords[idx], self.max_degree[1])
                return max(3, min(bc, self.max_degree[1]))
            return self.max_degree[node_type]

        def is_foreground(x, y):
            h, w = skeleton.shape
            return 0 <= x < h and 0 <= y < w and skeleton[x, y] > 0

        def get_direct_neighbors(node_idx):
            start = coords[node_idx]
            visited = {start}
            q = deque()
            for dx, dy in self.directions:
                nx, ny = start[0] + dx, start[1] + dy
                if is_foreground(nx, ny):
                    if (nx, ny) in coord_to_idx and coord_to_idx[(nx, ny)] != node_idx:
                        yield coord_to_idx[(nx, ny)], 1
                    else:
                        visited.add((nx, ny))
                        q.append((nx, ny, dx, dy, 1))
            while q:
                x, y, ldx, ldy, steps = q.popleft()
                if (x, y) in coord_to_idx:
                    idx = coord_to_idx[(x, y)]
                    if idx != node_idx:
                        yield idx, steps
                    continue
                for dx, dy in self.directions:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in visited:
                        continue
                    if not is_foreground(nx, ny):
                        continue
                    if dx == -ldx and dy == -ldy:
                        continue
                    visited.add((nx, ny))
                    q.append((nx, ny, dx, dy, steps + 1))

        def is_endpoint_idx(idx):
            return nodes[idx][2] == 0

        def add_edge(i, j):
            edges.append((coords[i], coords[j]))
            adj[i, j] = adj[j, i] = 1
            degree[i] += 1
            degree[j] += 1

        # ==================== 阶段 1：主循环建边 ====================
        type_priority = {0: 0, 2: 1, 3: 1, 1: 2}
        priority_order = sorted(range(n), key=lambda i: type_priority.get(nodes[i][2], 99))

        for idx in priority_order:
            if degree[idx] >= limit_of(idx):
                continue
            neighbors = list(get_direct_neighbors(idx))
            if not neighbors:
                continue
            neighbors.sort(key=lambda x: (is_endpoint_idx(x[0]), degree[x[0]], x[1]))
            current_is_endpoint = (nodes[idx][2] == 0)
            for nbr_idx, dist in neighbors:
                if degree[idx] >= limit_of(idx):
                    break
                if adj[idx, nbr_idx] == 1:
                    continue
                if degree[idx] < limit_of(idx) and degree[nbr_idx] < limit_of(nbr_idx):
                    add_edge(idx, nbr_idx)
                elif current_is_endpoint and degree[idx] == 0 and nodes[nbr_idx][2] != 0:
                    if degree[nbr_idx] >= limit_of(nbr_idx):
                        continue
                    add_edge(idx, nbr_idx)
                    break

        # ==================== 阶段 2：补连 ====================
        for idx in range(n):
            node_type = nodes[idx][2]
            if node_type not in (1, 2, 3):
                continue   # 端点不补连（上限 1，主循环已强制处理）

            limit = limit_of(idx)
            if degree[idx] >= limit:
                continue

            # 下限：交叉点至少 3 条边，中点/环点至少补到上限
            if node_type == 1:
                min_required = min(3, limit)   # limit 本身 >= 3，所以 min_required = 3
            else:
                min_required = limit

            neighbors = list(get_direct_neighbors(idx))
            unconnected = [(nbr, d) for nbr, d in neighbors if adj[idx, nbr] == 0]
            if not unconnected:
                continue

            # 统一排序：端点优先，然后度数低的优先
            unconnected.sort(key=lambda x: (is_endpoint_idx(x[0]), degree[x[0]], x[1]))

            # ---------- 第一轮：优先连非交叉点（尊重上限） ----------
            if node_type == 1:
                for nbr_idx, _ in unconnected:
                    if degree[idx] >= limit:
                        break
                    if nodes[nbr_idx][2] == 1:
                        continue   # 第一轮不连交叉点
                    if degree[nbr_idx] >= limit_of(nbr_idx):
                        continue
                    add_edge(idx, nbr_idx)
            else:
                # 中点/环点：直接补到 limit
                for nbr_idx, _ in unconnected:
                    if degree[idx] >= limit:
                        break
                    if degree[nbr_idx] >= limit_of(nbr_idx):
                        continue
                    add_edge(idx, nbr_idx)

            # ---------- 第二轮：交叉点未达到下限 3，放宽连交叉点（尊重上限） ----------
            if node_type == 1 and degree[idx] < min_required:
                for nbr_idx, _ in unconnected:
                    if degree[idx] >= min_required:
                        break
                    if adj[idx, nbr_idx] == 1:
                        continue
                    if degree[nbr_idx] >= limit_of(nbr_idx):
                        continue
                    add_edge(idx, nbr_idx)

            # ---------- 第三轮：仍未达到下限 3，强制连非端点（忽略上限） ----------
            if node_type == 1 and degree[idx] < min_required:
                for nbr_idx, _ in unconnected:
                    if degree[idx] >= min_required:
                        break
                    if adj[idx, nbr_idx] == 1:
                        continue
                    if nodes[nbr_idx][2] == 0:
                        continue   # 端点不强制连，避免破坏端点上限
                    add_edge(idx, nbr_idx)

        return edges, adj