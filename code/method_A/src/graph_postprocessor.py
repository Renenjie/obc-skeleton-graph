import math


class GraphPostProcessor:
    def __init__(self, merge_dist=5, type_priority=None, enable_merge=True, use_type_priority=True):
        """
        merge_dist : 合并窗口（切比雪夫距离）
        type_priority : 节点类型优先级，数值越小越优先；优先级 0 表示不参与合并
        enable_merge : 是否启用邻近节点合并
        use_type_priority : 是否在“多个可合并中点”之间使用类型优先级选择保留节点
        """
        self.merge_dist = merge_dist
        if type_priority is None:
            # 端点(0) 与交叉点(1) 优先级 0 —— 不参与合并；中点优先级 2
            self.type_priority = {0: 0, 1: 0, 2: 2, 3: 2}
        else:
            self.type_priority = type_priority
        self.enable_merge = enable_merge
        self.use_type_priority = use_type_priority

    def _is_protected(self, node_type):
        """优先级为 0 的类型（端点、交叉点）不参与合并"""
        return self.type_priority.get(node_type, 99) == 0

    # ---------- 合并邻近节点 ----------
    def _merge_close_nodes(self, nodes, edges):
        if len(nodes) < 2:
            return nodes, edges
    
        n = len(nodes)
        to_remove = set()
        used = set()
    
        for i in range(n):
            if i in used:
                continue
            x1, y1, t1 = nodes[i]
            group = [i]
            for j in range(i + 1, n):
                if j in used:
                    continue
                x2, y2, t2 = nodes[j]
                if max(abs(x1 - x2), abs(y1 - y2)) <= self.merge_dist:
                    group.append(j)
    
            if len(group) <= 1:
                continue
    
            used.update(group)
    
            endpoints = [idx for idx in group if nodes[idx][2] == 0]
            junctions = [idx for idx in group if nodes[idx][2] == 1]
            midpoints = [idx for idx in group if nodes[idx][2] in (2, 3)]
    
            # 1. 端点：全部保留，不参与合并
            # 2. 交叉点：多个时保留一个（按 type_priority + 索引），其余删除
            keep_junction = None
            if junctions:
                keep_junction = min(
                    junctions,
                    key=lambda idx: (self.type_priority.get(nodes[idx][2], 99), idx)
                )
                for idx in junctions:
                    if idx != keep_junction:
                        to_remove.add(idx)
    
            # 3. 中点：若组内有端点或交叉点，全部删除；否则保留一个
            if endpoints or keep_junction is not None:
                to_remove.update(midpoints)
            elif len(midpoints) > 1:
                keep_mid = min(
                    midpoints,
                    key=lambda idx: (self.type_priority.get(nodes[idx][2], 99), idx)
                )
                for idx in midpoints:
                    if idx != keep_mid:
                        to_remove.add(idx)
    
        if not to_remove:
            return nodes, edges
    
        # 边重映射：被移除节点 -> 最近的未移除节点
        coord_map = {}
        for idx in to_remove:
            xi, yi, _ = nodes[idx]
            best_dist = float('inf')
            best_coord = None
            for j in range(n):
                if j not in to_remove:
                    xj, yj, _ = nodes[j]
                    d = max(abs(xi - xj), abs(yi - yj))
                    if d < best_dist:
                        best_dist = d
                        best_coord = (xj, yj)
            if best_coord:
                coord_map[(xi, yi)] = best_coord
    
        new_nodes = [nodes[i] for i in range(n) if i not in to_remove]
        new_edges = []
        seen = set()
        for p1, p2 in edges:
            np1 = coord_map.get(p1, p1)
            np2 = coord_map.get(p2, p2)
            if np1 == np2:
                continue
            if (np1, np2) in seen or (np2, np1) in seen:
                continue
            seen.add((np1, np2))
            new_edges.append((np1, np2))
    
        return new_nodes, new_edges
    # ---------- 统一入口 ----------
    def process(self, nodes, edges, skeleton=None):
        if self.enable_merge:
            nodes, edges = self._merge_close_nodes(nodes, edges)
        return nodes, edges