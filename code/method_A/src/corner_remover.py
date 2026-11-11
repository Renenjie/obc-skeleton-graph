# corner_remover.py
import math

class RedundantCornerRemover:
    """
    根据邻边夹角删除 type=2 的冗余拐点。
    """
    def remove(self, nodes, edges, angle_min=150, angle_max=180):
        coord_to_idx = {n[:2]: i for i, n in enumerate(nodes)}
        n = len(nodes)
        adj_list = {i: [] for i in range(n)}
        for p1, p2 in edges:
            if p1 in coord_to_idx and p2 in coord_to_idx:
                i, j = coord_to_idx[p1], coord_to_idx[p2]
                adj_list[i].append(j)
                adj_list[j].append(i)

        changed = True
        while changed:
            changed = False
            to_remove = None
            for i, nbs in adj_list.items():
                if nodes[i][2] != 2 or len(nbs) != 2:
                    continue
                a, b = nbs[0], nbs[1]
                xi, yi, _ = nodes[i]
                xa, ya, _ = nodes[a]
                xb, yb, _ = nodes[b]
                va = (xa - xi, ya - yi)
                vb = (xb - xi, yb - yi)
                len_a = math.hypot(*va)
                len_b = math.hypot(*vb)
                if len_a == 0 or len_b == 0:
                    continue
                dot = (va[0]*vb[0] + va[1]*vb[1]) / (len_a * len_b)
                dot = max(-1.0, min(1.0, dot))
                angle = math.degrees(math.acos(dot))
                if angle_min <= angle <= angle_max:
                    to_remove = i
                    break

            if to_remove is not None:
                a, b = adj_list[to_remove][0], adj_list[to_remove][1]
                coord_rem = nodes[to_remove][:2]
                coord_a = nodes[a][:2]
                coord_b = nodes[b][:2]
                new_edges = []
                for p1, p2 in edges:
                    if p1 == coord_rem or p2 == coord_rem:
                        continue
                    new_edges.append((p1, p2))
                new_edges.append((coord_a, coord_b))
                new_nodes = [node for idx, node in enumerate(nodes) if idx != to_remove]
                nodes, edges = new_nodes, new_edges
                n = len(nodes)
                coord_to_idx = {n[:2]: i for i, n in enumerate(nodes)}
                adj_list = {i: [] for i in range(n)}
                for p1, p2 in edges:
                    if p1 in coord_to_idx and p2 in coord_to_idx:
                        i, j = coord_to_idx[p1], coord_to_idx[p2]
                        adj_list[i].append(j)
                        adj_list[j].append(i)
                changed = True
        return nodes, edges