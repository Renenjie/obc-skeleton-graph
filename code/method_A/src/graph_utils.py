# graph_utils.py
import numpy as np
from collections import deque
from scipy.ndimage import distance_transform_edt

def snap_to_skeleton(skeleton_img, nodes):
    """
    将节点坐标吸附到最近骨架点，输入输出均为 (row, col) 格式。
    """
    skeleton = skeleton_img > 0
    if not np.any(skeleton):
        return nodes
    dist, indices = distance_transform_edt(~skeleton, return_indices=True)
    snapped = []
    for x, y in nodes:  # x=row, y=col
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < skeleton.shape[0] and 0 <= yi < skeleton.shape[1]:
            if skeleton[xi, yi]:
                snapped.append((x, y))
            else:
                # indices[0]: row, indices[1]: col
                near_row = indices[0, xi, yi]
                near_col = indices[1, xi, yi]
                snapped.append((near_row, near_col))
        else:
            snapped.append((x, y))
    return snapped

def build_graph_from_skeleton(skeleton_img, node_coords, min_edge_len=3, snap=False):
    """
    基于骨架路径建图
    """
    if snap:
        node_coords = snap_to_skeleton(skeleton_img, node_coords)
    snapped = node_coords
    node_set = set(snapped)
    n = len(node_coords)
    adj = np.zeros((n, n), dtype=int)
    coord_to_idx = {c: i for i, c in enumerate(snapped)}

    for idx, start in enumerate(snapped):
        if start not in coord_to_idx:
            continue
        visited = {start}
        q = deque([(start[0], start[1], 0)])
        while q:
            x, y, dist = q.popleft()
            for dx, dy in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in visited:
                    continue
                if 0 <= nx < skeleton_img.shape[0] and 0 <= ny < skeleton_img.shape[1] and skeleton_img[nx, ny]:
                    visited.add((nx, ny))
                    if (nx, ny) in node_set and (nx, ny) != start:
                        tgt_idx = coord_to_idx.get((nx, ny))
                        if tgt_idx is not None and dist + 1 >= min_edge_len:
                            adj[idx, tgt_idx] = 1
                            adj[tgt_idx, idx] = 1
                    else:
                        q.append((nx, ny, dist + 1))
    return adj