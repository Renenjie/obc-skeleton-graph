# global_midpoint_detector.py
import math
from collections import deque

class GlobalMidpointDetector:
    def __init__(self, min_chord_length=15, use_curvature=False, straightness_threshold=0.8, min_spacing=10):
        self.min_chord_length = min_chord_length
        self.use_curvature = use_curvature
        self.straightness_threshold = straightness_threshold
        self.min_spacing = min_spacing
        self.directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                           (1, 0), (1, -1), (0, -1), (-1, -1)]

    def _is_foreground(self, skeleton, x, y):
        h, w = skeleton.shape
        return 0 <= x < h and 0 <= y < w and skeleton[x, y] > 0

    def _extract_path(self, skeleton, start_xy, end_xy):
        visited = set()
        queue = deque()
        queue.append((start_xy[0], start_xy[1], [start_xy]))
        visited.add(start_xy)
        while queue:
            x, y, path = queue.popleft()
            if (x, y) == end_xy:
                return path
            for dx, dy in self.directions:
                nx, ny = x + dx, y + dy
                if self._is_foreground(skeleton, nx, ny) and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny, path + [(nx, ny)]))
        return []

    def _spatial_nms(self, midpoints):
        if not midpoints or self.min_spacing <= 0:
            return midpoints
        kept = []
        for pt in midpoints:
            x, y = pt[0], pt[1]
            if all(math.hypot(x - kx, y - ky) >= self.min_spacing for kx, ky, _ in kept):
                kept.append(pt)
        return kept

    def find_global_midpoints(self, skeleton, nodes, adj):
        midpoints = []
        n = len(nodes)
        visited_edges = set()
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i, j]:
                    continue
                if nodes[i][2] > 1 and nodes[j][2] > 1:
                    continue
                edge = (i, j) if i < j else (j, i)
                if edge in visited_edges:
                    continue
                visited_edges.add(edge)

                start = (nodes[i][0], nodes[i][1])
                end = (nodes[j][0], nodes[j][1])
                chord_len = math.hypot(end[0] - start[0], end[1] - start[1])
                if chord_len < self.min_chord_length:
                    continue

                path = self._extract_path(skeleton, start, end)
                if not path:
                    continue

                if self.use_curvature:
                    path_len = len(path)
                    if path_len < 2:
                        continue
                    if chord_len / path_len >= self.straightness_threshold:
                        continue

                mid_x = (start[0] + end[0]) / 2.0
                mid_y = (start[1] + end[1]) / 2.0
                best_pt = None
                min_dist = float('inf')
                for px, py in path:
                    dist = (px - mid_x) ** 2 + (py - mid_y) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        best_pt = (px, py)
                if best_pt is not None:
                    midpoints.append((best_pt[0], best_pt[1], 3))

        midpoints = self._spatial_nms(midpoints)
        return midpoints