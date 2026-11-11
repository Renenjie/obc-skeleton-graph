import cv2
import numpy as np

class CircularMidpointDetector:
    """基于 N,M 局部特征 + 路径追踪验证的中点检测器，使用切比雪夫距离去重。
    参数：
        min_distance : int, 拐点间最小切比雪夫距离，小于此距离的候选点将被合并。
        track_steps  : int, 沿每个分支追踪的步数，用于过滤不稳定拐点（默认 5）。
    """
    def __init__(self, min_distance=5, track_steps=5, **kwargs):
        self.min_distance = min_distance
        self.track_steps = track_steps

        # 8邻域方向（用于追踪，与 neighbours 的顺序对齐）
        # 这里的方向向量索引与环上邻居的顺序一致：上、右上、右、右下、下、左下、左、左上
        self.directions = [(0, -1), (1, -1), (1, 0), (1, 1),
                           (0, 1), (-1, 1), (-1, 0), (-1, -1)]

    def _is_foreground(self, skeleton, x, y):
        h, w = skeleton.shape
        if 0 <= x < h and 0 <= y < w:
            return skeleton[x, y] > 0
        return False

    def _track_branch(self, skeleton, start_x, start_y, dir_idx):
        """沿给定方向追踪 self.track_steps 步，成功返回 True，否则 False。"""
        x, y = start_x, start_y
        last_x, last_y = x, y
        dx, dy = self.directions[dir_idx]

        # 第一步
        nx, ny = x + dx, y + dy
        if not self._is_foreground(skeleton, nx, ny):
            return False
        x, y = nx, ny

        steps_done = 1
        while steps_done < self.track_steps:
            found = False
            # 优先直行，其次左右微调，避免回头
            for offset in [0, -1, 1]:
                ndir = (dir_idx + offset) % 8
                ndx, ndy = self.directions[ndir]
                nx, ny = x + ndx, y + ndy
                if self._is_foreground(skeleton, nx, ny):
                    if (nx, ny) == (last_x, last_y):
                        continue
                    last_x, last_y = x, y
                    x, y = nx, ny
                    dir_idx = ndir
                    dx, dy = ndx, ndy
                    found = True
                    steps_done += 1
                    break
            if not found:
                return False
        return True

    def find_circular_midpoints(self, skeleton_img, endpoints=None, junctions=None):
        if len(skeleton_img.shape) > 2:
            skeleton_img = cv2.cvtColor(skeleton_img, cv2.COLOR_BGR2GRAY)
        binary = (skeleton_img > 0).astype(np.uint8)
        skeleton = binary

        h, w = skeleton.shape
        raw_midpoints = []

        # 顺时针八邻域（用于 N, M 计算）
        neighbours = [
            ( 0, -1),   # p2  上
            ( 1, -1),   # p3  右上
            ( 1,  0),   # p4  右
            ( 1,  1),   # p5  右下
            ( 0,  1),   # p6  下
            (-1,  1),   # p7  左下
            (-1,  0),   # p8  左
            (-1, -1)    # p9  左上
        ]

        for x in range(h):
            for y in range(w):
                if skeleton[x, y] == 0:
                    continue

                N = 0
                vals = []
                for dx, dy in neighbours:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and skeleton[nx, ny]:
                        N += 1
                        vals.append(1)
                    else:
                        vals.append(0)

                extended = vals + [vals[0]]
                M = sum(1 for i in range(8) if extended[i] != extended[i+1])

                if N == 2 and M == 4:
                    # 找到两个前景邻居在环上的索引（0‑7）
                    neighbor_indices = [i for i, v in enumerate(vals) if v == 1]
                    idx1, idx2 = neighbor_indices[0], neighbor_indices[1]

                    # 分别沿两个方向追踪
                    if self._track_branch(skeleton, x, y, idx1) and \
                       self._track_branch(skeleton, x, y, idx2):
                        raw_midpoints.append((x, y))

        if not raw_midpoints:
            return []

        # 去重：切比雪夫距离抑制
        raw_midpoints.sort()
        filtered = []
        for pt in raw_midpoints:
            keep = True
            for fp in filtered:
                if max(abs(pt[0] - fp[0]), abs(pt[1] - fp[1])) <= self.min_distance:
                    keep = False
                    break
            if keep:
                filtered.append(pt)

        return filtered