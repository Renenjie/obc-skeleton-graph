import numpy as np
from typing import List, Tuple, Set

class GraphBuilder:
    def __init__(self, skeleton: np.ndarray, keypoints: List[Tuple[int, int, str]],
                 stroke_len_thresh: int = 3):
        self.skeleton = skeleton.astype(np.uint8)
        self.keypoints = keypoints          # 保留原始带类型的关键点
        self.h, self.w = skeleton.shape
        self.kp_set = {(x, y) for (x, y, _) in keypoints}
        self.thresh = stroke_len_thresh
        self.edges = []
        self.mask = skeleton.copy()

    def build(self):
        self._scan_horizontal()

        self._scan_vertical()

        self._scan_diagonal_45()

        self._scan_diagonal_135()

        return self.keypoints, self.edges

    # ---------- 辅助函数 ----------
    def _is_connected_and_length(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[bool, int]:
        """检查直线路径上是否全为前景（且未被掩码），返回(是否连通, 步数)"""
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return False, 0
        x, y = x1, y1
        sx = 1 if dx > 0 else -1 if dx < 0 else 0
        sy = 1 if dy > 0 else -1 if dy < 0 else 0
        err = abs(dx) - abs(dy)
        while True:
            if (x, y) != p1 and (x, y) != p2:
                if not (0 <= x < self.w and 0 <= y < self.h and self.mask[y, x] == 1):
                    return False, 0
            if x == x2 and y == y2:
                break
            e2 = 2 * err
            if e2 > -abs(dy):
                err -= abs(dy)
                x += sx
            if e2 < abs(dx):
                err += abs(dx)
                y += sy
        return True, steps
    def _trace_diagonal(self, start_x: int, start_y: int, direction: str):
        """
        从 (start_x, start_y) 沿斜向方向追踪，直到无法移动。
        返回：
            keypoint_seq: 按顺序经过的关键点坐标列表（包括起点和终点）
            path_pixels:  所有经过的像素坐标列表（用于掩码）
        direction: '45' 表示左下方向（优先级：左下>正下>正左）
                   '135' 表示右下方向（优先级：右下>正下>正右）
        """
        keypoint_seq = []
        path_pixels = []
        x, y = start_x, start_y
        while True:
            path_pixels.append((x, y))
            if (x, y) in self.kp_set:
                keypoint_seq.append((x, y))
            # 根据方向确定下一步优先级
            if direction == '45':
                moves = [(-1, 1), (0, 1), (-1, 0)]   # 左下, 正下, 正左
            else:  # '135'
                moves = [(1, 1), (0, 1), (1, 0)]     # 右下, 正下, 正右
            moved = False
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.w and 0 <= ny < self.h and self.mask[ny, nx] == 1:
                    x, y = nx, ny
                    moved = True
                    break
            if not moved:
                break
        return keypoint_seq, path_pixels
    def _mark_path(self, p1: Tuple[int, int], p2: Tuple[int, int]):
        """将两点间路径上的像素（不含端点）掩码置0"""
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return
        x, y = x1, y1
        sx = 1 if dx > 0 else -1 if dx < 0 else 0
        sy = 1 if dy > 0 else -1 if dy < 0 else 0
        err = abs(dx) - abs(dy)
        while True:
            if (x, y) != p1 and (x, y) != p2:
                if 0 <= x < self.w and 0 <= y < self.h:
                    self.mask[y, x] = 0
            if x == x2 and y == y2:
                break
            e2 = 2 * err
            if e2 > -abs(dy):
                err -= abs(dy)
                x += sx
            if e2 < abs(dx):
                err += abs(dx)
                y += sy

    def _add_edge(self, start: Tuple[int, int], curr: Tuple[int, int]):
        """尝试连接并标记，若长度≥阈值则添加边"""
        connected, length = self._is_connected_and_length(start, curr)
        if connected and length >= self.thresh:
            self.edges.append((start, curr, float(length)))
            self._mark_path(start, curr)

    # ---------- 0° 横向扫描 ----------
    def _scan_horizontal(self):
        for y in range(self.h):
            start_kp = None
            for x in range(self.w):
                if (x, y) in self.kp_set and self.mask[y, x] == 1:
                    if start_kp is None:
                        start_kp = (x, y)
                    else:
                        self._add_edge(start_kp, (x, y))
                        start_kp = (x, y)

    # ---------- 90° 纵向扫描 ----------
    def _scan_vertical(self):
        for x in range(self.w):
            start_kp = None
            for y in range(self.h):
                if (x, y) in self.kp_set and self.mask[y, x] == 1:
                    if start_kp is None:
                        start_kp = (x, y)
                    else:
                        self._add_edge(start_kp, (x, y))
                        start_kp = (x, y)

    # ---------- 45° 左斜下扫描（优先级：左下 > 正下 > 正左）----------
    def _scan_diagonal_45(self):
        # 遍历所有关键点（按列表顺序，但使用掩码控制是否处理）
        for (x, y, _) in self.keypoints:
            if self.mask[y, x] == 0:
                continue   # 已被掩码，跳过
            # 从该点沿左下方向追踪
            seq, path = self._trace_diagonal(x, y, '45')
            if len(seq) < 2:
                continue
            # 依次连接相邻的关键点
            for i in range(len(seq) - 1):
                p1 = seq[i]
                p2 = seq[i+1]
                # 直接添加边，不用再检查连通性（已经追踪确认）
                # 但需检查长度阈值
                dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
                if dist >= self.thresh:
                    self.edges.append((p1, p2, dist))
            # 掩码整个路径（包括端点），防止重复处理
            for px, py in path:
                self.mask[py, px] = 0
    
        # ---------- 135° 右斜下扫描（优先级：右下 > 正下 > 正右）----------
    def _scan_diagonal_135(self):
        for (x, y, _) in self.keypoints:
            if self.mask[y, x] == 0:
                continue
            seq, path = self._trace_diagonal(x, y, '135')
            if len(seq) < 2:
                continue
            for i in range(len(seq) - 1):
                p1 = seq[i]
                p2 = seq[i+1]
                dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
                if dist >= self.thresh:
                    self.edges.append((p1, p2, dist))
            for px, py in path:
                self.mask[py, px] = 0