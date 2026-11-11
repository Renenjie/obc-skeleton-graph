import numpy as np
from typing import List, Tuple

class KeypointDetector:
    """
    基于骨架图像（单像素宽度，前景=1，背景=0）检测汉字关键点。
    严格按照论文定义：M 值为八邻域序列中像素值发生变化的次数（0↔1）。
    """
    def __init__(self, skeleton: np.ndarray):
        self.skeleton = skeleton.astype(np.uint8)
        self.h, self.w = skeleton.shape

    def detect_all(self) -> List[Tuple[int, int, str]]:
        keypoints = []
        for y in range(self.h):
            for x in range(self.w):
                if self.skeleton[y, x] == 0:
                    continue
                N = self._count_foreground_neighbors(x, y)
                M = self._count_changes(x, y)   # 变化次数
                ktype = self._classify(N, M)
                if ktype:
                    keypoints.append((x, y, ktype))
        return keypoints

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
            self._get_pixel(x, y-1),   # p2
            self._get_pixel(x+1, y-1), # p3
            self._get_pixel(x+1, y),   # p4
            self._get_pixel(x+1, y+1), # p5
            self._get_pixel(x, y+1),   # p6
            self._get_pixel(x-1, y+1), # p7
            self._get_pixel(x-1, y),   # p8
            self._get_pixel(x-1, y-1)  # p9
        ]
        extended = neigh + [neigh[0]]
        changes = 0
        for i in range(8):
            if extended[i] != extended[i+1]:
                changes += 1
        return changes

    def _get_pixel(self, x: int, y: int) -> int:
        if 0 <= x < self.w and 0 <= y < self.h:
            return self.skeleton[y, x]
        return 0

    def _classify(self, N: int, M: int) -> str:
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