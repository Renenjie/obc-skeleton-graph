"""
关键点检测模块
实现论文 Algorithm 1：基于 Moore 邻域变化量的端点与交叉点检测
"""

from typing import List, Tuple, Dict
import numpy as np


class KeypointDetector:
    """检测骨架上的端点和交叉点"""
    def _moore_neighborhood_variation(self, skeleton: np.ndarray, x: int, y: int) -> int:
        """
        计算像素 (x,y) 的 Moore 邻域变化量 Ψ
        :param skeleton: 二值骨架 (0/1)
        :param x: 行坐标
        :param y: 列坐标
        :return: Ψ 值
        """
        # Moore 邻域顺时针顺序（从左上角开始）
        offsets = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
        values = []
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < skeleton.shape[0] and 0 <= ny < skeleton.shape[1]:
                values.append(int(skeleton[nx, ny]))   # 强制转为 int
            else:
                values.append(0)
        psi = 0
        for i in range(8):
            psi += abs(values[i] - values[(i + 1) % 8])
        return psi

    def detect(self, skeleton: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], Dict]:
        """
        检测骨架上的关键点
        :param skeleton: 二值骨架 (0/1)
        :return: (端点列表, 交叉点列表, 所有像素类型字典)
                像素类型字典键为 (x,y)，值为 'endpoint' / 'intersection' / 'ordinary'
        """
        endpoints = []
        intersections = []
        all_types = {}
        h, w = skeleton.shape
        for x in range(h):
            for y in range(w):
                if skeleton[x, y] == 0:
                    continue
                psi = self._moore_neighborhood_variation(skeleton, x, y)
                if psi == 2:
                    endpoints.append((x, y))
                    all_types[(x, y)] = 'endpoint'
                elif psi in (6, 8):
                    intersections.append((x, y))
                    all_types[(x, y)] = 'intersection'
                else:
                    all_types[(x, y)] = 'ordinary'
        return endpoints, intersections, all_types