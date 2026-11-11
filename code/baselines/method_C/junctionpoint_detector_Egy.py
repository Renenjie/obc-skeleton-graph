import cv2
import numpy as np
from itertools import combinations

class Junctionpoint_detector:
    """交叉点检测：基于形态学击中/击不中变换，并合并邻近簇"""

    def __init__(self, merge_distance: float = 5.0):
        """
        参数:
            merge_distance: 交叉点合并距离阈值（像素）。
                            若两个交叉点距离小于此值，则合并为一个。
        """
        self.merge_distance = merge_distance
        self._kernels = self._create_intersection_kernels()

    def _create_intersection_kernels(self):
        """生成 3‑邻居 和 4‑邻居 的精确击中/击不中核"""
        kernels = []
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),          (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]

        # 3 邻居（56个）
        for combo in combinations(range(8), 3):
            kernel = np.full((3, 3), -1, dtype=np.int8)
            kernel[1, 1] = 1
            for idx in combo:
                dr, dc = offsets[idx]
                kernel[1 + dr, 1 + dc] = 1
            kernels.append(kernel)

        # 4 邻居（70个）
        for combo in combinations(range(8), 4):
            kernel = np.full((3, 3), -1, dtype=np.int8)
            kernel[1, 1] = 1
            for idx in combo:
                dr, dc = offsets[idx]
                kernel[1 + dr, 1 + dc] = 1
            kernels.append(kernel)

        return kernels

    @staticmethod
    def _merge_points(points, distance_thresh):
        """将距离小于阈值的点合并为质心"""
        if not points:
            return []
        pts = np.array(points)
        merged = []
        used = np.zeros(len(pts), dtype=bool)

        for i in range(len(pts)):
            if used[i]:
                continue
            # 寻找所有距离当前点 < distance_thresh 的点
            dists = np.linalg.norm(pts - pts[i], axis=1)
            cluster_idx = np.where(dists < distance_thresh)[0]
            cluster_pts = pts[cluster_idx]
            centroid = np.mean(cluster_pts, axis=0)
            merged.append((int(round(centroid[0])), int(round(centroid[1]))))
            used[cluster_idx] = True

        return merged

    def detect(self, image_path: str):
        """检测交叉点，返回合并后的坐标列表 [(x, y), ...]"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # 击中/击不中匹配
        hitmiss_result = np.zeros_like(binary, dtype=np.uint8)
        for kernel in self._kernels:
            matched = cv2.morphologyEx(binary, cv2.MORPH_HITMISS, kernel)
            hitmiss_result = cv2.bitwise_or(hitmiss_result, matched)

        raw_points = [(x, y) for y, x in zip(*np.where(hitmiss_result > 0)[::-1])]

        # 合并邻近点
        merged_points = self._merge_points(raw_points, self.merge_distance)
        return merged_points