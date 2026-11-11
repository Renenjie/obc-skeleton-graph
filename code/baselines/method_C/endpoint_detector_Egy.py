import cv2
import numpy as np

class EndpointDetector:
    """端点检测：基于形态学击中/击不中变换，支持毛刺剪除"""

    def __init__(self, prune_length=3):
        """
        参数:
            prune_length: 毛刺剪除长度阈值，≤0 则不剪除
        """
        self.prune_length = prune_length
        # 预生成 8 个方向的端点 Hit‑or‑Miss 核
        self.endpoint_kernels = self._generate_endpoint_kernels()

    def _generate_endpoint_kernels(self):
        """生成 8 个方向的端点检测核 (中心1，仅一个邻居为1，其余为背景 -1)"""
        kernels = []
        # 8 邻域方向 (行偏移, 列偏移)
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                      (1, 0), (1, -1), (0, -1), (-1, -1)]
        for dr, dc in directions:
            kernel = np.zeros((3, 3), dtype=np.int8)
            kernel[1, 1] = 1                # 中心必须是前景
            kernel[1 + dr, 1 + dc] = 1      # 唯一邻居必须是前景
            # 其余位置必须是背景（用 -1 表示）
            kernel[kernel == 0] = -1
            kernels.append(kernel)
        return kernels

    def _count_neighbors(self, binary, x, y):
        """统计 8 邻域前景像素个数（用于毛刺剪除中的分支点判定）"""
        h, w = binary.shape
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),          (0, 1),
                   (1, -1), (1, 0), (1, 1)]
        cnt = 0
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and binary[ny, nx] == 255:
                cnt += 1
        return cnt

    def _prune_spurs(self, binary, candidate_points):
        """毛刺剪除：从候选端点出发追踪，过滤短毛刺（原逻辑不变）"""
        if self.prune_length <= 0:
            return candidate_points

        # 标记分支点（邻居数 ≥ 3）
        ys, xs = np.where(binary == 255)
        branch_set = set()
        for x, y in zip(xs, ys):
            if self._count_neighbors(binary, x, y) >= 3:
                branch_set.add((x, y))

        kept = []
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),          (0, 1),
                   (1, -1), (1, 0), (1, 1)]

        for ex, ey in candidate_points:
            curr = (ex, ey)
            prev = None
            for _ in range(self.prune_length + 1):
                if curr in branch_set:       # 遇到主干交叉点 → 保留
                    kept.append((ex, ey))
                    break
                # 获取下一邻居（排除来路）
                neighbors = []
                for dx, dy in offsets:
                    nx, ny = curr[0] + dx, curr[1] + dy
                    if 0 <= nx < binary.shape[1] and 0 <= ny < binary.shape[0]:
                        if binary[ny, nx] == 255:
                            if prev is None or (nx, ny) != prev:
                                neighbors.append((nx, ny))
                if len(neighbors) == 1:
                    prev = curr
                    curr = neighbors[0]
                elif len(neighbors) == 0:     # 孤立点 → 丢弃
                    break
                else:                        # 多邻居点（交叉点） → 保留
                    kept.append((ex, ey))
                    break
            else:
                # 追踪长度 ≥ prune_length，认为连接到主干，保留
                kept.append((ex, ey))
        return kept

    def detect(self, image_path):
        """
        检测骨架图像中的端点（严格使用击中/击不中变换）

        参数:
            image_path: 图像文件路径

        返回:
            points: 端点坐标列表 [(x, y), ...]
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # === 核心改动：使用击中/击不中变换 ===
        hitmiss_result = np.zeros_like(binary, dtype=np.uint8)
        for kernel in self.endpoint_kernels:
            # 每个核进行一次匹配，结果中非零像素即为匹配的端点
            matched = cv2.morphologyEx(binary, cv2.MORPH_HITMISS, kernel)
            hitmiss_result = cv2.bitwise_or(hitmiss_result, matched)

        # 提取候选端点坐标（注意 OpenCV 坐标系：x 为列，y 为行）
        candidates = [(x, y) for y, x in np.argwhere(hitmiss_result > 0)]

        # 毛刺剪除（与原始逻辑相同，只是输入 candidates 来源变了）
        points = self._prune_spurs(binary, candidates)

        return points