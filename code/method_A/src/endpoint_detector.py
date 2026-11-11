import cv2
import numpy as np

class EndpointDetector:
    def __init__(self, min_branch_len=5, ignore_isolated=True):
        """
        min_branch_len : int
            从候选端点沿骨架行走的最小步数。
            若行走达到该步数，或提前遇到度≠2的点（如交叉点、拐点）且步数达标，则保留端点；
            否则视为毛刺丢弃。
            设为 0 则不验证（保留所有度=1的点）。
        ignore_isolated : bool
            是否忽略度=0的孤立点。默认 True
        """
        self.min_branch_len = min_branch_len
        self.ignore_isolated = ignore_isolated
        # 8 邻域方向
        self.directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                           (1, 0), (1, -1), (0, -1), (-1, -1)]

    def is_skeleton_pixel(self, img, x, y):
        if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
            return img[x, y] > 0
        return False

    def count_neighbors(self, img, x, y):
        count = 0
        for dx, dy in self.directions:
            if self.is_skeleton_pixel(img, x + dx, y + dy):
                count += 1
        return count

    def _is_junction_or_endpoint(self, img, x, y):
        """判断一个骨架点是否为关键点（度 ≠ 2），即端点、交叉点或拐点"""
        if not self.is_skeleton_pixel(img, x, y):
            return False
        return self.count_neighbors(img, x, y) != 2

    def _trace_branch(self, img, start_x, start_y):
        """
        从端点出发沿骨架追踪，返回是否应保留该端点。
        条件：走满 min_branch_len 步则保留，未走满（路径终止）则不保留。
        未走满且遇到关键点则不保留。走满且遇到关键点则保留。
        """
        if self.min_branch_len <= 0:
            return True

        visited = {(start_x, start_y)}
        cur_x, cur_y = start_x, start_y
        steps = 0

        while steps < self.min_branch_len:
            found_next = False
            for dx, dy in self.directions:
                nx, ny = cur_x + dx, cur_y + dy
                if (nx, ny) in visited:
                    continue
                if self.is_skeleton_pixel(img, nx, ny):
                    visited.add((nx, ny))
                    cur_x, cur_y = nx, ny
                    found_next = True
                    steps += 1
                    # 如果新位置是关键点（度≠2），说明已经走到另一个特征点
                    if self._is_junction_or_endpoint(img, nx, ny):
                        # 当行走步数达到最小要求时保留端点
                        return steps >= self.min_branch_len
                    break
            if not found_next:
                # 走到死胡同（路径终止），步数达标则保留
                return steps >= self.min_branch_len

        # 走满 min_branch_len 步，即使没遇到关键点也保
        return True

    def find_endpoints(self, skeleton_img):
        if len(skeleton_img.shape) > 2:
            skeleton_img = cv2.cvtColor(skeleton_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(skeleton_img, 127, 255, cv2.THRESH_BINARY)
        endpoints = []
        height, width = binary.shape

        for x in range(height):
            for y in range(width):
                if binary[x, y] == 0:
                    continue
                n = self.count_neighbors(binary, x, y)
                if n == 0:
                    # 孤立点处理
                    if not self.ignore_isolated:
                        endpoints.append((x, y))
                    continue
                if n == 1:
                    # 度=1，进行分支长度验证
                    if self._trace_branch(binary, x, y):
                        endpoints.append((x, y))
        return endpoints