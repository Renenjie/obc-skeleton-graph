import numpy as np
from typing import List, Tuple, Set


class Pruner:
    """
    骨架剪枝类
    对应论文 Algorithm 1 第 7-11 行
    功能：删除骨架中从端点到第一个分支点且长度小于阈值 l_min 的短枝
    """

    def _get_neighbor_count(self, skeleton: np.ndarray, x: int, y: int) -> int:
        """
        计算骨架像素 (x, y) 在 8-邻域内的骨架邻居数量（不包含自身）

        Args:
            skeleton: 二值骨架图 (H, W)，值为 0 或 1
            x: 列坐标
            y: 行坐标

        Returns:
            8-邻域中值为 1 的像素个数
        """
        h, w = skeleton.shape
        count = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx] == 1:
                    count += 1
        return count

    def _is_branch_point(self, skeleton: np.ndarray, x: int, y: int) -> bool:
        """判断 (x, y) 是否为分支点（邻居数 >= 3）"""
        return self._get_neighbor_count(skeleton, x, y) >= 3

    def _is_endpoint(self, skeleton: np.ndarray, x: int, y: int) -> bool:
        """判断 (x, y) 是否为端点（邻居数 == 1）"""
        return self._get_neighbor_count(skeleton, x, y) == 1

    def _trace_path_to_branch(self, skeleton: np.ndarray, start_x: int, start_y: int):
        """
        从端点开始沿骨架追踪路径，直到遇到第一个分支点或无法继续。

        Args:
            skeleton: 二值骨架图 (H, W)
            start_x, start_y: 起始端点坐标

        Returns:
            (path, reached_branch)
            - path: 从起点到终点的坐标列表（含起点和终点）
            - reached_branch: True 表示成功到达分支点，False 表示到达死胡同或另一个端点
        """
        path = [(start_x, start_y)]

        # 当前像素和前一像素
        curr_x, curr_y = start_x, start_y
        prev_x, prev_y = None, None

        while True:
            # 查找当前像素的所有骨架邻居（排除前一个像素）
            neighbors = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = curr_x + dx, curr_y + dy
                    # 边界检查
                    if not (0 <= nx < skeleton.shape[1] and 0 <= ny < skeleton.shape[0]):
                        continue
                    # 跳过前一个像素（回溯）
                    if prev_x is not None and nx == prev_x and ny == prev_y:
                        continue
                    if skeleton[ny, nx] == 1:
                        neighbors.append((nx, ny))

            # 没有可继续的邻居 -> 死胡同
            if len(neighbors) == 0:
                return path, False

            # 正常情况下只有一个前进方向，取第一个
            next_x, next_y = neighbors[0]

            # 检查下一个点是否为分支点
            if self._is_branch_point(skeleton, next_x, next_y):
                path.append((next_x, next_y))
                return path, True

            # 检查下一个点是否为端点（说明到了另一端，没有遇到分支点）
            if self._is_endpoint(skeleton, next_x, next_y):
                path.append((next_x, next_y))
                return path, False

            # 否则继续前进
            path.append((next_x, next_y))
            prev_x, prev_y = curr_x, curr_y
            curr_x, curr_y = next_x, next_y

    def prune(self,
              skeleton_img: np.ndarray,
              endpoints: List[Tuple[int, int]],
              branch_points: List[Tuple[int, int]],
              min_length: int = 10) -> np.ndarray:
        """
        对骨架图进行剪枝，删除长度小于 min_length 的短枝。

        Args:
            skeleton_img: 二值骨架图，dtype=np.uint8，值为 0（背景）或 1（骨架）
            endpoints: 端点列表，每个元素为 (x, y)
            branch_points: 分支点列表，每个元素为 (x, y)（本方法中主要用于辅助，实际判断基于实时邻居数）
            min_length: 路径长度阈值（像素数），论文设为 10

        Returns:
            剪枝后的二值骨架图，值仍为 0 或 1
        """
        # 输入校验
        if skeleton_img.ndim != 2:
            raise ValueError("skeleton_img 必须是二维数组")
        if skeleton_img.dtype != np.uint8:
            raise ValueError("skeleton_img.dtype 必须为 np.uint8")

        # 复制图像，避免修改原图
        skeleton = skeleton_img.copy()

        # 用集合加速端点查找
        endpoints_set = set(endpoints)

        # 记录需要删除的像素
        pixels_to_remove: Set[Tuple[int, int]] = set()

        for (ex, ey) in endpoints:
            # 如果该端点已经被标记删除或其像素已被移除，则跳过
            if (ex, ey) in pixels_to_remove:
                continue
            if skeleton[ey, ex] == 0:
                continue

            # 追踪从该端点到分支点的路径
            path, reached_branch = self._trace_path_to_branch(skeleton, ex, ey)

            # 情况1：成功到达分支点
            if reached_branch:
                # 路径中包含分支点本身，需要排除（保留分支点）
                path_excluding_branch = path[:-1]  # 去掉最后一个元素（分支点）
                path_length = len(path_excluding_branch)  # 像素个数

                if path_length < min_length:
                    for (px, py) in path_excluding_branch:
                        pixels_to_remove.add((px, py))
            else:
                # 情况2：未到达分支点（例如两端都是端点的孤立短线）
                # 如果整条路径长度 < min_length，视为噪声一并删除
                if len(path) < min_length:
                    for (px, py) in path:
                        pixels_to_remove.add((px, py))

        # 执行像素删除
        for (px, py) in pixels_to_remove:
            if 0 <= py < skeleton.shape[0] and 0 <= px < skeleton.shape[1]:
                skeleton[py, px] = 0

        return skeleton