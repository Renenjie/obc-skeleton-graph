"""
后处理模块
实现伪分支去除、交叉点合并、结构验证
"""

from typing import List, Tuple, Set, Dict
from collections import deque
import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import DBSCAN
from endpoint_junctionpoint_detector_h import KeypointDetector


class PostProcessor:
    """伪分支去除、节点合并、结构验证"""

    def __init__(self, spur_threshold: float = 5.0, merge_radius: float = 3.0):
        """
        :param spur_threshold: 伪分支长度阈值（欧氏距离，像素）
        :param merge_radius: 交叉点合并半径（像素）
        """
        self.spur_threshold = spur_threshold
        self.merge_radius = merge_radius

    def remove_spurs(self, skeleton: np.ndarray,
                     endpoints: List[Tuple[int, int]],
                     intersections: List[Tuple[int, int]]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        去除长度小于阈值的伪分支（公式 4）
        :return: (新的骨架图像, 更新后的端点列表)
        """
        # 需要导入 KeypointDetector，避免循环导入，放在方法内
        kp_detector = KeypointDetector()

        intersection_set = set(intersections)
        to_remove = set()

        for ep in endpoints:
            if ep in to_remove:
                continue
            # BFS 搜索从端点到最近的交叉点
            queue = deque()
            queue.append((ep[0], ep[1], [ep]))
            visited = {ep}
            found_intersection = None
            path = []
            while queue and found_intersection is None:
                x, y, path_so_far = queue.popleft()
                if (x, y) in intersection_set and (x, y) != ep:
                    found_intersection = (x, y)
                    path = path_so_far
                    break
                # 扩展 8 邻域
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < skeleton.shape[0] and 0 <= ny < skeleton.shape[1]:
                            if skeleton[nx, ny] == 1 and (nx, ny) not in visited:
                                visited.add((nx, ny))
                                queue.append((nx, ny, path_so_far + [(nx, ny)]))
            if found_intersection is not None:
                # 计算欧氏距离
                dist = np.linalg.norm(np.array(ep) - np.array(found_intersection))
                if dist < self.spur_threshold:
                    # 删除路径上的点（不包括交叉点）
                    for px, py in path[:-1]:
                        to_remove.add((px, py))

        # 应用删除
        new_skeleton = skeleton.copy()
        for x, y in to_remove:
            new_skeleton[x, y] = 0

        # 重新检测端点
        new_endpoints, _, _ = kp_detector.detect(new_skeleton)
        return new_skeleton, new_endpoints

    def merge_intersections(self, skeleton: np.ndarray,
                            intersections: List[Tuple[int, int]],
                            stroke_widths: Dict[Tuple[int, int], float] = None) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        合并距离过近的交叉点（公式 5）
        若未提供笔画宽度，则使用距离变换估计局部半径
        :return: (更新后的骨架图像, 新的交叉点列表)
        """
        if not intersections:
            return skeleton, []

        # 如果没有提供笔画宽度，用距离变换估计每个点的局部半径
        if stroke_widths is None:
            # 计算骨架点到背景的距离（欧氏距离变换）
            dist_transform = distance_transform_edt(1 - skeleton)
            stroke_widths = {}
            for (x, y) in intersections:
                stroke_widths[(x, y)] = dist_transform[x, y]  # 半径近似

        coords = np.array(intersections)
        # 使用 DBSCAN 聚类，eps 为固定合并半径（简化实现）
        clustering = DBSCAN(eps=self.merge_radius, min_samples=1).fit(coords)
        labels = clustering.labels_

        new_intersections = []
        for label in set(labels):
            cluster = coords[labels == label]
            center = np.mean(cluster, axis=0).round().astype(int)
            new_intersections.append(tuple(center))

        # 注意：此处不修改骨架图像（仅返回新交叉点列表）
        return skeleton, new_intersections

    def structural_validation(self, skeleton: np.ndarray,
                               endpoints: List[Tuple[int, int]],
                               intersections: List[Tuple[int, int]]) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        结构验证：端点冲突解决（公式 6）和环结构补偿（公式 7）
        这里只实现端点冲突解决
        :return: (新的骨架, 更新后的端点, 更新后的交叉点)
        """
        kp_detector = KeypointDetector()

        # 端点冲突：若两个端点相邻（8邻域），则移除其中一个
        to_remove_endpoints = set()
        for i, ep1 in enumerate(endpoints):
            if ep1 in to_remove_endpoints:
                continue
            for ep2 in endpoints[i + 1:]:
                if abs(ep1[0] - ep2[0]) <= 1 and abs(ep1[1] - ep2[1]) <= 1:
                    to_remove_endpoints.add(ep2)

        new_endpoints = [ep for ep in endpoints if ep not in to_remove_endpoints]
        new_skeleton = skeleton.copy()
        for x, y in to_remove_endpoints:
            new_skeleton[x, y] = 0

        # 重新检测端点（因为可能删除后导致其他变化，但这里简单返回）
        new_endpoints, _, _ = kp_detector.detect(new_skeleton)
        # 环结构补偿（公式 7）暂不实现
        return new_skeleton, new_endpoints, intersections