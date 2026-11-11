import numpy as np
import math
from typing import List, Tuple, Set, Optional


class CornerDetector:
    """
    拐点检测类（多边形近似）
    对应论文 Algorithm 1 第 19-29 行，公式 (3) 和 (4)
    
    功能：在剪枝后的骨架图中，检测每对相邻关键点路径上的转折点，
         并将其作为新的关键点加入到图模型中，以保留笔画的几何形状。
    """

    def _get_neighbors(self, skeleton: np.ndarray, x: int, y: int) -> List[Tuple[int, int]]:
        """
        获取像素 (x, y) 的 8-邻域中所有骨架像素的坐标（不包含自身）

        Args:
            skeleton: 二值骨架图 (H, W)，值为 0 或 1
            x, y: 像素坐标

        Returns:
            骨架邻居坐标列表
        """
        h, w = skeleton.shape
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx] == 1:
                    neighbors.append((nx, ny))
        return neighbors

    def _trace_path_between(self,
                            skeleton: np.ndarray,
                            start: Tuple[int, int],
                            end: Tuple[int, int],
                            other_nodes: Set[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        """
        追踪骨架图中从 start 到 end 的路径（不含其他关键点）。
        增加 visited 集合防止循环，并加入最大步数限制防止死循环。
        """
        if start == end:
            return [start]
    
        path = [start]
        prev = None
        curr = start
        visited = {start}
        max_steps = 5000  # 安全限制，防止意外无限循环
    
        steps = 0
        while curr != end and steps < max_steps:
            steps += 1
            neighbors = self._get_neighbors(skeleton, curr[0], curr[1])
    
            next_point = None
            for nbr in neighbors:
                if prev is not None and nbr == prev:
                    continue
                if nbr in visited:
                    continue
                if nbr in other_nodes and nbr != end:
                    continue
                if nbr == end:
                    next_point = nbr
                    break
                next_point = nbr
    
            if next_point is None:
                return None
    
            path.append(next_point)
            prev, curr = curr, next_point
            visited.add(curr)
    
        if curr == end:
            return path
        else:
            # 达到最大步数，说明可能陷入循环或路径过长，视为无效
            return None

    def _compute_distance(self,
                          p: Tuple[int, int],
                          a: Tuple[int, int],
                          b: Tuple[int, int]) -> float:
        """
        计算点 p 到线段 ab 的距离（对应论文公式 (4) 的效果）
        使用叉积法计算点到直线的垂直距离，数值更稳定

        Args:
            p: 目标点 (x, y)
            a, b: 线段端点 (x, y)

        Returns:
            点 p 到线段 ab 的垂直距离
        """
        ax, ay = a
        bx, by = b
        px, py = p

        # 向量 ab 和 ap
        ab_x = bx - ax
        ab_y = by - ay
        ap_x = px - ax
        ap_y = py - ay

        # 叉积的绝对值 = |ab × ap|
        cross = abs(ab_x * ap_y - ab_y * ap_x)

        # 线段 ab 的长度
        ab_len = math.sqrt(ab_x * ab_x + ab_y * ab_y)

        if ab_len < 1e-9:
            return 0.0

        # 垂直距离 = |叉积| / |ab|
        return cross / ab_len

    def _compute_angle(self,
                       p: Tuple[int, int],
                       a: Tuple[int, int],
                       b: Tuple[int, int]) -> float:
        """
        计算点 p 处的夹角（对应论文公式 (3)）
        角度为 ∠apb，即向量 pa 与 pb 的夹角

        Args:
            p: 目标点 (x, y)
            a, b: 线段端点 (x, y)

        Returns:
            角度（弧度），范围 [0, π]
        """
        ax, ay = a
        bx, by = b
        px, py = p

        # 向量 pa 和 pb
        pa_x = ax - px
        pa_y = ay - py
        pb_x = bx - px
        pb_y = by - py

        # 模长
        len_pa = math.sqrt(pa_x * pa_x + pa_y * pa_y)
        len_pb = math.sqrt(pb_x * pb_x + pb_y * pb_y)

        if len_pa < 1e-9 or len_pb < 1e-9:
            return math.pi  # 点重合时返回最大值

        # 点积
        dot = pa_x * pb_x + pa_y * pb_y

        # 限制 cos 值范围，避免数值溢出
        cos_theta = max(-1.0, min(1.0, dot / (len_pa * len_pb)))
        return math.acos(cos_theta)

    def _find_max_distance_point(self,
                                 path: List[Tuple[int, int]],
                                 start: Tuple[int, int],
                                 end: Tuple[int, int]) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        在路径中寻找距离线段 start-end 最远的点

        Args:
            path: 路径坐标列表（含 start 和 end）
            start: 起点
            end: 终点

        Returns:
            (最远点坐标, 最大距离值)，若路径长度 <= 2 则返回 (None, 0)
        """
        if len(path) <= 2:
            return None, 0.0

        max_dist = -1.0
        farthest_point = None

        for p in path[1:-1]:  # 排除起点和终点本身
            d = self._compute_distance(p, start, end)
            if d > max_dist:
                max_dist = d
                farthest_point = p

        return farthest_point, max_dist

    def detect(self,
               skeleton_img: np.ndarray,
               nodes: List[Tuple[int, int]],
               edges: List[Tuple[int, int]],
               d_max: float = 5.0,
               theta_min: float = 2.36) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        使用多边形近似检测拐点

        Args:
            skeleton_img: 剪枝后的二值骨架图，值为 0 或 1
            nodes: 初始关键点列表，每个元素为 (x, y)（端点 + 分支点）
            edges: 初始边列表，每个元素为 (i, j)，表示 nodes[i] 与 nodes[j] 相连
            d_max: 距离阈值，论文 Table 1 设为 5
            theta_min: 角度阈值（弧度），论文 Table 1 设为 2.36（约 135°）

        Returns:
            (updated_nodes, updated_edges)
            - updated_nodes: 添加了拐点的完整关键点列表
            - updated_edges: 更新后的边列表
        """
        # 输入校验
        if skeleton_img.ndim != 2:
            raise ValueError("skeleton_img 必须为二维数组")

        # 复制节点和边，避免修改原始数据
        current_nodes = list(nodes)
        current_edges = list(edges)

        # 用于快速查找节点的坐标->索引映射（随节点增加而更新）
        node_to_idx = {node: idx for idx, node in enumerate(current_nodes)}

        # 需要处理的边列表（动态增加）
        edges_to_process = list(current_edges)

        while edges_to_process:
            # 取出一条边
            idx_i, idx_j = edges_to_process.pop(0)

            # 获取端点坐标
            node_i = current_nodes[idx_i]
            node_j = current_nodes[idx_j]

            # 如果两个端点相同，跳过
            if node_i == node_j:
                continue

            # 追踪路径（避开其他关键点）
            # 获取当前所有关键点的集合（用于路径约束，但为了准确追踪，不包含起点终点）
            other_nodes_set = set(current_nodes) - {node_i, node_j}
            path = self._trace_path_between(skeleton_img, node_i, node_j, other_nodes_set)

            if path is None or len(path) <= 2:
                continue

            # 找路径中距离线段最远的点
            farthest_point, max_dist = self._find_max_distance_point(path, node_i, node_j)

            # 如果最远点不存在，或距离不够大，则跳过
            if farthest_point is None or max_dist <= d_max:
                continue

            # 计算该点处的夹角
            angle = self._compute_angle(farthest_point, node_i, node_j)

            # 判定是否为拐点（距离 > 阈值 且 角度 < 阈值）
            if angle >= theta_min:
                continue

            # 确认这是一个拐点，加入节点列表
            # 避免重复添加已存在的点
            if farthest_point in node_to_idx:
                new_idx = node_to_idx[farthest_point]
            else:
                new_idx = len(current_nodes)
                current_nodes.append(farthest_point)
                node_to_idx[farthest_point] = new_idx

            # 移除旧边 (idx_i, idx_j)
            # 注意：current_edges 中可能存在多条相同边，只移除一条
            try:
                current_edges.remove((idx_i, idx_j))
            except ValueError:
                # 如果边已经被其他操作移除，则尝试查找反向边
                try:
                    current_edges.remove((idx_j, idx_i))
                except ValueError:
                    pass

            # 添加两条新边 (idx_i, new_idx) 和 (new_idx, idx_j)
            # 为避免重复，检查是否已存在
            if (idx_i, new_idx) not in current_edges and (new_idx, idx_i) not in current_edges:
                current_edges.append((idx_i, new_idx))
                edges_to_process.append((idx_i, new_idx))
            if (new_idx, idx_j) not in current_edges and (idx_j, new_idx) not in current_edges:
                current_edges.append((new_idx, idx_j))
                edges_to_process.append((new_idx, idx_j))

        return current_nodes, current_edges