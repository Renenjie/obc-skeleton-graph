import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


class SkeletonVisualizer:
    """
    骨架可视化类
    用于绘制骨架图像、关键点（端点/分支点/拐点）以及边。
    风格完全仿照 CornerVisualizer，但扩展支持多类型节点。
    """

    def __init__(self, save_visualizations: bool = True, colors: Optional[Dict[str, Tuple[int, int, int]]] = None):
        """
        初始化可视化器

        Args:
            save_visualizations: 是否允许保存图像
            colors: 颜色字典，包含以下键（默认值）：
                - 'endpoint': (0, 0, 255)     红色
                - 'branch': (0, 255, 0)       绿色
                - 'corner': (255, 0, 0)       蓝色
                - 'edge': (255, 255, 0)       青色
                - 'background': (150, 150, 150) 灰色
        """
        self.save_visualizations = save_visualizations
        if colors is None:
            self.colors = {
                'endpoint': (0, 0, 255),      # 红色
                'branch': (0, 255, 0),        # 绿色
                'corner': (255, 0, 0),        # 蓝色
                'edge': (255, 255, 0),        # 青色
                'background': (150, 150, 150), # 灰色
            }
        else:
            self.colors = colors

    def compute_connectivity(self, skeleton: np.ndarray, x: int, y: int) -> int:
        """
        计算骨架像素 (x, y) 在 8-邻域内的连接数（不包含自身）
        用于辅助显示连接度颜色。

        Args:
            skeleton: 二值骨架图 (H, W)
            x, y: 像素坐标

        Returns:
            8-邻域中值为 1 的像素个数
        """
        h, w = skeleton.shape
        if skeleton[y, x] == 0:
            return 0
        count = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx] > 0:
                    count += 1
        return count

    def get_node_color(self, branch_count: int) -> Tuple[int, int, int]:
        """
        根据连接度返回颜色（仿照原类中的逻辑）
        - 1: 端点（红色）
        - 2: 拐点（蓝色）
        - >=3: 分支点（黄色/绿色，这里用绿色）
        注意：此方法仅在未提供明确分类时使用。

        Args:
            branch_count: 连接度

        Returns:
            BGR 颜色元组
        """
        if branch_count == 1:
            return self.colors['endpoint']
        elif branch_count == 2:
            return self.colors['corner']
        else:
            return self.colors['branch']

    def visualize(
        self,
        skeleton_img: np.ndarray,
        endpoints: Optional[List[Tuple[int, int]]] = None,
        branch_points: Optional[List[Tuple[int, int]]] = None,
        corners: Optional[List[Tuple[int, int]]] = None,
        edges: Optional[List[List[Tuple[int, int]]]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        draw_edges: bool = True,
        draw_all_points: bool = False,
        point_color: Optional[Tuple[int, int, int]] = None,
        point_radius: int = 3,
        corner_radius: int = 4,
    ) -> np.ndarray:
        """
        可视化骨架和关键点

        Args:
            skeleton_img: 二值骨架图 (H, W)，值为 0 或 1（或 0/255）
            endpoints: 端点坐标列表 [(x, y), ...]
            branch_points: 分支点坐标列表 [(x, y), ...]
            corners: 拐点坐标列表 [(x, y), ...]
            edges: 边列表，每条边为坐标列表 [[(x1,y1), (x2,y2), ...], ...]
            save_path: 若指定则保存图像
            show: 是否显示图像（cv2.imshow）
            draw_edges: 是否绘制边
            draw_all_points: 若为 True，则所有点用同一颜色绘制（point_color）
            point_color: 当 draw_all_points=True 时使用的颜色
            point_radius: 端点/分支点的绘制半径
            corner_radius: 拐点的绘制半径（略大以示区分）

        Returns:
            可视化后的 BGR 图像
        """
        # ---------- 构建 BGR 图像 ----------
        if len(skeleton_img.shape) == 2:
            # 如果是灰度图，转为三通道
            vis_img = cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = skeleton_img.copy()

        # 将骨架像素染成背景色（灰色）
        # 注意：skeleton_img 中值为 1 或 255 都视为骨架
        mask = skeleton_img > 0
        vis_img[mask] = self.colors['background']

        # ---------- 绘制边 ----------
        if draw_edges and edges is not None:
            for edge in edges:
                if len(edge) < 2:
                    continue
                pts = np.array(edge, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_img, [pts], False, self.colors['edge'], 1)

        # ---------- 绘制点 ----------
        if draw_all_points:
            # 所有点用同一颜色绘制（收集所有提供的点）
            all_points = []
            if endpoints:
                all_points.extend(endpoints)
            if branch_points:
                all_points.extend(branch_points)
            if corners:
                all_points.extend(corners)
            color = point_color if point_color is not None else (255, 255, 255)  # 默认白色
            for (x, y) in all_points:
                if 0 <= x < skeleton_img.shape[1] and 0 <= y < skeleton_img.shape[0]:
                    cv2.circle(vis_img, (x, y), point_radius, color, -1)
        else:
            # 分别按类型绘制
            # 端点（红色）
            if endpoints is not None:
                for (x, y) in endpoints:
                    if 0 <= x < skeleton_img.shape[1] and 0 <= y < skeleton_img.shape[0]:
                        cv2.circle(vis_img, (x, y), point_radius, self.colors['endpoint'], -1)
            # 分支点（绿色）
            if branch_points is not None:
                for (x, y) in branch_points:
                    if 0 <= x < skeleton_img.shape[1] and 0 <= y < skeleton_img.shape[0]:
                        cv2.circle(vis_img, (x, y), point_radius, self.colors['branch'], -1)
            # 拐点（蓝色）
            if corners is not None:
                for (x, y) in corners:
                    if 0 <= x < skeleton_img.shape[1] and 0 <= y < skeleton_img.shape[0]:
                        cv2.circle(vis_img, (x, y), corner_radius, self.colors['corner'], -1)

        # ---------- 保存 ----------
        if save_path and self.save_visualizations:
            cv2.imwrite(save_path, vis_img)

        # ---------- 显示 ----------
        if show:
            cv2.imshow("Skeleton Visualization", vis_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return vis_img