import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

class GraphVisualizer:
    def __init__(self, save_visualizations=True):
        self.save_visualizations = save_visualizations
        self.colors = {
            'endpoint': (0, 0, 255),        # 红色
            'junction_3': (0, 255, 255),    # 黄色
            'junction_4': (255, 0, 255),    # 品红
            'junction_5+': (0, 165, 255),   # 橙色
            'midpoint': (255, 255, 0),      # 青色
            'global_midpoint': (0, 255, 0), # 绿色
            'edge': (255, 255, 0),          # 黄色
        }

    def get_junction_color(self, branch_count):
        if branch_count == 3:
            return self.colors['junction_3']
        elif branch_count == 4:
            return self.colors['junction_4']
        else:
            return self.colors['junction_5+']

    def visualize_keypoints(self, skeleton_img: np.ndarray,
                            endpoints: List[Tuple[int, int]],
                            junctions: List[Dict[str, Any]],
                            all_keypoints: Optional[List[Tuple[int, int]]] = None,
                            corners: Optional[List[Tuple[int, int]]] = None,
                            paths: Optional[List[List[Tuple[int, int]]]] = None,
                            save_path: Optional[str] = None) -> np.ndarray:
        """
        参数：
            skeleton_img: 二值骨架图（0/255）
            endpoints: 端点坐标列表 (x, y)
            junctions: 叉点列表，每个元素为 {'position': (x,y), 'branch_count': n}
            all_keypoints: 所有关键点坐标列表（可选），若提供则用蓝色小点绘制
            corners: 拐点列表（可选）
            paths: 边路径（点序列列表）
            save_path: 保存路径
        """
        if len(skeleton_img.shape) == 2:
            vis_img = cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = skeleton_img.copy()
        vis_img[skeleton_img > 0] = [150, 150, 150]   # 灰色背景

        # 绘制边
        if paths:
            for path in paths:
                if len(path) < 2:
                    continue
                pts = np.array(path, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_img, [pts], False, self.colors['edge'], 1)

        # 绘制所有关键点（作为辅助，用蓝色小圆点）
        if all_keypoints:
            for (x, y) in all_keypoints:
                cv2.circle(vis_img, (x, y), 2, (255, 0, 0), -1)  # 蓝色

        # 绘制端点
        for (x, y) in endpoints:
            cv2.circle(vis_img, (x, y), 3, self.colors['endpoint'], -1)

        # 绘制拐点
        if corners:
            for (x, y) in corners:
                cv2.circle(vis_img, (x, y), 2, (0, 255, 0), -1)   # 绿色

        # 绘制叉点
        for j in junctions:
            x, y = j['position']
            branch_count = j.get('branch_count', 3)
            color = self.get_junction_color(branch_count)
            cv2.rectangle(vis_img, (x-2, y-2), (x+2, y+2), color, -1)

        if save_path and self.save_visualizations:
            cv2.imwrite(save_path, vis_img)
        return vis_img