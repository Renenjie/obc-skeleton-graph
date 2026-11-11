import cv2
import numpy as np

class GraphVisualizer:
    """
    本文方法的图可视化类
    用于展示骨架图、端點、交叉点、等距采样点以及边连接
    """
    def __init__(self, save_visualizations=True):
        self.save_visualizations = save_visualizations
        self.colors = {
            'endpoint': (0, 0, 255),       # 红色：端点
            'junction': (255, 0, 0),       # 蓝色：交叉点
            'sample_point': (0, 255, 255), # 黄色：等距采样点
            'edge': (255, 255, 255),       # 白色：边
            'skeleton': (150, 150, 150),   # 灰色：骨架背景
        }

    def visualize_graph(self, skeleton_img, endpoints, junctions, sample_points,
                        edges, save_path=None):
        """
        可视化整个图结构
        :param skeleton_img: 二值骨架图（前景255或1）
        :param endpoints: 端点列表 [(x, y), ...]
        :param junctions: 交叉点列表 [(x, y), ...]
        :param sample_points: 等距采样点列表 [(x, y), ...]
        :param edges: 边列表 [(i, j), ...] (节点索引，基于 nodes 顺序)
        :param save_path: 保存路径，若不提供且 save_visualizations=True 则自动保存
        """
        # 构建节点列表（顺序保持一致）
        nodes = endpoints + junctions + sample_points
        # 去重（保留顺序）
        nodes = list(dict.fromkeys(nodes))   # Python 3.7+ 保证顺序
        # 创建坐标到索引的映射
        node_index = {pt: idx for idx, pt in enumerate(nodes)}

        # 准备图像
        if len(skeleton_img.shape) == 2:
            vis_img = cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = skeleton_img.copy()
        # 将骨架前景设为灰色
        vis_img[skeleton_img > 0] = self.colors['skeleton']

        # 绘制边（白色线）
        for u, v in edges:
            if u < len(nodes) and v < len(nodes):
                x1, y1 = nodes[u]
                x2, y2 = nodes[v]
                # 注意 cv2.line 使用 (col, row) -> (x, y) 实际为 (列, 行)
                cv2.line(vis_img, (x1, y1), (x2, y2), self.colors['edge'], 1)

        # 绘制等距采样点（黄色小圆）
        for (x, y) in sample_points:
            cv2.circle(vis_img, (x, y), 2, self.colors['sample_point'], -1)

        # 绘制端点（红色圆）
        for (x, y) in endpoints:
            cv2.circle(vis_img, (x, y), 3, self.colors['endpoint'], -1)

        # 绘制交叉点（蓝色方块）
        for (x, y) in junctions:
            cv2.rectangle(vis_img, (x-2, y-2), (x+2, y+2), self.colors['junction'], -1)

        if save_path and self.save_visualizations:
            cv2.imwrite(save_path, vis_img)
        return vis_img

    def visualize_keypoints(self, skeleton_img, endpoints, junctions, sample_points=None,
                            edges=None, save_path=None):
        """
        兼容旧接口的可视化方法（仅绘制关键点，不绘制边）
        """
        if sample_points is None:
            sample_points = []
        if edges is None:
            edges = []
        return self.visualize_graph(skeleton_img, endpoints, junctions, sample_points,
                                    edges, save_path)