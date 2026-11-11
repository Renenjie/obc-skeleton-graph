import numpy as np
import cv2
from collections import defaultdict

class KeypointDetector:
    """
    端点与交叉点检测类
    输入：骨架图（单像素宽，前景=255或1，背景=0）
    输出：端点列表，交叉点列表（包括环形结构的人工交叉点）
    """
    
    def __init__(self, merge_distance=3):
        self.merge_distance = merge_distance
    
    def detect(self, skeleton):
        """
        检测骨架图中的端点和交叉点，并对环形结构添加人工交叉点
        :param skeleton: np.ndarray, dtype=uint8 或 bool，前景为1或255
        :return: (endpoints, junctions) 两个列表，每个元素为 (x, y)
        """

        skel = skeleton.astype(np.uint8)
        h, w = skel.shape
        offsets = [(-1,-1), (-1,0), (-1,1),
                   (0,-1),          (0,1),
                   (1,-1),  (1,0),  (1,1)]
        
        endpoints = []
        junctions = []
        
        # 第一遍：标记每个像素的邻居数
        neighbor_count_map = {}
        for y in range(h):
            for x in range(w):
                if skel[y, x] == 0:
                    continue
                count = 0
                for dy, dx in offsets:
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] > 0:
                        count += 1
                neighbor_count_map[(x, y)] = count
                if count == 1:
                    endpoints.append((x, y))
                elif count >= 3:
                    junctions.append((x, y))
        
        # 环形处理：找出没有自然端点和交叉点的连通分量，加入人工交叉点（左上角）
        # 获取所有连通分量
        num_labels, labels = cv2.connectedComponents(skel, connectivity=8)
        for label in range(1, num_labels):
            # 获取该连通分量的所有点
            ys, xs = np.where(labels == label)
            if len(xs) == 0:
                continue
            pts = list(zip(xs, ys))
            # 检查该分量是否有自然端点和交叉点
            has_endpoint = any((x, y) in endpoints for (x, y) in pts)
            has_junction = any((x, y) in junctions for (x, y) in pts)
            if not has_endpoint and not has_junction:
                # 环形结构：取左上角点（最小y，然后最小x）
                # 注意坐标格式 (x, y)
                topmost_y = min(y for (x, y) in pts)
                candidates = [(x, y) for (x, y) in pts if y == topmost_y]
                leftmost_x = min(x for (x, y) in candidates)
                artificial_junction = (leftmost_x, topmost_y)
                junctions.append(artificial_junction)
                # 可选：将该点从端点/交叉点检测中排除？不需要，它现在就是交叉点
        
        # 合并邻近点
        if self.merge_distance > 0:
            endpoints = self._merge_nearby_points(endpoints)
            junctions = self._merge_nearby_points(junctions)
        
        return endpoints, junctions
    
    def _merge_nearby_points(self, points):
        if not points:
            return []
        points = list(set(points))
        merged = []
        used = [False] * len(points)
        for i in range(len(points)):
            if used[i]:
                continue
            cluster = [points[i]]
            used[i] = True
            for j in range(i+1, len(points)):
                if used[j]:
                    continue
                for p in cluster:
                    dist = np.hypot(points[j][0] - p[0], points[j][1] - p[1])
                    if dist <= self.merge_distance:
                        cluster.append(points[j])
                        used[j] = True
                        break
            avg_x = int(np.mean([p[0] for p in cluster]))
            avg_y = int(np.mean([p[1] for p in cluster]))
            merged.append((avg_x, avg_y))
        return merged

    def visualize(self, skeleton, endpoints, junctions, save_path=None):
        import matplotlib.pyplot as plt
        img = skeleton.copy().astype(np.uint8) * 255
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for (x, y) in endpoints:
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        for (x, y) in junctions:
            cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Keypoint Detection (Red: endpoints, Blue: junctions)')
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()