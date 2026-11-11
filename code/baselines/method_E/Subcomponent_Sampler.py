import numpy as np
import cv2

class SubcomponentSampler:
    """
    删除交叉点、提取子连通分量、等距采样生成节点
    """
    def __init__(self, sample_distance=5):
        """
        :param sample_distance: 等距采样间隔 D (像素)
        """
        self.sample_distance = sample_distance

    def sample(self, skeleton, endpoints, junctions):
        """
        执行删除交叉点和等距采样
        :param skeleton: 原始骨架图 (前景255或1，背景0)
        :param endpoints: 端点列表 [(x,y), ...]
        :param junctions: 交叉点列表 [(x,y), ...]
        :return: 节点列表，每个元素为 (x, y)
        """
        # 确保二值，前景=1
        skel = (skeleton > 0).astype(np.uint8)
        
        # 1. 删除交叉点
        skel_no_junc = self._remove_junctions(skel, junctions)
        
        # 2. 寻找所有连通分量（8连通）
        num_labels, labels = cv2.connectedComponents(skel_no_junc, connectivity=8)
        
        # 3. 初始化节点集合（包含端点和交叉点）
        nodes = set()
        for (x, y) in endpoints:
            nodes.add((x, y))
        for (x, y) in junctions:
            nodes.add((x, y))
        
        # 4. 对每个子连通分量进行路径追踪和等距采样
        for label in range(1, num_labels):
            mask = (labels == label).astype(np.uint8)
            # 提取有序路径点
            path_points = self._extract_path(mask)
            if len(path_points) < 2:
                if len(path_points) == 1:
                    nodes.add(tuple(path_points[0]))
                continue
            # 等距采样
            sampled = self._equidistant_sampling(path_points, self.sample_distance)
            for pt in sampled:
                nodes.add(pt)
        
        return list(nodes)

    def _remove_junctions(self, skeleton, junctions):
        """在副本上将交叉点像素置为0"""
        skel_copy = skeleton.copy()
        for (x, y) in junctions:
            # 注意OpenCV图像索引为 (行, 列)
            skel_copy[y, x] = 0
        return skel_copy

    def _extract_path(self, component_mask):
        """
        从无分支的连通分量中提取有序路径点列表
        :param component_mask: 二值图，前景=1，仅含一个连通分量
        :return: list of (x, y) 坐标（按路径顺序）
        """
        h, w = component_mask.shape
        ys, xs = np.where(component_mask > 0)
        if len(xs) == 0:
            return []
        points = list(zip(xs, ys))
        if len(points) == 1:
            return points
        
        # 构建邻接关系（8邻域）
        adj = {pt: [] for pt in points}
        for (x, y) in points:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and component_mask[ny, nx] > 0:
                        adj[(x, y)].append((nx, ny))
        
        # 查找端点（邻居数为1的点）
        endpoints = [pt for pt in points if len(adj[pt]) == 1]
        
        if len(endpoints) >= 2:
            # 开曲线：从一端走到另一端
            start = endpoints[0]
            path = []
            cur = start
            prev = None
            while True:
                path.append(cur)
                neighbors = adj[cur]
                nxt = None
                for nb in neighbors:
                    if nb != prev:
                        nxt = nb
                        break
                if nxt is None:
                    break
                prev, cur = cur, nxt
                if len(path) > len(points):
                    break
            return path
        else:
            # 闭合曲线（如圆环）：任意起点，走一圈
            start = points[0]
            path = []
            cur = start
            prev = None
            while True:
                path.append(cur)
                neighbors = adj[cur]
                # 闭合环每个点有2个邻居，选择不同于prev的那个
                if len(neighbors) == 2:
                    nxt = neighbors[0] if neighbors[0] != prev else neighbors[1]
                else:
                    break
                prev, cur = cur, nxt
                if cur == start:
                    # 闭合，结束
                    break
                if len(path) > len(points):
                    break
            return path

    def _equidistant_sampling(self, path_points, D):
        """
        沿路径等距采样，包含首尾点
        :param path_points: 有序坐标列表 [(x,y), ...]
        :param D: 采样间距（像素）
        :return: 采样点列表
        """
        if len(path_points) <= 1:
            return path_points[:]
        
        # 计算累积弧长
        dist = [0.0]
        for i in range(1, len(path_points)):
            x0, y0 = path_points[i-1]
            x1, y1 = path_points[i]
            d = np.hypot(x1 - x0, y1 - y0)
            dist.append(dist[-1] + d)
        total_len = dist[-1]
        
        sampled = [path_points[0]]
        cur_dist = D
        idx = 1
        while cur_dist < total_len - 1e-6:
            # 找到 cur_dist 所在的区间
            while idx < len(dist) and dist[idx] < cur_dist:
                idx += 1
            if idx >= len(dist):
                break
            # 线性插值
            t = (cur_dist - dist[idx-1]) / (dist[idx] - dist[idx-1])
            x0, y0 = path_points[idx-1]
            x1, y1 = path_points[idx]
            xi = int(x0 + t * (x1 - x0) + 0.5)
            yi = int(y0 + t * (y1 - y0) + 0.5)
            sampled.append((xi, yi))
            cur_dist += D
        
        # 确保终点包含在内
        last = sampled[-1]
        if np.hypot(last[0] - path_points[-1][0], last[1] - path_points[-1][1]) > 1e-3:
            sampled.append(path_points[-1])
        else:
            sampled[-1] = path_points[-1]
        return sampled