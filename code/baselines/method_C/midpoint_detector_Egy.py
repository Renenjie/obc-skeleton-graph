import numpy as np
import cv2
from scipy.optimize import least_squares

from endpoint_detector_Egy import EndpointDetector
from junctionpoint_detector_Egy import Junctionpoint_detector


class InflexionDetector:
    """骨架拐点检测器 —— 基于贝塞尔曲线拟合与曲率极值（带 NMS）"""

    def __init__(self, curvature_thresh: float = 0.02, t_suppress: float = 0.1):
        """
        参数:
            curvature_thresh: 拐点最小曲率阈值，低于该值的极值点被忽略。
                              建议值 0.02~0.05，视图像噪声而定。
            t_suppress: 参数 t 方向非极大抑制距离（归一化 t ∈ [0,1]）。
                        两个极大值点 t 间距小于此值，则仅保留曲率更大者。
        """
        self.curvature_thresh = curvature_thresh
        self.t_suppress = t_suppress

    # ---------- 贝塞尔数学工具 ----------
    @staticmethod
    def _bezier(t, P0, P1, P2, P3):
        return (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3

    @staticmethod
    def _bezier_deriv1(t, P0, P1, P2, P3):
        return 3 * (1 - t) ** 2 * (P1 - P0) + 6 * (1 - t) * t * (P2 - P1) + 3 * t ** 2 * (P3 - P2)

    @staticmethod
    def _bezier_deriv2(t, P0, P1, P2, P3):
        return 6 * (1 - t) * (P2 - 2 * P1 + P0) + 6 * t * (P3 - 2 * P2 + P1)

    def _curvature(self, t, P0, P1, P2, P3):
        d1 = self._bezier_deriv1(t, P0, P1, P2, P3)
        d2 = self._bezier_deriv2(t, P0, P1, P2, P3)
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        norm = np.linalg.norm(d1)
        if norm < 1e-10:
            return 0.0
        return abs(cross) / (norm ** 3)

    def _fit_bezier(self, points):
        if len(points) < 4:
            return None
        pts = np.array(points, dtype=np.float64)
        diffs = np.diff(pts, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cum = np.insert(np.cumsum(dists), 0, 0)
        t_vals = cum / cum[-1]

        P0 = pts[0].copy()
        P3 = pts[-1].copy()
        idx1 = min(int(len(pts) * 0.33), len(pts) - 1)
        idx2 = min(int(len(pts) * 0.66), len(pts) - 1)
        P1_guess = pts[idx1].copy()
        P2_guess = pts[idx2].copy()
        init = np.concatenate([P0, P1_guess, P2_guess, P3])

        def residuals(params):
            p0 = params[0:2]; p1 = params[2:4]
            p2 = params[4:6]; p3 = params[6:8]
            curve_pts = np.array([self._bezier(ti, p0, p1, p2, p3) for ti in t_vals])
            return (curve_pts - pts).ravel()

        try:
            result = least_squares(residuals, init, max_nfev=2000)
            if result.success:
                p = result.x
                return p[0:2], p[2:4], p[4:6], p[6:8]
        except Exception:
            pass
        return None

    def _find_curvature_extrema(self, P0, P1, P2, P3, n_samples=200):
        """搜索曲率局部极大值，并执行 t 方向 NMS"""
        t_vals = np.linspace(0.01, 0.99, n_samples)
        curv = np.array([self._curvature(t, P0, P1, P2, P3) for t in t_vals])

        # 1. 找出所有局部极大值点（同时满足阈值）
        candidates = []
        for i in range(1, n_samples - 1):
            if curv[i] > curv[i - 1] and curv[i] > curv[i + 1]:
                if curv[i] >= self.curvature_thresh:
                    candidates.append((t_vals[i], curv[i]))

        # 2. 如果没有局部极大值，则取全局最大（若超过阈值）
        if not candidates:
            max_idx = np.argmax(curv)
            if curv[max_idx] >= self.curvature_thresh:
                return [t_vals[max_idx]]
            else:
                return []

        # 3. 按 t 排序，进行非极大值抑制
        candidates.sort(key=lambda x: x[0])
        suppressed = []
        for t, c in candidates:
            if not suppressed:
                suppressed.append((t, c))
            else:
                last_t, last_c = suppressed[-1]
                if t - last_t < self.t_suppress:
                    # 距离过近，保留曲率更大者
                    if c > last_c:
                        suppressed[-1] = (t, c)
                else:
                    suppressed.append((t, c))

        # 返回抑制后的 t 值
        return [t for t, _ in suppressed]

    # ---------- 主检测接口 ----------
    def detect(self, image_path: str):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        ep_detector = EndpointDetector()
        jp_detector = Junctionpoint_detector()
        endpoints = ep_detector.detect(image_path)
        junctions = jp_detector.detect(image_path)

        key_set = set((x, y) for (x, y) in endpoints + junctions)

        visited_edges = set()
        all_segments = []
        h, w = binary.shape
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),          (0, 1),
                   (1, -1), (1, 0), (1, 1)]

        def neighbors(x, y):
            nb = []
            for dx, dy in offsets:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and binary[ny, nx] == 255:
                    nb.append((nx, ny))
            return nb

        def trace_from(start_point):
            segment = [start_point]
            current = start_point
            previous = None
            while True:
                nbs = neighbors(current[0], current[1])
                if previous is not None and previous in nbs:
                    nbs.remove(previous)
                if len(nbs) == 0:
                    break
                if len(nbs) == 1:
                    next_pt = nbs[0]
                else:
                    break
                edge = tuple(sorted((current, next_pt)))
                if edge in visited_edges:
                    break
                visited_edges.add(edge)
                segment.append(next_pt)
                if next_pt in key_set:
                    break
                previous, current = current, next_pt
            return segment

        for pt in endpoints:
            seg = trace_from(pt)
            if len(seg) >= 4:
                all_segments.append(seg)

        for pt in junctions:
            nbs = neighbors(pt[0], pt[1])
            for nb in nbs:
                edge = tuple(sorted((pt, nb)))
                if edge not in visited_edges and nb not in key_set:
                    visited_edges.add(edge)
                    seg = [pt, nb]
                    current = nb
                    previous = pt
                    while True:
                        nxt = [n for n in neighbors(current[0], current[1]) if n != previous]
                        if len(nxt) == 0:
                            break
                        next_pt = nxt[0]
                        edge = tuple(sorted((current, next_pt)))
                        if edge in visited_edges:
                            break
                        visited_edges.add(edge)
                        seg.append(next_pt)
                        if next_pt in key_set:
                            break
                        previous, current = current, next_pt
                    if len(seg) >= 4:
                        all_segments.append(seg)

        all_inflexions = []
        for seg in all_segments:
            pts = np.array(seg, dtype=np.float64)
            ctrl = self._fit_bezier(pts)
            if ctrl is None:
                continue
            P0, P1, P2, P3 = ctrl
            t_extrema = self._find_curvature_extrema(P0, P1, P2, P3)
            for t_val in t_extrema:
                pt_on_curve = self._bezier(t_val, P0, P1, P2, P3)
                dists = np.linalg.norm(pts - pt_on_curve, axis=1)
                idx = np.argmin(dists)
                ix, iy = int(pts[idx, 0]), int(pts[idx, 1])
                all_inflexions.append((ix, iy))

        return list(set(all_inflexions))