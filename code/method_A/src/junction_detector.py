import cv2
import numpy as np


class JunctionDetector:
    # 4 邻接偏移：用于统计局部分支连通分量
    FOUR_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def find_junctions(self, skeleton_img, endpoints=None):
        if len(skeleton_img.shape) > 2:
            skeleton_img = cv2.cvtColor(skeleton_img, cv2.COLOR_BGR2GRAY)
        binary = (skeleton_img > 0).astype(np.uint8)
        skeleton = binary
        h, w = skeleton.shape
        junctions = []

        # 8 邻域环形顺序（用于 psi）
        offsets = [(-1, -1), (0, -1), (1, -1), (1, 0),
                   (1, 1), (0, 1), (-1, 1), (-1, 0)]

        for x in range(h):
            for y in range(w):
                if skeleton[x, y] == 0:
                    continue

                # ---------- 1) 计算莫尔邻域变化量 psi ----------
                values = []
                for dx, dy in offsets:
                    nx, ny = x + dx, y + dy
                    values.append(int(skeleton[nx, ny])
                                  if 0 <= nx < h and 0 <= ny < w else 0)
                psi = sum(abs(values[i] - values[(i + 1) % 8]) for i in range(8))
                if psi not in (6, 8):
                    continue

                # ---------- 2) 收集 8 邻域前景点 ----------
                local = set()
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and skeleton[nx, ny]:
                            local.add((nx, ny))

                # ---------- 3) 用 4 连通统计局部分支数（关键修改） ----------
                visited = set()
                components = 0
                for px, py in local:
                    if (px, py) in visited:
                        continue
                    components += 1
                    stack = [(px, py)]
                    visited.add((px, py))
                    while stack:
                        cx, cy = stack.pop()
                        for ddx, ddy in self.FOUR_OFFSETS:   # ← 4 邻接
                            nx, ny = cx + ddx, cy + ddy
                            if (nx, ny) in local and (nx, ny) not in visited:
                                visited.add((nx, ny))
                                stack.append((nx, ny))

                branch_count = components
                if branch_count < 3:
                    continue

                # ---------- 4) N, M 分类 ----------
                order = [
                    skeleton[x, y - 1] > 0 if 0 <= y - 1 < w else False,
                    skeleton[x + 1, y - 1] > 0 if 0 <= x + 1 < h and 0 <= y - 1 < w else False,
                    skeleton[x + 1, y] > 0 if 0 <= x + 1 < h else False,
                    skeleton[x + 1, y + 1] > 0 if 0 <= x + 1 < h and 0 <= y + 1 < w else False,
                    skeleton[x, y + 1] > 0 if 0 <= y + 1 < w else False,
                    skeleton[x - 1, y + 1] > 0 if 0 <= x - 1 < h and 0 <= y + 1 < w else False,
                    skeleton[x - 1, y] > 0 if 0 <= x - 1 < h else False,
                    skeleton[x - 1, y - 1] > 0 if 0 <= x - 1 < h and 0 <= y - 1 < w else False,
                ]
                N = sum(order)
                extended = order + [order[0]]
                M = sum(1 for i in range(8) if extended[i] != extended[i + 1])

                if N >= 3 and M in (6, 8):
                    if branch_count == 3:
                        type_name = "T型交叉"
                    elif branch_count == 4:
                        type_name = "十字交叉"
                    else:
                        type_name = "复杂交叉"
                    junctions.append({
                        'position': (x, y),
                        'branch_count': branch_count,
                        'type': type_name
                    })

        return junctions