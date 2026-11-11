
import cv2
import numpy as np

class GraphVisualizer:
    def __init__(self, save_visualizations=True):
        self.save_visualizations = save_visualizations
        self.colors = {
            'endpoint': (0, 0, 255),
            'junction_3': (0, 255, 255),
            'junction_4': (255, 0, 255),
            'junction_5+': (0, 165, 255),
            'midpoint': (255, 255, 0),
            'edge': (255, 255, 0),
        }

    def get_junction_color(self, branch_count):
        if branch_count == 3:
            return self.colors['junction_3']
        elif branch_count == 4:
            return self.colors['junction_4']
        else:
            return self.colors['junction_5+']

    def visualize_keypoints(self, skeleton_img,
                            endpoints, junctions, circular_midpoints=None,
                            paths=None, save_path=None):
        if len(skeleton_img.shape) == 2:
            vis_img = cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = skeleton_img.copy()
        vis_img[skeleton_img > 0] = [150, 150, 150]

        if paths:
            for path in paths:
                if len(path) < 2:
                    continue
                # 路径点也需转换坐标
                pts = np.array([(y, x) for (x, y) in path], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_img, [pts], False, self.colors['edge'], 1)

        for (x, y) in endpoints:
            cv2.circle(vis_img, (y, x), 3, self.colors['endpoint'], -1)  # 转换为(col, row)

        if circular_midpoints:
            for (x, y) in circular_midpoints:
                cv2.circle(vis_img, (y, x), 2, self.colors['midpoint'], -1)

        for j in junctions:
            x, y = j['position']
            branch_count = j.get('branch_count', 3)
            color = self.get_junction_color(branch_count)
            cv2.rectangle(vis_img, (y-2, x-2), (y+2, x+2), color, -1)  # 转换

        if save_path and self.save_visualizations:
            cv2.imwrite(save_path, vis_img)
        return vis_img