import numpy as np
import cv2
from typing import List, Tuple


class EndpointBranchDetector:
    """
    端点和分支点检测类
    对应论文 Algorithm 1 第 4-6 行，基于 3x3 邻域计数识别关键点。

    重要说明：
    论文公式 (2) 存在明显笔误，若按原公式 (C_i==2 为端点, C_i==3 为分支点) 执行，
    会将直线上的普通点误判为分支点，真正的三分叉点 (C_i==4) 反而被忽略。
    
    本类采用标准骨架检测逻辑进行修正：
        - C_i == 2  -> 端点 (Endpoint)
        - C_i >= 4  -> 分支点/交叉点 (Branch/Junction)
        - C_i == 3  -> 普通直线点，忽略
        - C_i == 1  -> 孤立噪点，忽略
    """

    def detect(self, skeleton_img: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        检测骨架图像中的端点和分支点。

        Args:
            skeleton_img: 二值化骨架图，要求 dtype 为 uint8，骨架像素值为 1，背景为 0。
                         形状为 (H, W)。

        Returns:
            包含两个列表的元组：
            - 端点列表: List[Tuple[int, int]]，每个元素为 (x, y)，x为列坐标，y为行坐标。
            - 分支点列表: List[Tuple[int, int]]，每个元素为 (x, y)。

        Raises:
            ValueError: 当输入数组维度不为 2，或像素值不全是 0 和 1 时抛出。
        """
        # ---------- 输入校验 ----------
        if skeleton_img.ndim != 2:
            raise ValueError(f"输入图像必须为二维数组，当前维度为 {skeleton_img.ndim}")

        unique_vals = np.unique(skeleton_img)
        if not np.all(np.isin(unique_vals, [0, 1])):
            raise ValueError("输入图像的像素值必须仅为 0（背景）和 1（骨架），请检查二值化逻辑")

        # ---------- 3x3 邻域计数 ----------
        # 使用全 1 卷积核计算每个像素邻域内的骨架像素总数（包含中心点自身）
        kernel = np.ones((3, 3), dtype=np.float32)
        # 边界使用常量填充 0，确保边缘像素的邻域不会越界
        conv_result = cv2.filter2D(
            skeleton_img.astype(np.float32),
            ddepth=-1,
            kernel=kernel,
            borderType=cv2.BORDER_CONSTANT
        )
        # conv_result[y, x] 即为论文中的 C_i

        # 获取所有骨架像素的坐标 (y, x)
        ys, xs = np.where(skeleton_img == 1)

        endpoints = []
        branch_points = []

        # ---------- 分类逻辑 ----------
        for y, x in zip(ys, xs):
            c_val = int(conv_result[y, x])

            if c_val == 2:
                # 只有自身 + 1 个相邻骨架点 -> 端点（线段末端）
                endpoints.append((x, y))
            elif c_val >= 4:
                # 自身 + 3 个及以上相邻骨架点 -> 分支点或交叉点
                branch_points.append((x, y))
            # c_val == 3 时是直线上的普通点，不保留
            # c_val == 1 时是孤立噪点，不保留

        return endpoints, branch_points