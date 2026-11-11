import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class GraphVisualizer:
    def __init__(self, save_visualizations=True, font_path=None):
        self.save_visualizations = save_visualizations
        
        # 定义固定颜色（BGR格式）
        self.colors = {
            'endpoint': (0, 0, 255),           # 红色
            'inflexion': (255, 0, 0),           # 蓝色
            'junction_3': (0, 255, 255),        # 黄色
            'junction_4': (255, 0, 255),        # 品红
            'junction_5+': (0, 165, 255),       # 橙色
        }
        
        # 字体加载逻辑
        try:
            if font_path and os.path.exists(font_path):
                self.font_small = ImageFont.truetype(font_path, 4)
                self.font_medium = ImageFont.truetype(font_path, 4)
                self.font_large = ImageFont.truetype(font_path, 4)
            else:
                # 尝试常见的系统字体路径
                font_paths = [
                    "C:/Windows/Fonts/simhei.ttf",       # Windows 黑体
                    "C:/Windows/Fonts/msyh.ttc",         # Windows 微软雅黑
                    "/System/Library/Fonts/PingFang.ttc", # macOS
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux
                ]
                loaded = False
                for path in font_paths:
                    if os.path.exists(path):
                        self.font_small = ImageFont.truetype(path, 8)
                        self.font_medium = ImageFont.truetype(path, 8)
                        self.font_large = ImageFont.truetype(path, 8)
                        loaded = True
                        break
                if not loaded:
                    # 使用默认字体
                    self.font_small = ImageFont.load_default()
                    self.font_medium = ImageFont.load_default()
                    self.font_large = ImageFont.load_default()
        except Exception:
            # 如果所有尝试都失败，使用默认字体
            self.font_small = ImageFont.load_default()
            self.font_medium = ImageFont.load_default()
            self.font_large = ImageFont.load_default()

    def get_junction_color(self, branch_count):
        """根据分支数量返回对应的交叉点颜色"""
        if branch_count == 3:
            return self.colors['junction_3']
        elif branch_count == 4:
            return self.colors['junction_4']
        else:  # 5个及以上分支
            return self.colors['junction_5+']

    def visualize_keypoints(self, skeleton_img, endpoints, junctions, 
                            inflexions=None, paths=None, save_path=None):
        """
        在骨架图上可视化端点、交叉点、拐点和路径
        skeleton_img: 二值骨架图像
        endpoints: 端点坐标列表 [(x,y), ...]
        junctions: 交叉点列表 [{'position':(x,y), ...}, ...]
        inflexions: 拐点列表 [(x,y), ...] 或带信息的字典列表
        paths: 路径列表（可选）
        save_path: 保存路径
        """
        # 转为彩色图
        if len(skeleton_img.shape) == 2:
            vis_img = cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = skeleton_img.copy()
        
        # 骨架显示为浅灰色
        vis_img[skeleton_img > 0] = [150, 150, 150]
        
        # 绘制路径（若提供）
        if paths:
            path_colors = [
                (255, 255, 0),   # 青
                (0, 255, 255),   # 黄
                (255, 0, 255),   # 品红
                (128, 128, 0),   # 橄榄
                (0, 128, 128),   # 蓝绿
                (128, 0, 128),   # 紫
                (255, 128, 0),   # 橙
                (0, 255, 128),   # 亮绿
                (128, 255, 0),   # 黄绿
                (255, 0, 128)    # 粉
            ]
            for i, path in enumerate(paths):
                if len(path) < 2:
                    continue
                color = path_colors[i % len(path_colors)]
                pts = np.array([[y, x] for x, y in path], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_img, [pts], False, color, 1)
        
        # 绘制端点（红色实心圆）
        for (x, y) in endpoints:
            cv2.circle(vis_img, (x, y), 2, self.colors['endpoint'], -1)
        
        # 绘制交叉点（根据分支数颜色不同的矩形）
        for junction in junctions:
            x, y = junction['position']
            branch_count = junction.get('branch_count', 3)
            color = self.get_junction_color(branch_count)
            cv2.rectangle(vis_img, (x-2, y-2), (x+2, y+2), color, -1)
        
        # 绘制拐点（蓝色实心圆，稍大）
        if inflexions:
            for item in inflexions:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    x, y = item
                elif isinstance(item, dict):
                    x, y = item['position']
                else:
                    continue
                cv2.circle(vis_img, (x, y), 4, self.colors['inflexion'], -1)
        
        # 转换为PIL添加文字
        vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(vis_img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        # 添加端点编号
        for i, (x, y) in enumerate(endpoints):
            color_rgb = (self.colors['endpoint'][2], self.colors['endpoint'][1], self.colors['endpoint'][0])
            draw.text((y+6, x-10), f"E{i}", fill=color_rgb, font=self.font_small)
        
        # 添加交叉点编号
        for i, junction in enumerate(junctions):
            x, y = junction['position']
            branch_count = junction.get('branch_count', 3)
            color = self.get_junction_color(branch_count)
            color_rgb = (color[2], color[1], color[0])
            draw.text((y+8, x-12), f"J{i}", fill=color_rgb, font=self.font_small)
        
        # 添加拐点编号（I前缀）
        if inflexions:
            for i, item in enumerate(inflexions):
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    x, y = item
                elif isinstance(item, dict):
                    x, y = item['position']
                else:
                    continue
                color_rgb = (self.colors['inflexion'][2], 
                             self.colors['inflexion'][1], 
                             self.colors['inflexion'][0])
                draw.text((y+8, x-12), f"I{i}", fill=color_rgb, font=self.font_small)
        
        # 统计信息
        draw.text((10, 10), f"端点: {len(endpoints)} 个", fill=(255,255,255), font=self.font_medium)
        draw.text((10, 40), f"交叉点: {len(junctions)} 个", fill=(0,255,0), font=self.font_medium)
        if inflexions:
            draw.text((10, 70), f"拐点: {len(inflexions)} 个", fill=(255,0,0), font=self.font_medium)
        if paths:
            draw.text((10, 100), f"路径段: {len(paths)} 条", fill=(255,255,0), font=self.font_small)
        
        # 图例
        legend_y = pil_img.height - 150
        draw.text((10, legend_y), "图例:", fill=(255,255,255), font=self.font_small)
        if endpoints:
            draw.text((10, legend_y + 20), "● 端点", fill=(255,255,255), font=self.font_small)
        if junctions:
            draw.text((10, legend_y + 40), "■ 交叉点", fill=(0,255,255), font=self.font_small)
        if inflexions:
            draw.text((10, legend_y + 60), "● 拐点", fill=(255,0,0), font=self.font_small)
        if paths:
            draw.text((10, legend_y + 80), "— 路径", fill=(255,255,0), font=self.font_small)
        
        # 转回OpenCV格式
        vis_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        if save_path and self.save_visualizations:
            cv2.imwrite(save_path, vis_img)
        
        return vis_img