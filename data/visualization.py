import numpy as np
from PIL import Image, ImageDraw

class VisualizationUtils:
    def __init__(self):
        self.colors = {
            '2d_box': (0, 255, 0),
            '3d_box': (255, 0, 0),
        }
    
    def draw_2d_box(self, image, box2d, color=(0, 255, 0), thickness=2):
        draw = ImageDraw.Draw(image)
        min_x, min_y, max_x, max_y = box2d
        draw.rectangle([min_x, min_y, max_x, max_y], outline=color, width=thickness)
        return image
    
    def project_3d_to_2d(self, box3d, intrinsic):
        cx, cy, cz, sx, sy, sz, alpha = box3d
        
        corners_3d = np.array([
            [cx - sx/2, cy - sy/2, cz - sz/2],
            [cx + sx/2, cy - sy/2, cz - sz/2],
            [cx + sx/2, cy - sy/2, cz + sz/2],
            [cx - sx/2, cy - sy/2, cz + sz/2],
            [cx - sx/2, cy + sy/2, cz - sz/2],
            [cx + sx/2, cy + sy/2, cz - sz/2],
            [cx + sx/2, cy + sy/2, cz + sz/2],
            [cx - sx/2, cy + sy/2, cz + sz/2],
        ])
        
        rot_matrix = np.array([
            [np.cos(alpha), 0, np.sin(alpha)],
            [0, 1, 0],
            [-np.sin(alpha), 0, np.cos(alpha)]
        ])
        
        corners_3d_rotated = (rot_matrix @ (corners_3d - [cx, cy, cz]).T).T + [cx, cy, cz]
        
        fx, fy, cx_img, cy_img = intrinsic
        corners_2d = []
        
        for corner in corners_3d_rotated:
            x = (fx * corner[0]) / corner[2] + cx_img
            y = (fy * corner[1]) / corner[2] + cy_img
            corners_2d.append([x, y])
        
        return np.array(corners_2d)
    
    def draw_3d_box_projection(self, image, box3d, intrinsic, color=(255, 0, 0), thickness=2):
        corners_2d = self.project_3d_to_2d(box3d, intrinsic)
        
        draw = ImageDraw.Draw(image)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        for edge in edges:
            start_idx, end_idx = edge
            start_point = tuple(corners_2d[start_idx].astype(int))
            end_point = tuple(corners_2d[end_idx].astype(int))
            draw.line([start_point, end_point], fill=color, width=thickness)
        
        return image
    
    def create_visualization(self, image_path, box2d_query, pred_box3d, intrinsic):
        image = Image.open(image_path).convert('RGB')
        
        image = self.draw_2d_box(image, box2d_query, color=self.colors['2d_box'])
        image = self.draw_3d_box_projection(image, pred_box3d, intrinsic, color=self.colors['3d_box'])
        
        draw = ImageDraw.Draw(image)
        cx, cy, cz, sx, sy, sz, alpha = pred_box3d
        info_text = [
            f"3D Center: ({cx:.2f}, {cy:.2f}, {cz:.2f})",
            f"Dimensions: ({sx:.2f}, {sy:.2f}, {sz:.2f})",
            f"Rotation: {alpha:.2f} rad"
        ]
        
        for i, text in enumerate(info_text):
            draw.text((15, 15 + i * 20), text, fill=(255, 255, 255))
        
        return image