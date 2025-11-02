import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import os

from model import ViT3DDetector

class InferenceEngine:
    def __init__(self, cfg, model_path):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔍 Initializing Inference Engine")
        print(f"Device: {self.device}")
        
        # 加载模型
        self.model = ViT3DDetector(
            backbone=cfg.BACKBONE,
            hidden_dim=cfg.HIDDEN_DIM,
            num_queries=cfg.NUM_QUERIES,
            dropout=0.1
        ).to(self.device)
        
        self.load_model(model_path)
        self.model.eval()
        
        print(f"✓ Model loaded and ready")
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize(cfg.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Visualization colors
        self.colors = {
            '2d_box': (0, 255, 0),      # 绿色
            '3d_box': (255, 0, 0),      # 红色
            '3d_front': (0, 0, 255)     # 蓝色（前面）
        }
    
    def load_model(self, model_path):
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            loss = checkpoint.get('best_loss', 0)
            print(f"✓ Loaded from epoch {epoch}, loss: {loss:.4f}")
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"✓ Model: {model_path}")
    
    def run_inference(self, image_path, box2d_query, intrinsic_params, conf_threshold=0.3):
        """运行推理"""
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 预处理
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        box2d_tensor = torch.FloatTensor([box2d_query]).unsqueeze(0).to(self.device)
        intrinsic_tensor = torch.FloatTensor(intrinsic_params).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            pred_boxes, pred_conf = self.model(image_tensor, box2d_tensor, intrinsic_tensor)
        
        # 提取结果
        pred_box3d = pred_boxes[0, 0].cpu().numpy()
        confidence = pred_conf[0, 0, 0].cpu().item()
        
        # 打印结果
        print(f"\\n📊 Prediction:")
        print(f"  Center (x,y,z): ({pred_box3d[0]:.2f}, {pred_box3d[1]:.2f}, {pred_box3d[2]:.2f}) m")
        print(f"  Size (l,h,w): ({pred_box3d[3]:.2f}, {pred_box3d[4]:.2f}, {pred_box3d[5]:.2f}) m")
        print(f"  Rotation: {np.degrees(pred_box3d[6]):.1f}°")
        print(f"  Confidence: {confidence:.3f}")
        
        # 可视化
        result_image = self.create_visualization(
            image_path,
            box2d_query,
            pred_box3d,
            intrinsic_params,
            confidence
        )
        
        return result_image, pred_box3d, confidence
    
    def project_3d_to_2d(self, box3d, intrinsic):
        """将3D框投影到2D平面"""
        cx, cy, cz, sx, sy, sz, alpha = box3d
        fx, fy, cx_img, cy_img = intrinsic
        
        # 8个角点（本地坐标系，底面中心为原点）
        x_corners = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * sx / 2
        y_corners = np.array([0, 0, 0, 0, -1, -1, -1, -1]) * sy  # y=0是底面
        z_corners = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * sz / 2
        
        corners_3d = np.vstack([x_corners, y_corners, z_corners]).T  # (8, 3)
        
        # 旋转矩阵（绕Y轴）
        rot_matrix = np.array([
            [np.cos(alpha), 0, np.sin(alpha)],
            [0, 1, 0],
            [-np.sin(alpha), 0, np.cos(alpha)]
        ])
        
        # 应用旋转和平移
        corners_3d_world = corners_3d @ rot_matrix.T  # 旋转
        corners_3d_world[:, 0] += cx  # 平移x
        corners_3d_world[:, 1] += cy  # 平移y
        corners_3d_world[:, 2] += cz  # 平移z
        
        # 投影到2D
        corners_2d = []
        for corner in corners_3d_world:
            if corner[2] > 0:  # 确保在相机前方
                x = (fx * corner[0]) / corner[2] + cx_img
                y = (fy * corner[1]) / corner[2] + cy_img
                corners_2d.append([x, y])
            else:
                return None  # 在相机后方
        
        return np.array(corners_2d)
    
    def draw_2d_box(self, draw, box2d, color=(0, 255, 0), thickness=3):
        """绘制2D边界框"""
        min_x, min_y, max_x, max_y = box2d
        draw.rectangle([min_x, min_y, max_x, max_y], outline=color, width=thickness)
    
    def draw_3d_box(self, draw, corners_2d, color=(255, 0, 0), thickness=2):
        """绘制3D框投影"""
        if corners_2d is None:
            return
        
        # 定义边
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7)   # 垂直边
        ]
        
        # 绘制所有边
        for edge in edges:
            start_idx, end_idx = edge
            start_point = tuple(corners_2d[start_idx].astype(int))
            end_point = tuple(corners_2d[end_idx].astype(int))
            draw.line([start_point, end_point], fill=color, width=thickness)
        
        # 高亮前面（蓝色）
        front_edges = [(0, 1), (1, 5), (5, 4), (4, 0)]
        for edge in front_edges:
            start_idx, end_idx = edge
            start_point = tuple(corners_2d[start_idx].astype(int))
            end_point = tuple(corners_2d[end_idx].astype(int))
            draw.line([start_point, end_point], fill=self.colors['3d_front'], width=thickness+1)
    
    def create_visualization(self, image_path, box2d_query, pred_box3d, intrinsic, confidence):
        """创建完整的可视化"""
        # 加载原始图像
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image, 'RGBA')
        
        # 1. 绘制2D查询框（绿色）
        self.draw_2d_box(draw, box2d_query, color=self.colors['2d_box'], thickness=3)
        
        # 2. 投影并绘制3D框（红色+蓝色前面）
        corners_2d = self.project_3d_to_2d(pred_box3d, intrinsic)
        if corners_2d is not None:
            self.draw_3d_box(draw, corners_2d, color=self.colors['3d_box'], thickness=2)
        
        # 3. 添加文本信息
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        cx, cy, cz, sx, sy, sz, alpha = pred_box3d
        info_text = [
            f"3D Center: ({cx:.2f}, {cy:.2f}, {cz:.2f}) m",
            f"Dimensions: ({sx:.2f}, {sy:.2f}, {sz:.2f}) m",
            f"Rotation: {np.degrees(alpha):.1f}°",
            f"Confidence: {confidence:.3f}"
        ]
        
        # 文本背景
        text_bg_height = len(info_text) * 25 + 20
        draw.rectangle([10, 10, 420, text_bg_height], fill=(0, 0, 0, 180))
        
        # 绘制文本
        y_offset = 15
        for text in info_text:
            draw.text((15, y_offset), text, fill=(255, 255, 255), font=font)
            y_offset += 25
        
        # 4. 添加图例
        legend_y = text_bg_height + 10
        draw.rectangle([10, legend_y, 300, legend_y + 80], fill=(0, 0, 0, 180))
        
        legend_items = [
            ("Green: 2D Query", self.colors['2d_box']),
            ("Red: 3D Box", self.colors['3d_box']),
            ("Blue: Front Face", self.colors['3d_front'])
        ]
        
        y_offset = legend_y + 10
        for text, color in legend_items:
            draw.rectangle([15, y_offset, 35, y_offset + 15], fill=color)
            draw.text((45, y_offset), text, fill=(255, 255, 255), font=font)
            y_offset += 25
        
        return image
    
    def batch_inference(self, image_paths, box2d_queries, intrinsic_params_list, output_dir):
        """批量推理"""
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        print(f"\\n{'='*60}")
        print(f"Batch Inference: {len(image_paths)} images")
        print(f"{'='*60}")
        
        for idx, (img_path, box2d, intrinsic) in enumerate(zip(image_paths, box2d_queries, intrinsic_params_list)):
            print(f"\\n[{idx+1}/{len(image_paths)}] {os.path.basename(img_path)}")
            
            result_image, pred_box3d, confidence = self.run_inference(img_path, box2d, intrinsic)
            
            # 保存
            output_path = os.path.join(output_dir, f'result_{idx:04d}.jpg')
            result_image.save(output_path)
            print(f"✓ Saved: {output_path}")
            
            results.append({
                'image': img_path,
                'pred_box3d': pred_box3d.tolist(),
                'confidence': confidence,
                'output': output_path
            })
        
        # 保存JSON
        import json
        json_path = os.path.join(output_dir, 'predictions.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\\n{'='*60}")
        print(f"✅ Batch complete! Results: {output_dir}")
        print(f"{'='*60}")
        
        return results