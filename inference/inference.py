import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model import ViT3DDetector


class InferenceEngine:
    def __init__(self, cfg, model_path):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = ViT3DDetector().to(self.device)
        self.load_model(model_path)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(cfg.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Model loaded from {model_path}")
    
    def run_inference(self, image_path, box2d_query, intrinsic_params):
        image = Image.open(image_path).convert('RGB')
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        box2d_tensor = torch.FloatTensor([box2d_query]).unsqueeze(0).to(self.device)
        intrinsic_tensor = torch.FloatTensor([intrinsic_params]).to(self.device)
        
        with torch.no_grad():
            pred_boxes = self.model(image_tensor, box2d_tensor, intrinsic_tensor)
        
        pred_box3d = pred_boxes[0, 0].cpu().numpy()
        
        result_image = self.visualizer.create_visualization(
            image_path, box2d_query, pred_box3d, intrinsic_params
        )
        
        return result_image