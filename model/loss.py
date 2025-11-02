import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTask3DLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {
            'center': 0.5,
            'size': 0.3,
            'angle': 0.3,
            'depth': 1.0,
            'conf': 0.1
        }
        
        # 用于归一化的统计值（KITTI数据集）
        self.depth_mean = 20.0  # 平均深度约20米
        self.size_mean = 2.0    # 平均尺寸约2米
    
    def smooth_l1_loss(self, pred, target, beta=1.0):
        diff = torch.abs(pred - target)
        loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
        return loss.mean()
    
    def forward(self, pred_boxes, gt_boxes, pred_conf=None):
        # 确保输入是3D的 [B, N, 7]
        if pred_boxes.dim() == 2:
            pred_boxes = pred_boxes.unsqueeze(0)
            gt_boxes = gt_boxes.unsqueeze(0)
            if pred_conf is not None:
                pred_conf = pred_conf.unsqueeze(0)
        
        # XY损失（归一化到图像尺度）
        xy_loss = self.smooth_l1_loss(pred_boxes[..., :2] / 10.0, gt_boxes[..., :2] / 10.0)
        
        # 深度损失（归一化）
        pred_depth_norm = pred_boxes[..., 2] / self.depth_mean
        gt_depth_norm = gt_boxes[..., 2] / self.depth_mean
        depth_loss = self.smooth_l1_loss(pred_depth_norm, gt_depth_norm)
        
        # 尺寸损失（对数空间 + clip避免极值）
        pred_size = torch.clamp(pred_boxes[..., 3:6], min=0.1, max=20.0)
        gt_size = torch.clamp(gt_boxes[..., 3:6], min=0.1, max=20.0)
        
        size_loss = F.l1_loss(torch.log(pred_size + 1e-6), torch.log(gt_size + 1e-6))
        
        # 角度损失（sin/cos）
        pred_angle = pred_boxes[..., 6]
        gt_angle = gt_boxes[..., 6]
        
        angle_loss = (
            F.l1_loss(torch.sin(pred_angle), torch.sin(gt_angle)) +
            F.l1_loss(torch.cos(pred_angle), torch.cos(gt_angle))
        )
        
        # 置信度损失
        conf_loss = torch.tensor(0.0, device=pred_boxes.device)
        if pred_conf is not None:
            target_conf = torch.ones_like(pred_conf)
            conf_loss = F.binary_cross_entropy(pred_conf, target_conf)
        
        # 加权总损失
        total_loss = (
            self.weights['center'] * xy_loss +
            self.weights['depth'] * depth_loss +
            self.weights['size'] * size_loss +
            self.weights['angle'] * angle_loss +
            self.weights['conf'] * conf_loss
        )
        
        losses = {
            'total': total_loss,
            'xy': xy_loss,
            'depth': depth_loss,
            'size': size_loss,
            'angle': angle_loss,
            'conf': conf_loss
        }
        
        return total_loss, losses
