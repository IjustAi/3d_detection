import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTask3DLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {'center': 1.0, 'size': 0.5, 'angle': 0.2}
        
    def forward(self, pred_boxes, gt_boxes):
        # Center loss (L1)
        center_loss = F.l1_loss(pred_boxes[..., :3], gt_boxes[..., :3])
        
        # Size loss (L1)
        size_loss = F.l1_loss(pred_boxes[..., 3:6], gt_boxes[..., 3:6])
        
        # Angle loss (L1 for ry)
        angle_loss = F.l1_loss(pred_boxes[..., 6], gt_boxes[..., 6])
        
        # Weighted combination
        total_loss = (self.weights['center'] * center_loss +
                     self.weights['size'] * size_loss +
                     self.weights['angle'] * angle_loss)
        
        losses = {
            'total': total_loss,
            'center': center_loss,
            'size': size_loss,
            'angle': angle_loss
        }
        
        return total_loss, losses