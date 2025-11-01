import numpy as np
from .geometry import compute_3d_iou

def calculate_3d_iou(pred_boxes, gt_boxes):
    """Calculate 3D IoU between predicted and ground truth boxes"""
    ious = []
    for pred, gt in zip(pred_boxes, gt_boxes):
        iou = compute_3d_iou(pred, gt)
        ious.append(iou)
    return np.mean(ious) if ious else 0

def calculate_metrics(pred_boxes, gt_boxes):
    """
    Calculate comprehensive metrics for 3D detection
    
    Args:
        pred_boxes: (N, 7) predicted boxes
        gt_boxes: (N, 7) ground truth boxes
    Returns:
        Dictionary of metrics
    """
    if len(pred_boxes) == 0:
        return {}
    
    metrics = {}
    
    # 3D IoU
    metrics['iou_3d'] = calculate_3d_iou(pred_boxes, gt_boxes)
    
    # Center distance error
    center_error = np.linalg.norm(pred_boxes[:, :3] - gt_boxes[:, :3], axis=1)
    metrics['center_error_mean'] = np.mean(center_error)
    metrics['center_error_std'] = np.std(center_error)
    
    # Dimension error
    dim_error = np.abs(pred_boxes[:, 3:6] - gt_boxes[:, 3:6])
    metrics['dim_error_mean'] = np.mean(dim_error)
    metrics['dim_error_std'] = np.std(dim_error)
    
    # Angle error (considering periodicity)
    angle_error = np.abs(np.arctan2(np.sin(pred_boxes[:, 6] - gt_boxes[:, 6]),
                                  np.cos(pred_boxes[:, 6] - gt_boxes[:, 6])))
    metrics['angle_error_mean'] = np.mean(angle_error)
    metrics['angle_error_std'] = np.std(angle_error)
    
    # Success rates
    metrics['success_0.25'] = np.mean(metrics['iou_3d'] > 0.25)
    metrics['success_0.5'] = np.mean(metrics['iou_3d'] > 0.5)
    
    return metrics

def average_precision(precision, recall):
    """Calculate Average Precision from precision-recall curve"""
    # Append sentinel values
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    # Compute the precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Calculate area under curve
    indices = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = np.sum((mrec[indices] - mrec[indices - 1]) * mpre[indices])
    
    return ap