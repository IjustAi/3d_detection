import numpy as np
import torch

def rotation_matrix_y(angle):
    """Create rotation matrix around Y-axis"""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])

def compute_3d_iou(box1, box2):
    """
    Compute 3D IoU between two oriented 3D bounding boxes
    
    Args:
        box1, box2: [cx, cy, cz, sx, sy, sz, alpha]
    Returns:
        iou: 3D Intersection over Union
    """
    # Extract parameters
    center1, size1, angle1 = box1[:3], box1[3:6], box1[6]
    center2, size2, angle2 = box2[:3], box2[3:6], box2[6]
    
    # Convert to corner representation
    corners1 = get_3d_box_corners(center1, size1, angle1)
    corners2 = get_3d_box_corners(center2, size2, angle2)
    
    # Compute intersection volume (simplified approximation)
    # For exact computation, we would need complex polygon intersection
    inter_volume = compute_intersection_volume(corners1, corners2)
    
    # Compute volumes
    volume1 = size1[0] * size1[1] * size1[2]
    volume2 = size2[0] * size2[1] * size2[2]
    union_volume = volume1 + volume2 - inter_volume
    
    return inter_volume / union_volume if union_volume > 0 else 0

def get_3d_box_corners(center, size, angle):
    """Get 8 corners of 3D bounding box"""
    l, h, w = size  # length (x), height (y), width (z)
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]  # bottom center is at y=0
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    
    corners = np.vstack([x_corners, y_corners, z_corners])
    
    # Apply rotation
    rot_mat = rotation_matrix_y(angle)
    corners = rot_mat @ corners
    
    # Apply translation
    corners[0, :] += center[0]
    corners[1, :] += center[1]
    corners[2, :] += center[2]
    
    return corners.T

def compute_intersection_volume(corners1, corners2):
    """Compute intersection volume between two 3D boxes (axis-aligned approximation)"""
    # Get axis-aligned bounding boxes
    min1 = np.min(corners1, axis=0)
    max1 = np.max(corners1, axis=0)
    min2 = np.min(corners2, axis=0)
    max2 = np.max(corners2, axis=0)
    
    # Compute intersection
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_dims = np.maximum(0, inter_max - inter_min)
    
    return inter_dims[0] * inter_dims[1] * inter_dims[2]

def project_3d_to_2d_points(points_3d, intrinsic):
    """Project 3D points to 2D image coordinates"""
    fx, fy, cx, cy = intrinsic
    
    points_2d = []
    for point in points_3d:
        x = (fx * point[0]) / point[2] + cx
        y = (fy * point[1]) / point[2] + cy
        points_2d.append([x, y])
    
    return np.array(points_2d)