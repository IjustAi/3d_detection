import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class KITTIDataset(Dataset):
    def __init__(self, data_root, split='train', transform=None):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        
        # Load split file
        split_file = os.path.join(data_root, 'ImageSets', f'{split}.txt')
        with open(split_file, 'r') as f:
            self.sample_ids = [line.strip() for line in f.readlines()]
            
        self.image_dir = os.path.join(data_root, 'training', 'image_2')
        self.calib_dir = os.path.join(data_root, 'training', 'calib')
        self.label_dir = os.path.join(data_root, 'training', 'label_2')
        
        print(f'Loaded {len(self.sample_ids)} samples for {split} split')
        
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, f'{sample_id}.png')
        image = Image.open(image_path).convert('RGB')
        
        # Load calibration
        calib_path = os.path.join(self.calib_dir, f'{sample_id}.txt')
        intrinsic = self.load_calibration(calib_path)
        
        # Load labels
        label_path = os.path.join(self.label_dir, f'{sample_id}.txt')
        objects = self.load_labels(label_path)
        
        if self.transform:
            image = self.transform(image)
            
        # Prepare 2D boxes and 3D boxes
        box2d_queries = []
        box3d_targets = []
        
        for obj in objects:
            if obj['type'] in ['Car', 'Van', 'Truck']:
                # 2D bounding box
                box2d = obj['bbox']
                box2d_queries.append(box2d)
                
                # 3D bounding box
                h, w, l = obj['dimensions']
                x, y, z = obj['location']
                
                # 转换为底面中心
                cy = y - h / 2
                
                # [cx, cy, cz, sx, sy, sz, alpha]
                box3d = [x, cy, z, l, h, w, obj['rotation_y']]
                box3d_targets.append(box3d)
        
        # 记录有效对象数量
        num_valid = len(box2d_queries)
        
        if len(box2d_queries) == 0:
            # If no valid objects, use dummy data
            box2d_queries = [[0, 0, 10, 10]]
            box3d_targets = [[0, 0, 1, 1, 1, 1, 0]]
            num_valid = 0
            
        # Convert to tensors
        box2d_queries = torch.FloatTensor(box2d_queries)
        box3d_targets = torch.FloatTensor(box3d_targets)
        intrinsic = torch.FloatTensor(intrinsic)
        
        return image, box2d_queries, box3d_targets, intrinsic, num_valid
    
    def load_calibration(self, calib_path):
        """Load camera intrinsic parameters"""
        with open(calib_path, 'r') as f:
            lines = f.readlines()
            
        P2 = np.array([float(x) for x in lines[2].split()[1:]]).reshape(3, 4)
        fx, fy = P2[0, 0], P2[1, 1]
        cx, cy = P2[0, 2], P2[1, 2]
        
        return [fx, fy, cx, cy]
    
    def load_labels(self, label_path):
        """Load and parse label file"""
        objects = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            data = line.strip().split()
            if len(data) < 15:
                continue
                
            obj = {
                'type': data[0],
                'bbox': [float(x) for x in data[4:8]],
                'dimensions': [float(x) for x in data[8:11]],  # [h, w, l]
                'location': [float(x) for x in data[11:14]],   # [x, y, z]
                'rotation_y': float(data[14])
            }
            objects.append(obj)
            
        return objects


def collate_fn(batch):
    """Custom collate function with valid mask"""
    images, box2d_queries, box3d_targets, intrinsics, num_valids = zip(*batch)
    
    # Stack images and intrinsics
    images = torch.stack(images)
    intrinsics = torch.stack(intrinsics)
    
    # Find max objects in batch
    max_objects = max([len(q) for q in box2d_queries])
    
    padded_box2d = []
    padded_box3d = []
    valid_masks = []
    
    for i in range(len(batch)):
        num_objs = len(box2d_queries[i])
        num_valid = num_valids[i]
        
        # Pad to max_objects
        if num_objs < max_objects:
            pad2d = torch.cat([
                box2d_queries[i], 
                torch.zeros(max_objects - num_objs, 4)
            ])
            pad3d = torch.cat([
                box3d_targets[i], 
                torch.zeros(max_objects - num_objs, 7)
            ])
        else:
            pad2d = box2d_queries[i][:max_objects]
            pad3d = box3d_targets[i][:max_objects]
            num_valid = min(num_valid, max_objects)
        
        # Create valid mask
        valid_mask = torch.zeros(max_objects, dtype=torch.bool)
        valid_mask[:num_valid] = True
        
        padded_box2d.append(pad2d)
        padded_box3d.append(pad3d)
        valid_masks.append(valid_mask)
    
    box2d_queries = torch.stack(padded_box2d)
    box3d_targets = torch.stack(padded_box3d)
    valid_masks = torch.stack(valid_masks)
    
    return images, box2d_queries, box3d_targets, intrinsics, valid_masks