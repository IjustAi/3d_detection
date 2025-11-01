import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os

from model import ViT3DDetector
from loss import MultiTask3DLoss
from dataset import KITTIDataset, collate_fn

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = ViT3DDetector().to(self.device)
        
        # Initialize dataset and dataloader
        self.train_dataset = KITTIDataset(
            cfg.DATA_ROOT, 
            split='train',
            transform=self.get_transforms(train=True)
        )
        self.val_dataset = KITTIDataset(
            cfg.DATA_ROOT,
            split='val', 
            transform=self.get_transforms(train=False)
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        # Loss and optimizer
        self.criterion = MultiTask3DLoss(cfg.LOSS_WEIGHTS)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.LR,
            weight_decay=cfg.WEIGHT_DECAY
        )
        
        self.best_val_loss = float('inf')
        os.makedirs(os.path.dirname(cfg.MODEL_PATH), exist_ok=True)
        
    def get_transforms(self, train=True):
        from torchvision import transforms
        
        if train:
            return transforms.Compose([
                transforms.Resize(self.cfg.IMAGE_SIZE),
                transforms.ColorJitter(0.3, 0.3, 0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.cfg.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, (images, box2d_queries, box3d_targets, intrinsics) in enumerate(progress_bar):
            images = images.to(self.device)
            box2d_queries = box2d_queries.to(self.device)
            box3d_targets = box3d_targets.to(self.device)
            intrinsics = intrinsics.to(self.device)
            
            self.optimizer.zero_grad()
            pred_boxes = self.model(images, box2d_queries, intrinsics)
            
            mask = (box2d_queries.sum(dim=-1) != 0)
            if mask.sum() > 0:
                total_loss, loss_dict = self.criterion(
                    pred_boxes[mask], 
                    box3d_targets[mask]
                )
                
                total_loss.backward()
                self.optimizer.step()
                
                progress_bar.set_postfix({
                    'loss': total_loss.item(),
                    'center': loss_dict['center'].item(),
                    'size': loss_dict['size'].item(),
                    'angle': loss_dict['angle'].item()
                })
    
    def validate(self):
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, box2d_queries, box3d_targets, intrinsics in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                box2d_queries = box2d_queries.to(self.device)
                box3d_targets = box3d_targets.to(self.device)
                intrinsics = intrinsics.to(self.device)
                
                pred_boxes = self.model(images, box2d_queries, intrinsics)
                
                mask = (box2d_queries.sum(dim=-1) != 0)
                if mask.sum() > 0:
                    loss, _ = self.criterion(pred_boxes[mask], box3d_targets[mask])
                    val_loss += loss.item()
        
        val_loss /= len(self.val_loader)
        return val_loss
    
    def train(self):
        print("Starting training...")
        
        for epoch in range(self.cfg.EPOCHS):
            self.current_epoch = epoch
            
            self.train_epoch()
            val_loss = self.validate()
            
            print(f"Epoch {epoch}: Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint()
    
    def save_checkpoint(self):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, self.cfg.MODEL_PATH)
        print(f"Checkpoint saved to {self.cfg.MODEL_PATH}")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from {checkpoint_path}")