import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from model import ViT3DDetector
from model.loss import MultiTask3DLoss
from data.dataset import KITTIDataset, collate_fn  # 导入collate_fn

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("="*80)
        print(f"🚀 Initializing Trainer with EVA-02")
        print("="*80)
        print(f"Device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Model
        print(f"\\nLoading model...")
        self.model = ViT3DDetector(
            backbone=cfg.BACKBONE,
            hidden_dim=cfg.HIDDEN_DIM,
            num_queries=cfg.NUM_QUERIES,
            dropout=0.1
        ).to(self.device)
        
        print(f"✓ Model loaded on {self.device}")
        
        # Loss
        self.criterion = MultiTask3DLoss(weights=cfg.LOSS_WEIGHTS)
        print(f"✓ Loss function initialized")
        print(f"  Loss weights: {cfg.LOSS_WEIGHTS}")
        
        # Optimizer
        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': cfg.LR * 0.1},
            {'params': other_params, 'lr': cfg.LR}
        ], weight_decay=cfg.WEIGHT_DECAY)
        
        print(f"✓ Optimizer: AdamW")
        print(f"  Backbone LR: {cfg.LR * 0.1:.2e}")
        print(f"  Other LR: {cfg.LR:.2e}")
        
        # Data
        print(f"\\nLoading dataset from: {cfg.DATA_ROOT}")
        from torchvision import transforms
        
        train_transform = transforms.Compose([
            transforms.Resize(cfg.IMAGE_SIZE),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.train_dataset = KITTIDataset(
            cfg.DATA_ROOT, 
            split='train', 
            transform=train_transform
        )
        
        # 🔥 关键：使用自定义collate_fn
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=collate_fn  # 这里！
        )
        
        print(f"✓ Dataset loaded: {len(self.train_dataset)} samples")
        print(f"  Batch size: {cfg.BATCH_SIZE}")
        print(f"  Steps per epoch: {len(self.train_loader)}")
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[cfg.LR * 0.1, cfg.LR],
            epochs=cfg.EPOCHS,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        print(f"✓ Scheduler: OneCycleLR with warmup")
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        os.makedirs(os.path.dirname(cfg.MODEL_PATH), exist_ok=True)
        
        print("="*80)
        print("✅ Initialization complete!")
        print("="*80)
    
    def train(self):
        print(f"\\n{'='*80}")
        print(f"🔥 Starting Training")
        print(f"{'='*80}\\n")
        
        for epoch in range(self.current_epoch, self.cfg.EPOCHS):
            self.current_epoch = epoch
            self.model.train()
            
            epoch_losses = {'total': 0.0, 'xy': 0.0, 'depth': 0.0, 'size': 0.0, 'angle': 0.0, 'conf': 0.0}
            
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.cfg.EPOCHS}', ncols=120)
            
            # 🔥 关键：解包5个返回值（包括valid_mask）
            for batch_idx, (images, box2d, box3d, intrinsics, valid_mask) in enumerate(pbar):
                try:
                    images = images.to(self.device)
                    box2d = box2d.to(self.device)
                    box3d = box3d.to(self.device)
                    intrinsics = intrinsics.to(self.device)
                    valid_mask = valid_mask.to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    # Forward
                    pred_boxes, pred_conf = self.model(images, box2d, intrinsics)
                    
                    # 只对有效对象计算损失
                    if valid_mask.sum() > 0:
                        # Flatten batch和object维度
                        valid_pred = pred_boxes[valid_mask]
                        valid_gt = box3d[valid_mask]
                        valid_conf = pred_conf[valid_mask]
                        
                        loss, loss_dict = self.criterion(
                            valid_pred.unsqueeze(0),  # 添加batch维度
                            valid_gt.unsqueeze(0), 
                            valid_conf.unsqueeze(0)
                        )
                        
                        if torch.isfinite(loss):
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.optimizer.step()
                            self.scheduler.step()
                            
                            for key in epoch_losses.keys():
                                if key in loss_dict:
                                    epoch_losses[key] += loss_dict[key].item()
                            
                            pbar.set_postfix({
                                'loss': f'{loss.item():.4f}',
                                'depth': f'{loss_dict["depth"].item():.4f}',
                                'lr': f'{self.optimizer.param_groups[1]["lr"]:.2e}'
                            })
                    
                    self.global_step += 1
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f"\\n❌ GPU OOM! Reduce batch_size")
                        torch.cuda.empty_cache()
                        raise e
                    else:
                        raise e
            
            # Epoch summary
            steps = len(self.train_loader)
            avg_losses = {k: v / steps if steps > 0 else 0 for k, v in epoch_losses.items()}
            
            print(f"\\nEpoch {epoch+1}: Loss={avg_losses['total']:.4f}, Depth={avg_losses['depth']:.4f}, Angle={avg_losses['angle']:.4f}")
            
            if avg_losses['total'] < self.best_loss and avg_losses['total'] > 0:
                self.best_loss = avg_losses['total']
                self.save_checkpoint()
                print(f"✅ Best model saved! Loss: {self.best_loss:.4f}")
            
            # 每10个epoch保存一次
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        print(f"\\n🎉 Training Complete! Best loss: {self.best_loss:.4f}")
    
    def save_checkpoint(self, filename=None):
        save_path = filename if filename else self.cfg.MODEL_PATH
        if filename and not os.path.isabs(filename):
            save_path = os.path.join(os.path.dirname(self.cfg.MODEL_PATH), filename)
        
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
        }, save_path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0) + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"✓ Loaded from epoch {self.current_epoch - 1}")