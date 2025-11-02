import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
import math

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
    def forward(self, x):
        B, C, H, W = x.shape
        pos_h = torch.arange(H, device=x.device).float().unsqueeze(1).repeat(1, W)
        pos_w = torch.arange(W, device=x.device).float().unsqueeze(0).repeat(H, 1)
        div_term = torch.exp(torch.arange(0, C, 2, device=x.device).float() * -(math.log(10000.0) / C))
        pe = torch.zeros(C, H, W, device=x.device)
        pe[0::2] = torch.sin(pos_h * div_term[:C//2].unsqueeze(-1).unsqueeze(-1))
        pe[1::2] = torch.cos(pos_w * div_term[:C//2].unsqueeze(-1).unsqueeze(-1))
        return x + pe.unsqueeze(0)

class ViT3DDetector(nn.Module):
    def __init__(self, backbone='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', 
                 num_queries=100, hidden_dim=384, dropout=0.1):
        super().__init__()
        
        print(f"🔥 Loading EVA-02 Large (最强backbone!)")
        print(f"   Model: {backbone}")
        print(f"   This may take a few minutes to download...")
        
        # 加载EVA-02
        self.backbone = create_model(
            backbone, 
            pretrained=True, 
            num_classes=0, 
            global_pool='',
            img_size=448
        )
        
        # EVA-02 Large特征维度是1024
        self.feature_dim = 1024
        self.hidden_dim = hidden_dim
        self.backbone_name = backbone
        
        print(f"✓ Backbone loaded! Feature dim: {self.feature_dim}")
        
        # 特征投影 - 从1024降到hidden_dim
        self.input_proj = nn.Sequential(
            nn.Conv2d(self.feature_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding2D(hidden_dim)
        
        # Query embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # 深层2D box编码器
        self.box2d_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 相机内参编码器
        self.intrinsic_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Transformer Decoder - 增加到12层以匹配强大的backbone
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,  # 更大的FFN
            dropout=dropout,
            activation='gelu',
            batch_first=False,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=12)
        
        # 更深的预测头
        self.bbox_center = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        self.bbox_size = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        self.bbox_angle = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 2)
        )
        
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.num_queries = num_queries
        self._init_weights()
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✓ Total parameters: {total_params/1e6:.2f}M")
        print(f"✓ Trainable parameters: {trainable_params/1e6:.2f}M")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, images, box2d_queries, intrinsic_params):
        B, N = box2d_queries.shape[:2]
        
        # 提取EVA-02特征
        features = self.backbone.forward_features(images)
        
        # EVA-02输出格式处理
        if features.dim() == 3:  # [B, L, C]
            # EVA-02的输出包含CLS token，需要移除
            if features.shape[1] > 1024:  # 32*32=1024 patches for 448x448
                features = features[:, 1:, :]  # 移除CLS token
            
            B_feat, num_patches, C = features.shape
            h = w = int(num_patches ** 0.5)
            features = features.permute(0, 2, 1).reshape(B_feat, C, h, w)
        
        # 投影到hidden_dim
        features = self.input_proj(features)
        features = self.pos_encoding(features)
        
        # Flatten for transformer
        features = features.flatten(2).permute(2, 0, 1)  # [HW, B, hidden_dim]
        
        # 准备queries
        query_embed = self.query_embed.weight[:N].unsqueeze(1).repeat(1, B, 1)
        
        # 编码2D box
        box2d_flat = box2d_queries.view(B * N, 4)
        box2d_embed = self.box2d_encoder(box2d_flat).view(N, B, self.hidden_dim)
        
        # 编码相机内参
        intrinsic_embed = self.intrinsic_proj(intrinsic_params)
        intrinsic_embed = intrinsic_embed.unsqueeze(0).repeat(N, 1, 1)
        
        # 组合queries
        query_embed = query_embed + box2d_embed + 0.3 * intrinsic_embed
        
        # Transformer解码
        hs = self.decoder(query_embed, features)
        hs = hs.transpose(0, 1)  # [B, N, hidden_dim]
        
        # 预测
        center = self.bbox_center(hs)
        
        size_logit = self.bbox_size(hs)
        size = torch.exp(size_logit).clamp(min=0.1, max=20.0)
        
        angle_pred = self.bbox_angle(hs)
        angle_pred = F.normalize(angle_pred, dim=-1)
        angle = torch.atan2(angle_pred[..., 0:1], angle_pred[..., 1:2])
        
        conf = self.confidence(hs)
        
        pred_boxes = torch.cat([center, size, angle], dim=-1)
        
        return pred_boxes, conf