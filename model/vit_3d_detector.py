import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

class ViT3DDetector(nn.Module):
    def __init__(self, backbone='vit_base_patch16_224', num_queries=100, hidden_dim=256):
        super().__init__()
        self.backbone = create_model(backbone, pretrained=True, num_classes=0, global_pool='')
        self.feature_dim = 768 if 'base' in backbone else 384
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Conv2d(self.feature_dim, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.box2d_encoder = nn.Linear(4, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=2048, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.bbox_center = nn.Linear(hidden_dim, 3)
        self.bbox_size = nn.Linear(hidden_dim, 3)
        self.bbox_angle = nn.Linear(hidden_dim, 1)
        self.intrinsic_proj = nn.Linear(4, hidden_dim)
        self.num_queries = num_queries
    
    def forward(self, images, box2d_queries, intrinsic_params):
        B, N = box2d_queries.shape[:2]
        features = self.backbone.forward_features(images)
        features = self.input_proj(features)
        features = features.flatten(2).permute(2, 0, 1)
        query_embed = self.query_embed.weight[:N].unsqueeze(1).repeat(1, B, 1)
        box2d_embed = F.relu(self.box2d_encoder(box2d_queries.view(B*N, 4))).view(N, B, self.hidden_dim)
        query_embed = query_embed + box2d_embed
        intrinsic_embed = self.intrinsic_proj(intrinsic_params).unsqueeze(0).repeat(N, 1, 1)
        query_embed = query_embed + intrinsic_embed
        hs = self.decoder(query_embed, features).transpose(0, 1)
        center = self.bbox_center(hs)
        size = F.relu(self.bbox_size(hs))
        angle = self.bbox_angle(hs)
        return torch.cat([center, size, angle], dim=-1)
