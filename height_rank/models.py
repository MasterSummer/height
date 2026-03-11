import torch
import torch.nn as nn

class BBoxEncoder(nn.Module):
    def __init__(self, bbox_dim=4, d_model=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(bbox_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    def forward(self, x):
        return self.mlp(x)

class TemporalEncoder(nn.Module):
    def __init__(self, d_model=128, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, x, mask=None):
        if mask is not None:
            mask = ~mask.bool()
        return self.encoder(x, src_key_padding_mask=mask)

class HeightHead(nn.Module):
    def __init__(self, d_model=128, height_dim=32):
        super().__init__()
        self.fc = nn.Linear(d_model, height_dim)
    def forward(self, h):
        return self.fc(h)

class PerspectiveHead(nn.Module):
    def __init__(self, d_model=128, perspective_dim=16):
        super().__init__()
        self.fc = nn.Linear(d_model, perspective_dim)
    def forward(self, h):
        return self.fc(h)

class RankingHead(nn.Module):
    def __init__(self, height_dim=32):
        super().__init__()
        self.fc = nn.Linear(height_dim, 1)
    def forward(self, z_v_H):
        return self.fc(z_v_H).squeeze(-1)

class HeightRankModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bbox_encoder = BBoxEncoder(cfg['bbox_dim'], cfg['d_model'])
        self.temporal_encoder = TemporalEncoder(cfg['d_model'], cfg['num_layers'])
        self.height_head = HeightHead(cfg['d_model'], cfg['height_dim'])
        self.perspective_head = PerspectiveHead(cfg['d_model'], cfg['perspective_dim'])
        self.ranking_head = RankingHead(cfg['height_dim'])
    def forward(self, bboxes, mask):
        h = self.bbox_encoder(bboxes)
        h = self.temporal_encoder(h, mask)
        z_t_H = self.height_head(h)
        z_t_P = self.perspective_head(h)
        # 视频级身高表征
        masked_z_t_H = z_t_H * mask.unsqueeze(-1)
        z_v_H = masked_z_t_H.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        s_v = self.ranking_head(z_v_H)
        return {
            'z_t_H': z_t_H,
            'z_t_P': z_t_P,
            'z_v_H': z_v_H,
            's_v': s_v
        }
