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


class VideoEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        b, t, c, h, w = x.shape
        feat = self.cnn(x.reshape(b * t, c, h, w)).flatten(1)
        return self.fc(feat).reshape(b, t, -1)


class TemporalEncoder(nn.Module):
    def __init__(self, d_model=128, num_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None):
        # src_key_padding_mask: True 表示要被忽略
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


class StructuredBBoxDecoder(nn.Module):
    """重构 log(w,h)，由身高分量和透视分量共同决定。"""

    def __init__(self, height_dim=32, perspective_dim=16, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(height_dim + perspective_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, z_t_H, z_t_P):
        z = torch.cat([z_t_H, z_t_P], dim=-1)
        return self.net(z)


class HeightRankModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bbox_encoder = BBoxEncoder(cfg['bbox_dim'], cfg['d_model'])
        self.use_video = bool(cfg.get('use_video', False))
        self.video_encoder = None
        self.fusion = None
        if self.use_video:
            video_dim = cfg.get('video_dim', cfg['d_model'])
            self.video_encoder = VideoEncoder(video_dim)
            self.fusion = nn.Linear(cfg['d_model'] + video_dim, cfg['d_model'])
        self.temporal_encoder = TemporalEncoder(
            d_model=cfg['d_model'],
            num_layers=cfg['num_layers'],
            nhead=cfg.get('nhead', 4),
            dropout=cfg.get('dropout', 0.1),
        )
        self.height_head = HeightHead(cfg['d_model'], cfg['height_dim'])
        self.perspective_head = PerspectiveHead(cfg['d_model'], cfg['perspective_dim'])
        self.ranking_head = RankingHead(cfg['height_dim'])
        self.decoder = StructuredBBoxDecoder(
            height_dim=cfg['height_dim'],
            perspective_dim=cfg['perspective_dim'],
            hidden_dim=cfg.get('decoder_hidden_dim', cfg['d_model']),
        )

    def forward(self, bboxes, mask, video_frames=None):
        # 数值稳定化：对输入 bbox 做逐视频标准化，缓解大坐标尺度导致的爆炸
        b = torch.nan_to_num(bboxes, nan=0.0, posinf=1e6, neginf=-1e6).clone()
        b[..., 2:4] = b[..., 2:4].clamp_min(1e-6)
        if mask is not None:
            b = b * mask.unsqueeze(-1)

        xy = b[..., 0:2]
        wh = b[..., 2:4]

        # x,y 做幅值归一化
        xy_scale = xy.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
        xy_norm = xy / xy_scale

        # w,h 使用 log 后做标准化
        wh_log = torch.log(wh)
        wh_mean = wh_log.mean(dim=1, keepdim=True)
        wh_std = wh_log.std(dim=1, keepdim=True).clamp_min(1e-3)
        wh_norm = (wh_log - wh_mean) / wh_std

        bbox_feat = torch.cat([xy_norm, wh_norm], dim=-1)
        h = self.bbox_encoder(bbox_feat)
        video_feat = None
        if self.use_video:
            if video_frames is None:
                raise ValueError("video_frames must be provided when use_video=True")
            video_feat = self.video_encoder(video_frames)
            h = self.fusion(torch.cat([h, video_feat], dim=-1))
        h = self.temporal_encoder(h, mask)
        # z_t^H = g_H(h_t)
        z_t_H = self.height_head(h)
        z_t_P = self.perspective_head(h)
        recon_log_wh = self.decoder(z_t_H, z_t_P)

        # 视频级 z_v^H = masked mean(z_t^H)
        valid = mask.unsqueeze(-1)
        z_v_H = (z_t_H * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        s_v = self.ranking_head(z_v_H)
        s_t = self.ranking_head(z_t_H.reshape(-1, z_t_H.shape[-1])).reshape(z_t_H.shape[0], z_t_H.shape[1])

        return {
            'z_t_H': z_t_H,
            'z_t_P': z_t_P,
            'z_v_H': z_v_H,
            's_t': s_t,
            's_v': s_v,
            'recon_log_wh': recon_log_wh,
            'video_feat': video_feat,
        }
