from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=False)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = F.relu(out + identity, inplace=False)
        return out


class PairComparatorHead(nn.Module):
    def __init__(
        self,
        feat_ch: int = 16,
        comparator_type: str = "conv",
        comparator_layers: int = 2,
        comparator_num_heads: int = 4,
        comparator_patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.feat_ch = int(feat_ch)
        self.comparator_type = str(comparator_type).lower()
        self.comparator_layers = int(comparator_layers)
        self.comparator_num_heads = int(comparator_num_heads)
        self.comparator_patch_size = int(comparator_patch_size)

        if self.comparator_type == "conv":
            self.embed = self._build_conv_encoder()
        elif self.comparator_type == "resnet":
            self.embed = self._build_resnet_encoder()
        elif self.comparator_type == "vit":
            self.patch_embed = nn.Conv2d(
                1,
                self.feat_ch,
                kernel_size=self.comparator_patch_size,
                stride=self.comparator_patch_size,
                bias=False,
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.feat_ch,
                nhead=max(1, self.comparator_num_heads),
                dim_feedforward=self.feat_ch * 4,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
            )
            self.vit_encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, self.comparator_layers))
            self.vit_norm = nn.LayerNorm(self.feat_ch)
        else:
            raise ValueError(f"unsupported comparator_type: {self.comparator_type}")

        self.cls = nn.Sequential(
            nn.Linear(self.feat_ch * 3, self.feat_ch),
            nn.ReLU(inplace=False),
            nn.Linear(self.feat_ch, 1),
        )

    def _build_conv_encoder(self) -> nn.Module:
        layers: list[nn.Module] = []
        in_ch = 1
        for _ in range(max(1, self.comparator_layers)):
            layers.extend(
                [
                    nn.Conv2d(in_ch, self.feat_ch, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(self.feat_ch),
                    nn.ReLU(inplace=False),
                ]
            )
            in_ch = self.feat_ch
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*layers)

    def _build_resnet_encoder(self) -> nn.Module:
        layers: list[nn.Module] = [
            nn.Conv2d(1, self.feat_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.feat_ch),
            nn.ReLU(inplace=False),
        ]
        for idx in range(max(1, self.comparator_layers)):
            layers.append(ResidualBlock(self.feat_ch, stride=2 if idx == 0 else 1))
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*layers)

    def encode(self, fg_height: torch.Tensor) -> torch.Tensor:
        if self.comparator_type == "vit":
            x = self.patch_embed(fg_height)
            b, c, h, w = x.shape
            if h == 0 or w == 0:
                raise ValueError(
                    f"vit comparator patch size {self.comparator_patch_size} is too large for feature map {tuple(fg_height.shape)}"
                )
            tokens = x.flatten(2).transpose(1, 2)
            tokens = self.vit_encoder(tokens)
            tokens = self.vit_norm(tokens)
            return tokens.mean(dim=1)

        feat = self.embed(fg_height)
        return feat.flatten(1)

    def compare_encoded(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([fa, fb, fa - fb], dim=1)
        return self.cls(x).squeeze(1)

    def forward(self, fg_a: torch.Tensor, fg_b: torch.Tensor) -> torch.Tensor:
        fa = self.encode(fg_a)
        fb = self.encode(fg_b)
        return self.compare_encoded(fa, fb)


class HeightNetTiny(nn.Module):
    """Encoder-decoder for height map + pairwise comparator head."""

    def __init__(
        self,
        base_channels: int = 32,
        comparator_channels: int = 16,
        comparator_type: str = "conv",
        comparator_layers: int = 2,
        comparator_num_heads: int = 4,
        comparator_patch_size: int = 16,
    ) -> None:
        super().__init__()
        b = base_channels

        self.enc1 = ConvBlock(3, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(b * 4, b * 8)

        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(b * 8, b * 4)

        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2)

        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(b * 2, b)

        self.height_head = nn.Conv2d(b, 1, kernel_size=1)
        self.pair_head = PairComparatorHead(
            feat_ch=comparator_channels,
            comparator_type=comparator_type,
            comparator_layers=comparator_layers,
            comparator_num_heads=comparator_num_heads,
            comparator_patch_size=comparator_patch_size,
        )

    def predict_height(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bt = self.bottleneck(self.pool(e3))

        d3 = self.up3(bt)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return F.softplus(self.height_head(d1))

    def compare_pair(self, height_a: torch.Tensor, mask_a: torch.Tensor, height_b: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
        fa = self.encode_person(height_a, mask_a)
        fb = self.encode_person(height_b, mask_b)
        return self.compare_encoded(fa, fb)

    def encode_person(self, height: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        fg = height * (mask > 0.5).float()
        return self.pair_head.encode(fg)

    def compare_encoded(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        return self.pair_head.compare_encoded(fa, fb)

    def forward(
        self,
        image: torch.Tensor | None = None,
        pair_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> dict:
        pred_height_map = self.predict_height(image) if image is not None else None
        out = {"pred_height_map": pred_height_map, "pair_logit": None}
        if pair_inputs is not None:
            if len(pair_inputs) == 4:
                h_a, m_a, h_b, m_b = pair_inputs
            else:
                if pred_height_map is None:
                    raise ValueError("pred_height_map is required when pair_inputs is index-based")
                idx_i, idx_j, person_mask = pair_inputs
                source = pred_height_map[: person_mask.shape[0]]
                h_a = source.index_select(0, idx_i)
                m_a = person_mask.index_select(0, idx_i)
                h_b = source.index_select(0, idx_j)
                m_b = person_mask.index_select(0, idx_j)
            out["pair_logit"] = self.compare_pair(h_a, m_a, h_b, m_b)
        return out
