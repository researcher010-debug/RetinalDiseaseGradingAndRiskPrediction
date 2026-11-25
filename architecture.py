
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def reset_random(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, num_classes=6, base_channels=32):
        super().__init__()
        self.num_classes = num_classes
        filters = [base_channels * (2 ** i) for i in range(5)]
        self.pool = nn.MaxPool2d(2, 2)
        self.nodes = nn.ModuleDict()
        for i in range(5):
            self.nodes[f"{i}_0"] = ConvBlock(in_channels if i == 0 else filters[i-1], filters[i])
        for j in range(1, 5):
            for i in range(5-j):
                self.nodes[f"{i}_{j}"] = ConvBlock(filters[i] * j + filters[i+1], filters[i])
        self.output = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        nodes = {}
        for i in range(5):
            source = x if i == 0 else self.pool(nodes[(i-1, 0)])
            nodes[(i, 0)] = self.nodes[f"{i}_0"](source)
        for j in range(1, 5):
            for i in range(5-j):
                up = F.interpolate(nodes[(i+1, j-1)], size=nodes[(i, 0)].shape[2:], mode="bilinear", align_corners=False)
                features = [nodes[(i, k)] for k in range(j)] + [up]
                nodes[(i, j)] = self.nodes[f"{i}_{j}"](torch.cat(features, dim=1))
        return self.output(nodes[(0, 4)])

class MambaVisionEncoder(nn.Module):
    def __init__(self, pretrained=False, model_name="mamba_vision_T"):
        super().__init__()
        try:
            import timm
            self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
            self.feature_channels = self.backbone.feature_info.channels()[-1]
        except Exception as error:
            raise RuntimeError(
                f"Unable to load {model_name} through timm. Install a compatible timm version "
                "and ensure the MambaVision model is registered."
            ) from error

    def forward(self, x):
        features = self.backbone(x)
        return F.adaptive_avg_pool2d(features[-1], 1).flatten(1)

class SegmentationFeatureEncoder(nn.Module):
    def __init__(self, num_classes=6, feature_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(num_classes, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d(1)
        )
        self.projection = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.area_projection = nn.Sequential(
            nn.Linear(num_classes - 1, feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, segmentation_maps):
        if segmentation_maps.ndim == 4:
            labels = torch.argmax(segmentation_maps, dim=1)
        elif segmentation_maps.ndim == 3:
            labels = segmentation_maps.long()
        else:
            raise ValueError("Segmentation maps must have shape (B,C,H,W) or (B,H,W)")
        num_classes = self.area_projection[0].in_features + 1
        one_hot = F.one_hot(labels.clamp(0, num_classes - 1), num_classes=num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()
        area = one_hot[:, 1:].mean(dim=(2, 3))
        features = self.encoder(one_hot).flatten(1)
        return self.projection(features), self.area_projection(area), area

class AttentionRefinement(nn.Module):
    def __init__(self, feature_dim=256, num_heads=4, dropout=0.2):
        super().__init__()
        self.attention = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, features):
        attended, _ = self.attention(features, features, features, need_weights=False)
        features = self.norm1(features + attended)
        return self.norm2(features + self.feed_forward(features))

class SGMODLF(nn.Module):
    def __init__(self, num_dr_classes=5, num_dme_classes=3, num_seg_classes=6, feature_dim=256, pretrained=False, backbone_name="mamba_vision_T", dropout=0.3):
        super().__init__()
        self.retinal_encoder = MambaVisionEncoder(pretrained=pretrained, model_name=backbone_name)
        self.retinal_projection = nn.Sequential(
            nn.Linear(self.retinal_encoder.feature_channels, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.segmentation_encoder = SegmentationFeatureEncoder(num_seg_classes, feature_dim)
        self.area_projection = nn.Sequential(
            nn.Linear(num_seg_classes - 1, feature_dim),
            nn.ReLU(inplace=True)
        )
        self.refinement = AttentionRefinement(feature_dim, num_heads=4, dropout=dropout)
        self.shared_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.dr_head = nn.Linear(feature_dim, num_dr_classes)
        self.dme_head = nn.Linear(feature_dim, num_dme_classes)

    def forward(self, images, segmentation_maps):
        retinal_features = self.retinal_projection(self.retinal_encoder(images))
        segmentation_features, _, lesion_area = self.segmentation_encoder(segmentation_maps)
        area_features = self.area_projection(lesion_area)
        tokens = torch.stack([retinal_features, segmentation_features, area_features], dim=1)
        refined = self.refinement(tokens).mean(dim=1)
        shared_features = self.shared_classifier(refined)
        return {
            "dr_logits": self.dr_head(shared_features),
            "dme_logits": self.dme_head(shared_features),
            "lesion_area": lesion_area
        }

class SGMODLFSegmentation(nn.Module):
    def __init__(self, in_channels=3, num_classes=6, base_channels=32):
        super().__init__()
        self.model = UNetPlusPlus(in_channels, num_classes, base_channels)

    def forward(self, images):
        return self.model(images)

def build_models(pretrained=False, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    segmentation_model = SGMODLFSegmentation().to(device)
    classification_model = SGMODLF(pretrained=pretrained).to(device)
    return segmentation_model, classification_model
