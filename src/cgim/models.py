from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def _resnet18(pretrained_backbone=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
    return models.resnet18(weights=weights)


class SimCLRNetWithHead(nn.Module):
    TEACHER_EMBEDDING_DIM = 512

    def __init__(self, num_classes=2, embedding_dim=128, img_channels=3,
                 image_size=(224, 224), device='cpu', pretrained_backbone=True):
        super().__init__()
        self.device = device
        self.img_channels = img_channels
        self.image_size = image_size
        self.num_classes = num_classes
        resnet = _resnet18(pretrained_backbone)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
        self.classifier = nn.Linear(self.TEACHER_EMBEDDING_DIM, num_classes)

    def save_encoder(self, filepath: str):
        torch.save(self.encoder.state_dict(), filepath)
        print(f"[save_encoder] saved to {filepath}")

    def load_encoder(self, filepath: str, map_location=None, device=None, **_):
        loaded = torch.load(filepath, map_location=map_location)
        if isinstance(loaded, dict) and 'encoder_state_dict' in loaded:
            loaded = loaded['encoder_state_dict']
        elif isinstance(loaded, dict) and 'state_dict' in loaded:
            loaded = loaded['state_dict']
        self.encoder.load_state_dict(loaded)
        self.classifier = nn.Linear(self.TEACHER_EMBEDDING_DIM, self.num_classes).to(
            device or self.device
        )
        print("[load_encoder] loaded encoder state_dict (exact).")

    def forward(self, x):
        h_raw = self.encoder(x)
        if h_raw.ndim != 2:
            h_raw = h_raw.view(h_raw.size(0), -1)
        h512 = h_raw
        logits = self.classifier(h512)
        return h_raw, h512, logits


class SimCLRNetWithHeadPre(nn.Module):
    def __init__(self, embedding_dim=128, img_channels=3, regression_dim=64,
                 pretrained_backbone=True):
        super().__init__()
        resnet = _resnet18(pretrained_backbone)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
        self.classifier1 = nn.Sequential(nn.Linear(embedding_dim, 10))
        self.classifier2 = nn.Sequential(nn.Linear(embedding_dim, 100))
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256 * 7 * 7),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16, img_channels, kernel_size=4, stride=2, padding=1),
        )
        self.regression_head = nn.Linear(embedding_dim, regression_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = h
        logits1 = self.classifier1(z)
        logits2 = self.classifier2(z)
        regression_pred = self.regression_head(z)
        x_recon = self.decoder(z)
        return h, z, logits1, logits2, x_recon, regression_pred

    def save_encoder(self, filepath: str):
        torch.save(self.encoder.state_dict(), filepath)
        print(f"Encoder weights saved to {filepath}")

    def load_encoder(self, filepath: str, map_location=None):
        state_dict = torch.load(filepath, map_location=map_location)
        self.encoder.load_state_dict(state_dict)
        print(f"Encoder weights loaded from {filepath}")

