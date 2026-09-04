# vton/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# VASUNDHARA VTON
# Version 7
#
# Own trainable VTON architecture
#
# Inputs:
#   person         : RGB person image
#   garment        : RGB garment image
#   clothing_mask  : clothing/body-region mask
#   face_mask      : face-preservation mask
#
# Output:
#   RGB try-on image
# =========================================================


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.residual = ResidualBlock(out_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, 2, 1)

    def forward(self, x):
        feature = self.residual(self.conv(x))
        return self.downsample(feature), feature


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)
        self.residual = ResidualBlock(out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.residual(self.conv(x))


class GarmentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = DownBlock(3, 64)
        self.block2 = DownBlock(64, 128)
        self.block3 = DownBlock(128, 256)

    def forward(self, garment):
        x, feature1 = self.block1(garment)
        x, feature2 = self.block2(x)
        x, feature3 = self.block3(x)
        return x, feature1, feature2, feature3


class PersonEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = DownBlock(5, 64)
        self.block2 = DownBlock(64, 128)
        self.block3 = DownBlock(128, 256)

    def forward(self, person, clothing_mask, face_mask):
        x = torch.cat([person, clothing_mask, face_mask], dim=1)
        x, feature1 = self.block1(x)
        x, feature2 = self.block2(x)
        x, feature3 = self.block3(x)
        return x, feature1, feature2, feature3


class FeatureFusion(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            ResidualBlock(channels),
            ResidualBlock(channels),
        )

    def forward(self, person_features, garment_features):
        return self.fusion(torch.cat([person_features, garment_features], dim=1))


class FacePreservation(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.process = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, face_mask, target_size):
        mask = F.interpolate(face_mask, size=target_size, mode="bilinear", align_corners=False)
        return self.process(mask)


class VasundharaVTON(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.person_encoder = PersonEncoder()
        self.garment_encoder = GarmentEncoder()
        self.fusion = FeatureFusion(channels=256)
        self.face_preservation = FacePreservation(channels=256)
        self.middle = nn.Sequential(
            ResidualBlock(256), ResidualBlock(256), ResidualBlock(256), ResidualBlock(256)
        )
        self.up3 = UpBlock(256, 256 + 256, 256)
        self.up2 = UpBlock(256, 128 + 128, 128)
        self.up1 = UpBlock(128, 64 + 64, 64)
        self.output_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 3, 1),
            nn.Tanh(),
        )

    def forward(self, person, garment, clothing_mask, face_mask):
        person_low, person_f1, person_f2, person_f3 = self.person_encoder(
            person, clothing_mask, face_mask
        )
        garment_low, garment_f1, garment_f2, garment_f3 = self.garment_encoder(garment)
        x = self.fusion(person_low, garment_low)
        face_features = self.face_preservation(face_mask, x.shape[-2:])
        x = self.middle(x * (1.0 + face_features))

        x = self.up3(x, torch.cat([person_f3, garment_f3], dim=1))
        x = self.up2(x, torch.cat([person_f2, garment_f2], dim=1))
        x = self.up1(x, torch.cat([person_f1, garment_f1], dim=1))
        output = self.output_head(x)

        if face_mask is not None:
            protected_face = F.interpolate(
                face_mask, size=output.shape[-2:], mode="bilinear", align_corners=False
            ).clamp(0.0, 1.0)
            output = output * (1.0 - protected_face) + person * protected_face
        return output


def create_model(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return VasundharaVTON().to(device)


def load_checkpoint(model, checkpoint_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            raise ValueError("Checkpoint does not contain model_state_dict or model")
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def save_checkpoint(model, checkpoint_path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "VASUNDHARA-VTON",
            "version": "V7",
        },
        checkpoint_path,
    )
