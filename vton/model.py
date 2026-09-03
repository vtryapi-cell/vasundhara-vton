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


# ---------------------------------------------------------
# Basic convolution block
# ---------------------------------------------------------

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                num_groups=8,
                num_channels=out_channels,
            ),

            nn.SiLU(),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                num_groups=8,
                num_channels=out_channels,
            ),

            nn.SiLU(),
        )

    def forward(self, x):

        return self.block(x)


# ---------------------------------------------------------
# Residual block
# ---------------------------------------------------------

class ResidualBlock(nn.Module):

    def __init__(
        self,
        channels,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.norm1 = nn.GroupNorm(
            8,
            channels,
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.norm2 = nn.GroupNorm(
            8,
            channels,
        )

        self.activation = nn.SiLU()

    def forward(self, x):

        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + residual

        x = self.activation(x)

        return x


# ---------------------------------------------------------
# Downsampling block
# ---------------------------------------------------------

class DownBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.conv = ConvBlock(
            in_channels,
            out_channels,
        )

        self.residual = ResidualBlock(
            out_channels,
        )

        self.downsample = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x):

        feature = self.conv(x)

        feature = self.residual(
            feature
        )

        down = self.downsample(
            feature
        )

        return down, feature


# ---------------------------------------------------------
# Upsampling block
# ---------------------------------------------------------

class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.conv = ConvBlock(
            out_channels + skip_channels,
            out_channels,
        )

        self.residual = ResidualBlock(
            out_channels,
        )

    def forward(
        self,
        x,
        skip,
    ):

        x = self.up(x)

        # Handle possible resolution mismatch.
        if x.shape[-2:] != skip.shape[-2:]:

            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat(
            [
                x,
                skip,
            ],
            dim=1,
        )

        x = self.conv(x)

        x = self.residual(x)

        return x


# ---------------------------------------------------------
# Garment encoder
# ---------------------------------------------------------

class GarmentEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.block1 = DownBlock(
            3,
            64,
        )

        self.block2 = DownBlock(
            64,
            128,
        )

        self.block3 = DownBlock(
            128,
            256,
        )

    def forward(self, garment):

        x, feature1 = self.block1(
            garment
        )

        x, feature2 = self.block2(
            x
        )

        x, feature3 = self.block3(
            x
        )

        return (
            x,
            feature1,
            feature2,
            feature3,
        )


# ---------------------------------------------------------
# Person encoder
# ---------------------------------------------------------

class PersonEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        # Person RGB
        # Clothing mask
        # Face mask
        #
        # 3 + 1 + 1 = 5 channels

        self.block1 = DownBlock(
            5,
            64,
        )

        self.block2 = DownBlock(
            64,
            128,
        )

        self.block3 = DownBlock(
            128,
            256,
        )

    def forward(
        self,
        person,
        clothing_mask,
        face_mask,
    ):

        x = torch.cat(
            [
                person,
                clothing_mask,
                face_mask,
            ],
            dim=1,
        )

        x, feature1 = self.block1(
            x
        )

        x, feature2 = self.block2(
            x
        )

        x, feature3 = self.block3(
            x
        )

        return (
            x,
            feature1,
            feature2,
            feature3,
        )


# ---------------------------------------------------------
# Feature fusion
# ---------------------------------------------------------

class FeatureFusion(nn.Module):

    def __init__(
        self,
        channels=256,
    ):
        super().__init__()

        self.fusion = nn.Sequential(

            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1,
                bias=False,
            ),

            nn.GroupNorm(
                8,
                channels,
            ),

            nn.SiLU(),

            ResidualBlock(
                channels
            ),

            ResidualBlock(
                channels
            ),
        )

    def forward(
        self,
        person_features,
        garment_features,
    ):

        x = torch.cat(
            [
                person_features,
                garment_features,
            ],
            dim=1,
        )

        return self.fusion(x)


# ---------------------------------------------------------
# Face preservation module
# ---------------------------------------------------------

class FacePreservation(nn.Module):

    def __init__(
        self,
        channels=256,
    ):
        super().__init__()

        self.process = nn.Sequential(

            nn.Conv2d(
                1,
                channels,
                kernel_size=3,
                padding=1,
            ),

            nn.GroupNorm(
                8,
                channels,
            ),

            nn.SiLU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),

            nn.Sigmoid(),
        )

    def forward(
        self,
        face_mask,
        target_size,
    ):

        mask = F.interpolate(
            face_mask,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        return self.process(mask)


# ---------------------------------------------------------
# VASUNDHARA VTON
# ---------------------------------------------------------

class VasundharaVTON(nn.Module):

    def __init__(
        self,
        base_channels=64,
    ):

        super().__init__()

        # =================================================
        # Encoders
        # =================================================

        self.person_encoder = PersonEncoder()

        self.garment_encoder = GarmentEncoder()

        # =================================================
        # Fusion
        # =================================================

        self.fusion = FeatureFusion(
            channels=256
        )

        # =================================================
        # Face preservation
        # =================================================

        self.face_preservation = (
            FacePreservation(
                channels=256
            )
        )

        # =================================================
        # Middle processing
        # =================================================

        self.middle = nn.Sequential(

            ResidualBlock(
                256
            ),

            ResidualBlock(
                256
            ),

            ResidualBlock(
                256
            ),

            ResidualBlock(
                256
            ),
        )

        # =================================================
        # Decoder
        # =================================================

        self.up3 = UpBlock(
            256,
            256 + 256,
            256,
        )

        self.up2 = UpBlock(
            256,
            128 + 128,
            128,
        )

        self.up1 = UpBlock(
            128,
            64 + 64,
            64,
        )

        # =================================================
        # Final output
        # =================================================

        self.output_head = nn.Sequential(

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1,
            ),

            nn.GroupNorm(
                8,
                32,
            ),

            nn.SiLU(),

            nn.Conv2d(
                32,
                3,
                kernel_size=1,
            ),

            nn.Tanh(),
        )

    # =====================================================
    # Forward
    # =====================================================

    def forward(
        self,
        person,
        garment,
        clothing_mask,
        face_mask,
    ):

        # -------------------------------------------------
        # Person features
        # -------------------------------------------------

        (
            person_low,
            person_f1,
            person_f2,
            person_f3,
        ) = self.person_encoder(
            person,
            clothing_mask,
            face_mask,
        )

        # -------------------------------------------------
        # Garment features
        # -------------------------------------------------

        (
            garment_low,
            garment_f1,
            garment_f2,
            garment_f3,
        ) = self.garment_encoder(
            garment
        )

        # -------------------------------------------------
        # Person + garment fusion
        # -------------------------------------------------

        x = self.fusion(
            person_low,
            garment_low,
        )

        # -------------------------------------------------
        # Face preservation conditioning
        # -------------------------------------------------

        face_features = (
            self.face_preservation(
                face_mask,
                x.shape[-2:],
            )
        )

        # Combine face information.
        x = x * (
            1.0 + face_features
        )

        # -------------------------------------------------
        # Middle network
        # -------------------------------------------------

        x = self.middle(x)

        # -------------------------------------------------
        # Decoder
        # -------------------------------------------------

        skip3 = torch.cat(
            [
                person_f3,
                garment_f3,
            ],
            dim=1,
        )

        x = self.up3(
            x,
            skip3,
        )

        skip2 = torch.cat(
            [
                person_f2,
                garment_f2,
            ],
            dim=1,
        )

        x = self.up2(
            x,
            skip2,
        )

        skip1 = torch.cat(
            [
                person_f1,
                garment_f1,
            ],
            dim=1,
        )

        x = self.up1(
            x,
            skip1,
        )

        # -------------------------------------------------
        # Generated image
        # -------------------------------------------------

        output = self.output_head(x)

        # -------------------------------------------------
        # Identity preservation
        #
        # Blend the original person image into regions
        # protected by the face mask.
        # -------------------------------------------------

        if face_mask is not None:

            protected_face = F.interpolate(
                face_mask,
                size=output.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            protected_face = protected_face.clamp(
                0.0,
                1.0,
            )

            output = (
                output * (1.0 - protected_face)
                +
                person * protected_face
            )

        return output


# =========================================================
# Model creation
# =========================================================

def create_model(
    device=None,
):

    if device is None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = VasundharaVTON()

    model = model.to(
        device
    )

    return model


# =========================================================
# Checkpoint loading
# =========================================================

def load_checkpoint(
    model,
    checkpoint_path,
    device=None,
):

    if device is None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if isinstance(
        checkpoint,
        dict,
    ) and "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(
        device
    )

    model.eval()

    return model


# =========================================================
# Save checkpoint
# =========================================================

def save_checkpoint(
    model,
    checkpoint_path,
):

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "VASUNDHARA-VTON",
            "version": "V7",
        },
        checkpoint_path,
    )
