import os
import torch

from fashn_vton import TryOnPipeline


class VasundharaVTON:
    """
    VASUNDHARA VTON V7 foundation engine.

    Current foundation:
        FASHN VTON v1.5

    Future:
        VASUNDHARA-trained weights
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.weights_dir = os.environ.get(
            "VTON_WEIGHTS_DIR",
            "/workspace/weights",
        )

        print("==============================================")
        print("Loading VASUNDHARA VTON")
        print("Foundation: FASHN VTON v1.5")
        print(f"Device: {self.device}")
        print(f"Weights: {self.weights_dir}")
        print("==============================================")

        self.pipeline = TryOnPipeline(
            weights_dir=self.weights_dir,
            device=self.device,
        )

        print("VASUNDHARA VTON engine loaded.")

    def generate(
        self,
        person_image,
        garment_image,
        category="one-pieces",
    ):
        result = self.pipeline(
            person_image=person_image,
            garment_image=garment_image,
            category=category,
            garment_photo_type="flat-lay",
            num_samples=1,
            num_timesteps=30,
            guidance_scale=1.5,
            seed=42,
            segmentation_free=True,
        )

        return result.images[0]

    def info(self):
        return {
            "name": "VASUNDHARA VTON",
            "version": "V7",
            "foundation": "FASHN VTON v1.5",
            "device": self.device,
        }
