import torch

from .model import VasundharaVTON
from .preprocessing import prepare_inputs


# =========================================================
# VASUNDHARA VTON V7
# Inference
# =========================================================


class VTONInference:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = VasundharaVTON().to(self.device)
        self.model.eval()

    def generate(self, person_image, garment_image):
        """
        Run VASUNDHARA VTON inference.

        The trained VTON network will be connected here.
        """

        inputs = prepare_inputs(
            person_image,
            garment_image,
        )

        person = inputs["person"]
        garment = inputs["garment"]

        # -------------------------------------------------
        # The actual trained VTON generation will be
        # connected here.
        # -------------------------------------------------

        raise NotImplementedError(
            "VASUNDHARA VTON trained inference is not "
            "connected yet."
        )

    def info(self):
        return {
            "name": "VASUNDHARA VTON",
            "version": "V7",
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
        }
