import torch
import torch.nn as nn


# =========================================================
# VASUNDHARA VTON V7
# Model Interface
# =========================================================


class VasundharaVTON(nn.Module):
    """
    VASUNDHARA VTON model interface.

    This class is the place where our trained VTON
    architecture and weights will be connected.
    """

    def __init__(self):
        super().__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.initialized = True

    def forward(self, person, garment):
        """
        Main VTON forward pass.

        The trained VTON network will be connected here.
        """

        raise NotImplementedError(
            "VASUNDHARA VTON model weights/network "
            "have not been connected yet."
        )

    def info(self):
        return {
            "name": "VASUNDHARA VTON",
            "version": "V7",
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
        }
