import base64
import io
import traceback

import runpod
import torch
from PIL import Image

from vton.model import create_model


MODEL = None


def load_model():
    global MODEL

    if MODEL is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print("========================================")
        print("Loading VASUNDHARA VTON")
        print("Version: V7")
        print(f"Device: {device}")
        print("========================================")

        MODEL = create_model(device)

        MODEL.eval()

        print("VASUNDHARA model loaded.")

    return MODEL


def decode_image(value):

    if not value:
        raise ValueError("Image is missing")

    if value.startswith("data:image"):
        value = value.split(",", 1)[1]

    data = base64.b64decode(value)

    image = Image.open(
        io.BytesIO(data)
    ).convert("RGB")

    return image


def encode_image(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=95,
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


def handler(job):

    try:

        data = job.get("input", {})

        # --------------------------------------------
        # Worker health test
        # --------------------------------------------

        if data.get("test") is True:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            return {
                "status": "ok",
                "service": "VASUNDHARA VTON",
                "version": "V7",
                "device": device,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            }

        # --------------------------------------------
        # Load model
        # --------------------------------------------

        model = load_model()

        # --------------------------------------------
        # Images
        # --------------------------------------------

        person_data = data.get(
            "person_image"
        )

        garment_data = data.get(
            "garment_image"
        )

        if not person_data:
            raise ValueError(
                "person_image is required"
            )

        if not garment_data:
            raise ValueError(
                "garment_image is required"
            )

        person = decode_image(
            person_data
        )

        garment = decode_image(
            garment_data
        )

        print(
            f"Person: {person.size}"
        )

        print(
            f"Garment: {garment.size}"
        )

        # --------------------------------------------
        # Resize for model
        # --------------------------------------------

        size = (512, 768)

        person = person.resize(
            size,
            Image.Resampling.LANCZOS,
        )

        garment = garment.resize(
            size,
            Image.Resampling.LANCZOS,
        )

        # --------------------------------------------
        # Tensor conversion
        # --------------------------------------------

        person_tensor = (
            torch.from_numpy(
                __import__("numpy")
                .array(person)
            )
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        garment_tensor = (
            torch.from_numpy(
                __import__("numpy")
                .array(garment)
            )
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        # Normalize
        person_tensor = (
            person_tensor * 2.0 - 1.0
        )

        garment_tensor = (
            garment_tensor * 2.0 - 1.0
        )

        # --------------------------------------------
        # Temporary masks
        # --------------------------------------------

        clothing_mask = torch.ones(
            1,
            768,
            512,
        )

        face_mask = torch.ones(
            1,
            768,
            512,
        )

        # --------------------------------------------
        # Batch + device
        # --------------------------------------------

        device = next(
            model.parameters()
        ).device

        person_tensor = (
            person_tensor
            .unsqueeze(0)
            .to(device)
        )

        garment_tensor = (
            garment_tensor
            .unsqueeze(0)
            .to(device)
        )

        clothing_mask = (
            clothing_mask
            .unsqueeze(0)
            .to(device)
        )

        face_mask = (
            face_mask
            .unsqueeze(0)
            .to(device)
        )

        # --------------------------------------------
        # Inference
        # --------------------------------------------

        with torch.no_grad():

            output = model(
                person_tensor,
                garment_tensor,
                clothing_mask,
                face_mask,
            )

        # --------------------------------------------
        # Convert output
        # --------------------------------------------

        output = output[0]

        output = (
            output
            .clamp(-1, 1)
            .add(1)
            .div(2)
            .mul(255)
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        result = Image.fromarray(
            output
        )

        return {
            "status": "success",
            "service": "VASUNDHARA VTON",
            "version": "V7",
            "image": encode_image(result),
        }

    except Exception as exc:

        print("========================================")
        print("VASUNDHARA ERROR")
        print("========================================")

        traceback.print_exc()

        return {
            "status": "error",
            "error": str(exc),
        }


print("========================================")
print("VASUNDHARA VTON V7 WORKER")
print("Starting...")
print("========================================")

runpod.serverless.start(
    {
        "handler": handler
    }
)
