import base64
import io
import traceback

import runpod
import torch
from PIL import Image

from vton.inference import generate_tryon


# =========================================================
# VASUNDHARA VTON V7
# =========================================================

print("=================================================")
print("Starting VASUNDHARA VTON V7")
print("=================================================")

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version    : {torch.version.cuda}")
    print(f"GPU             : {torch.cuda.get_device_name(0)}")


# =========================================================
# BASE64 -> PIL
# =========================================================

def decode_image(value):
    if not value:
        raise ValueError("Image data is missing")

    if isinstance(value, dict):
        value = value.get("data") or value.get("image")

    if not isinstance(value, str):
        raise ValueError("Image must be a base64 string")

    if value.startswith("data:image"):
        value = value.split(",", 1)[1]

    try:
        raw = base64.b64decode(value)
        image = Image.open(io.BytesIO(raw))
        return image.convert("RGB")

    except Exception as exc:
        raise ValueError(
            f"Could not decode image: {exc}"
        )


# =========================================================
# PIL -> BASE64
# =========================================================

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


# =========================================================
# HEALTH CHECK
# =========================================================

def health_check():

    return {
        "status": "ok",
        "service": "VASUNDHARA VTON",
        "version": "V7",
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "cuda_version": (
            torch.version.cuda
            if torch.cuda.is_available()
            else None
        ),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }


# =========================================================
# VTON
# =========================================================

def run_vton(
    person_image,
    garment_image,
    category="one-pieces",
):

    print("=================================================")
    print("Running VASUNDHARA VTON")
    print("=================================================")

    print(
        f"Person image size  : {person_image.size}"
    )

    print(
        f"Garment image size : {garment_image.size}"
    )

    print(
        f"Category           : {category}"
    )

    result = generate_tryon(
        person_image=person_image,
        garment_image=garment_image,
        category=category,
    )

    if not result:
        raise RuntimeError(
            "VTON returned no image"
        )

    print("VTON generation complete.")

    return result


# =========================================================
# RUNPOD HANDLER
# =========================================================

def handler(job):

    try:

        job_input = job.get("input", {})

        if not isinstance(job_input, dict):
            raise ValueError(
                "input must be an object"
            )

        # -------------------------------------------------
        # HEALTH TEST
        # -------------------------------------------------

        if job_input.get("test") is True:

            return health_check()

        # -------------------------------------------------
        # INPUT IMAGES
        # -------------------------------------------------

        person_data = (
            job_input.get("person_image")
            or job_input.get("person")
        )

        garment_data = (
            job_input.get("garment_image")
            or job_input.get("garment")
            or job_input.get("saree_image")
        )

        if not person_data:
            raise ValueError(
                "person_image is required"
            )

        if not garment_data:
            raise ValueError(
                "garment_image is required"
            )

        # -------------------------------------------------
        # CATEGORY
        # -------------------------------------------------

        category = job_input.get(
            "category",
            "one-pieces",
        )

        allowed_categories = {
            "tops",
            "bottoms",
            "one-pieces",
        }

        if category not in allowed_categories:
            raise ValueError(
                "category must be one of: "
                "tops, bottoms, one-pieces"
            )

        # -------------------------------------------------
        # GARMENT PHOTO TYPE
        # -------------------------------------------------

        garment_photo_type = job_input.get(
            "garment_photo_type",
            "flat-lay",
        )

        if garment_photo_type not in {
            "model",
            "flat-lay",
        }:
            raise ValueError(
                "garment_photo_type must be "
                "model or flat-lay"
            )

        # -------------------------------------------------
        # DECODE
        # -------------------------------------------------

        person_image = decode_image(
            person_data
        )

        garment_image = decode_image(
            garment_data
        )

        # -------------------------------------------------
        # GENERATE
        # -------------------------------------------------

        result = run_vton(
            person_image,
            garment_image,
            category=category,
        )

        # -------------------------------------------------
        # ENCODE
        # -------------------------------------------------

        output_image = encode_image(result)

        return {
            "status": "success",
            "service": "VASUNDHARA VTON",
            "version": "V7",

            "category": category,

            "image": output_image,

            "format": "jpeg",
        }

    except Exception as exc:

        print("=================================================")
        print("VASUNDHARA VTON ERROR")
        print("=================================================")

        traceback.print_exc()

        return {
            "status": "error",
            "error": str(exc),
        }


# =========================================================
# START WORKER
# =========================================================

print("=================================================")
print("VASUNDHARA VTON WORKER READY")
print("=================================================")

runpod.serverless.start(
    {
        "handler": handler
    }
)
