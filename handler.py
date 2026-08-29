import os
import base64
import io
import tempfile

import runpod
import torch

from PIL import Image
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download

from model.pipeline import CatVTONPipeline
from model.cloth_masker import AutoMasker
from utils import resize_and_crop, resize_and_padding, init_weight_dtype


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

DEVICE = "cuda"
WIDTH = 768
HEIGHT = 1024

MODEL_REPO = "zhengchong/CatVTON"

print("Loading CatVTON...")

# Download CatVTON checkpoint
REPO_PATH = snapshot_download(repo_id=MODEL_REPO)

# Load pipeline
pipeline = CatVTONPipeline(
    base_ckpt="booksforcharlie/stable-diffusion-inpainting",
    attn_ckpt=REPO_PATH,
    attn_ckpt_version="mix",
    weight_dtype=torch.bfloat16,
    use_tf32=True,
    device=DEVICE,
)

pipeline.to(DEVICE)

# Mask processor
mask_processor = VaeImageProcessor(
    vae_scale_factor=8,
    do_normalize=False,
    do_binarize=True,
    do_convert_grayscale=True,
)

# Automatic human/clothing mask
automasker = AutoMasker(
    densepose_ckpt=os.path.join(REPO_PATH, "DensePose"),
    schp_ckpt=os.path.join(REPO_PATH, "SCHP"),
    device=DEVICE,
)

print("CatVTON loaded successfully.")


# ---------------------------------------------------------
# IMAGE HELPERS
# ---------------------------------------------------------

def decode_image(value):
    """
    Accepts:
      - base64 string
      - data:image/...;base64,... string
    """

    if not value:
        raise ValueError("Image is missing")

    if value.startswith("data:"):
        value = value.split(",", 1)[1]

    image_bytes = base64.b64decode(value)

    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def encode_image(image):
    buffer = io.BytesIO()

    image.save(buffer, format="PNG")

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ---------------------------------------------------------
# TRY-ON
# ---------------------------------------------------------

def run_tryon(
    person_image,
    cloth_image,
    cloth_type="overall",
    steps=50,
    guidance_scale=2.5,
    seed=-1,
):

    person_image = resize_and_crop(
        person_image,
        (WIDTH, HEIGHT)
    )

    cloth_image = resize_and_padding(
        cloth_image,
        (WIDTH, HEIGHT)
    )

    # Generate automatic mask
    mask = automasker(
        person_image,
        cloth_type
    )["mask"]

    mask = mask_processor.blur(
        mask,
        blur_factor=9
    )

    # Random generator
    generator = None

    if seed is not None and int(seed) >= 0:
        generator = torch.Generator(
            device=DEVICE
        ).manual_seed(int(seed))

    # CatVTON inference
    result = pipeline(
        image=person_image,
        condition_image=cloth_image,
        mask=mask,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
    )[0]

    return result


# ---------------------------------------------------------
# RUNPOD HANDLER
# ---------------------------------------------------------

def handler(job):

    try:

        job_input = job.get("input", {})

        person_value = (
            job_input.get("model_image")
            or job_input.get("person_image")
        )

        cloth_value = (
            job_input.get("garment_image")
            or job_input.get("cloth_image")
        )

        if not person_value:
            raise ValueError(
                "model_image/person_image is required"
            )

        if not cloth_value:
            raise ValueError(
                "garment_image/cloth_image is required"
            )

        cloth_type = job_input.get(
            "cloth_type",
            "overall"
        )

        steps = job_input.get(
            "steps",
            50
        )

        guidance_scale = job_input.get(
            "guidance_scale",
            2.5
        )

        seed = job_input.get(
            "seed",
            -1
        )

        print("Decoding input images...")

        person_image = decode_image(
            person_value
        )

        cloth_image = decode_image(
            cloth_value
        )

        print("Running CatVTON...")

        result = run_tryon(
            person_image=person_image,
            cloth_image=cloth_image,
            cloth_type=cloth_type,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        result_base64 = encode_image(result)

        print("Try-on completed.")

        return {
            "success": True,
            "model": "CatVTON",
            "provider": "vasundhara",
            "image_base64": result_base64,
        }

    except Exception as e:

        print("ERROR:", repr(e))

        return {
            "success": False,
            "error": str(e),
        }


# ---------------------------------------------------------
# START SERVERLESS WORKER
# ---------------------------------------------------------

runpod.serverless.start({
    "handler": handler
})
