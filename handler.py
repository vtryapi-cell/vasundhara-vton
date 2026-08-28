import os
import io
import base64
import tempfile
import traceback

import runpod
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from diffusers.image_processor import VaeImageProcessor

from model.pipeline import CatVTONPipeline
from model.cloth_masker import AutoMasker
from utils import resize_and_crop, resize_and_padding


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEVICE = "cuda"
WIDTH = 768
HEIGHT = 1024

MODEL_REPO = os.getenv("CATVTON_MODEL", "zhengchong/CatVTON")
BASE_MODEL = os.getenv(
    "BASE_MODEL",
    "booksforcharlie/stable-diffusion-inpainting",
)

STEPS = int(os.getenv("DEFAULT_STEPS", "30"))
GUIDANCE = float(os.getenv("DEFAULT_GUIDANCE", "2.5"))


print("========================================")
print("VASUNDHARA CATVTON SERVERLESS WORKER")
print("========================================")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required.")

print("GPU:", torch.cuda.get_device_name(0))


# ---------------------------------------------------------
# Download / locate CatVTON checkpoint
# ---------------------------------------------------------

print("Loading CatVTON checkpoint...")

repo_path = snapshot_download(
    repo_id=MODEL_REPO
)

print("CatVTON checkpoint:", repo_path)


# ---------------------------------------------------------
# Load CatVTON pipeline
# ---------------------------------------------------------

pipeline = CatVTONPipeline(
    base_ckpt=BASE_MODEL,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=torch.bfloat16,
    use_tf32=True,
    device=DEVICE,
)

print("CatVTON pipeline loaded.")


# ---------------------------------------------------------
# Automatic mask generator
# ---------------------------------------------------------

mask_processor = VaeImageProcessor(
    vae_scale_factor=8,
    do_normalize=False,
    do_binarize=True,
    do_convert_grayscale=True,
)

automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device=DEVICE,
)

print("AutoMasker loaded.")
print("Worker ready.")


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def decode_image(value):
    """
    Accepts:
      - base64 string
      - data:image/...;base64,... string
    """

    if not value:
        raise ValueError("Image data is missing.")

    if value.startswith("data:"):
        value = value.split(",", 1)[1]

    raw = base64.b64decode(value)

    return Image.open(io.BytesIO(raw)).convert("RGB")


def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_seed(value):
    if value is None:
        return None

    try:
        value = int(value)
    except Exception:
        return None

    if value < 0:
        return None

    return value


# ---------------------------------------------------------
# Main inference
# ---------------------------------------------------------

def handler(job):

    job_input = job.get("input", {})

    try:

        person_data = job_input.get("person_image")
        cloth_data = job_input.get("cloth_image")

        if not person_data:
            raise ValueError("person_image is required.")

        if not cloth_data:
            raise ValueError("cloth_image is required.")

        cloth_type = job_input.get(
            "cloth_type",
            "upper"
        )

        if cloth_type not in [
            "upper",
            "lower",
            "overall",
        ]:
            cloth_type = "upper"

        steps = int(
            job_input.get(
                "steps",
                STEPS
            )
        )

        guidance = float(
            job_input.get(
                "guidance_scale",
                GUIDANCE
            )
        )

        seed = get_seed(
            job_input.get("seed", -1)
        )

        print("Starting VTON job...")
        print("Cloth type:", cloth_type)
        print("Steps:", steps)
        print("Guidance:", guidance)
        print("Seed:", seed)

        # -------------------------------------------------
        # Decode images
        # -------------------------------------------------

        person_image = decode_image(person_data)
        cloth_image = decode_image(cloth_data)

        # -------------------------------------------------
        # Resize
        # -------------------------------------------------

        person_image = resize_and_crop(
            person_image,
            (WIDTH, HEIGHT)
        )

        cloth_image = resize_and_padding(
            cloth_image,
            (WIDTH, HEIGHT)
        )

        # -------------------------------------------------
        # Automatic clothing mask
        # -------------------------------------------------

        print("Generating clothing mask...")

        mask = automasker(
            person_image,
            cloth_type
        )["mask"]

        mask = mask_processor.blur(
            mask,
            blur_factor=9
        )

        # -------------------------------------------------
        # Random generator
        # -------------------------------------------------

        generator = None

        if seed is not None:
            generator = torch.Generator(
                device=DEVICE
            ).manual_seed(seed)

        # -------------------------------------------------
        # CatVTON inference
        # -------------------------------------------------

        print("Running CatVTON...")

        with torch.inference_mode():

            result = pipeline(
                image=person_image,
                condition_image=cloth_image,
                mask=mask,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )[0]

        # -------------------------------------------------
        # Encode result
        # -------------------------------------------------

        result_b64 = encode_image(result)

        print("VTON completed.")

        return {
            "status": "success",
            "image": result_b64,
            "format": "jpeg",
            "width": result.width,
            "height": result.height,
        }

    except Exception as exc:

        print("VTON ERROR:")
        traceback.print_exc()

        return {
            "status": "error",
            "message": str(exc),
        }

    finally:

        # Release temporary CUDA memory between jobs.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------
# RunPod Serverless
# ---------------------------------------------------------

runpod.serverless.start({
    "handler": handler
})
