import os
import sys
import io
import base64
import traceback

import runpod
import torch

from PIL import Image


# ============================================================
# CONFIG
# ============================================================

CATVTON_ROOT = os.environ.get(
    "CATVTON_ROOT",
    "/workspace/CatVTON"
)

if CATVTON_ROOT not in sys.path:
    sys.path.insert(0, CATVTON_ROOT)


DEVICE = "cuda"

WIDTH = 768
HEIGHT = 1024

MODEL_REPO = "zhengchong/CatVTON"
BASE_MODEL = "booksforcharlie/stable-diffusion-inpainting"

DEFAULT_STEPS = 40
DEFAULT_GUIDANCE = 2.5


# ============================================================
# CATVTON IMPORTS
# ============================================================

print("Loading CatVTON modules...", flush=True)

from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download

from model.pipeline import CatVTONPipeline
from model.cloth_masker import AutoMasker

from utils import (
    resize_and_crop,
    resize_and_padding,
    init_weight_dtype,
)


# ============================================================
# GPU CHECK
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required. "
        "This worker must run on an NVIDIA GPU."
    )

print("=" * 60, flush=True)
print("VASUNDHARA VTON WORKER", flush=True)
print("=" * 60, flush=True)

print(
    f"GPU: {torch.cuda.get_device_name(0)}",
    flush=True
)

print(
    f"CUDA: {torch.version.cuda}",
    flush=True
)

print(
    f"PyTorch: {torch.__version__}",
    flush=True
)


# ============================================================
# DOWNLOAD CATVTON WEIGHTS
# ============================================================

print("=" * 60, flush=True)
print("Downloading/loading CatVTON weights...", flush=True)
print("=" * 60, flush=True)

REPO_PATH = snapshot_download(
    repo_id=MODEL_REPO
)

print(
    f"CatVTON weights: {REPO_PATH}",
    flush=True
)


# ============================================================
# LOAD PIPELINE
# ============================================================

print("Loading CatVTON pipeline...", flush=True)

weight_dtype = init_weight_dtype("bf16")

pipeline = CatVTONPipeline(
    base_ckpt=BASE_MODEL,
    attn_ckpt=REPO_PATH,
    attn_ckpt_version="mix",
    weight_dtype=weight_dtype,
    use_tf32=True,
    device=DEVICE,
)

print("CatVTON pipeline loaded.", flush=True)


# ============================================================
# MASK PROCESSOR
# ============================================================

mask_processor = VaeImageProcessor(
    vae_scale_factor=8,
    do_normalize=False,
    do_binarize=True,
    do_convert_grayscale=True,
)


# ============================================================
# AUTOMASKER
# ============================================================

print("Loading AutoMasker...", flush=True)

automasker = AutoMasker(
    densepose_ckpt=os.path.join(
        REPO_PATH,
        "DensePose"
    ),
    schp_ckpt=os.path.join(
        REPO_PATH,
        "SCHP"
    ),
    device=DEVICE,
)

print("=" * 60, flush=True)
print("VASUNDHARA VTON READY", flush=True)
print("=" * 60, flush=True)


# ============================================================
# IMAGE DECODER
# ============================================================

def decode_image(value):

    if value is None:
        raise ValueError(
            "Image value is missing."
        )

    # PIL
    if isinstance(value, Image.Image):
        return value.convert("RGB")

    # Base64 / Data URI
    if isinstance(value, str):

        if value.startswith("data:"):
            try:
                value = value.split(",", 1)[1]
            except Exception:
                raise ValueError(
                    "Invalid data URI image."
                )

        try:
            raw = base64.b64decode(value)

            image = Image.open(
                io.BytesIO(raw)
            ).convert("RGB")

            return image

        except Exception as exc:
            raise ValueError(
                f"Could not decode base64 image: {exc}"
            )

    # Bytes
    if isinstance(value, (bytes, bytearray)):

        try:
            return Image.open(
                io.BytesIO(value)
            ).convert("RGB")

        except Exception as exc:
            raise ValueError(
                f"Could not decode image bytes: {exc}"
            )

    raise ValueError(
        "Unsupported image format."
    )


# ============================================================
# IMAGE ENCODER
# ============================================================

def encode_image(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# TRY-ON
# ============================================================

def run_tryon(
    person_image,
    garment_image,
    cloth_type="overall",
    steps=DEFAULT_STEPS,
    guidance_scale=DEFAULT_GUIDANCE,
    seed=-1,
):

    print("=" * 60, flush=True)
    print("Preparing person image...", flush=True)

    person_image = resize_and_crop(
        person_image,
        (WIDTH, HEIGHT)
    )

    print(
        "Preparing garment image...",
        flush=True
    )

    garment_image = resize_and_padding(
        garment_image,
        (WIDTH, HEIGHT)
    )


    # --------------------------------------------------------
    # MASK
    # --------------------------------------------------------

    print(
        f"Generating mask: {cloth_type}",
        flush=True
    )

    mask = automasker(
        person_image,
        cloth_type
    )["mask"]

    mask = mask_processor.blur(
        mask,
        blur_factor=9
    )


    # --------------------------------------------------------
    # SEED
    # --------------------------------------------------------

    generator = None

    if seed is not None:

        seed = int(seed)

        if seed >= 0:

            generator = torch.Generator(
                device=DEVICE
            ).manual_seed(seed)


    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    print(
        "Running CatVTON inference...",
        flush=True
    )

    with torch.inference_mode():

        result = pipeline(
            image=person_image,
            condition_image=garment_image,
            mask=mask,
            num_inference_steps=int(steps),
            guidance_scale=float(
                guidance_scale
            ),
            height=HEIGHT,
            width=WIDTH,
            generator=generator,
        )[0]


    print(
        "Generation complete.",
        flush=True
    )

    return result


# ============================================================
# RUNPOD HANDLER
# ============================================================

def handler(job):

    try:

        print("=" * 60, flush=True)
        print("NEW VTON REQUEST", flush=True)
        print("=" * 60, flush=True)

        job_input = job.get(
            "input",
            {}
        )


        # ----------------------------------------------------
        # PERSON IMAGE
        # ----------------------------------------------------

        person_value = (
            job_input.get("model_image")
            or job_input.get("person_image")
            or job_input.get("person")
        )

        if not person_value:
            raise ValueError(
                "Person image is required. "
                "Use model_image or person_image."
            )


        # ----------------------------------------------------
        # GARMENT IMAGE
        # ----------------------------------------------------

        garment_value = (
            job_input.get("garment_image")
            or job_input.get("cloth_image")
            or job_input.get("garment")
            or job_input.get("cloth")
        )

        if not garment_value:
            raise ValueError(
                "Garment image is required. "
                "Use garment_image or cloth_image."
            )


        # ----------------------------------------------------
        # PARAMETERS
        # ----------------------------------------------------

        cloth_type = job_input.get(
            "cloth_type",
            "overall"
        )

        steps = int(
            job_input.get(
                "steps",
                DEFAULT_STEPS
            )
        )

        guidance_scale = float(
            job_input.get(
                "guidance_scale",
                DEFAULT_GUIDANCE
            )
        )

        seed = int(
            job_input.get(
                "seed",
                -1
            )
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        steps = max(
            1,
            min(steps, 80)
        )

        guidance_scale = max(
            0.0,
            min(guidance_scale, 20.0)
        )


        # ----------------------------------------------------
        # DECODE
        # ----------------------------------------------------

        print(
            "Decoding person image...",
            flush=True
        )

        person_image = decode_image(
            person_value
        )

        print(
            f"Person size: {person_image.size}",
            flush=True
        )


        print(
            "Decoding garment image...",
            flush=True
        )

        garment_image = decode_image(
            garment_value
        )

        print(
            f"Garment size: {garment_image.size}",
            flush=True
        )


        # ----------------------------------------------------
        # RUN VTON
        # ----------------------------------------------------

        result = run_tryon(
            person_image=person_image,
            garment_image=garment_image,
            cloth_type=cloth_type,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )


        # ----------------------------------------------------
        # ENCODE RESULT
        # ----------------------------------------------------

        print(
            "Encoding output PNG...",
            flush=True
        )

        output_base64 = encode_image(
            result
        )


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        response = {
            "success": True,
            "image": output_base64,
            "format": "png",
            "width": result.width,
            "height": result.height,
            "cloth_type": cloth_type,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }

        print(
            "VTON request completed successfully.",
            flush=True
        )

        return response


    except Exception as exc:

        print(
            "=" * 60,
            flush=True
        )

        print(
            "VTON ERROR",
            flush=True
        )

        print(
            str(exc),
            flush=True
        )

        traceback.print_exc()

        print(
            "=" * 60,
            flush=True
        )

        return {
            "success": False,
            "error": str(exc),
        }


# ============================================================
# START SERVERLESS WORKER
# ============================================================

print(
    "Starting RunPod Serverless worker...",
    flush=True
)

runpod.serverless.start({
    "handler": handler
})
