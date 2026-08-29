import os
import sys
import io
import base64

import runpod
import torch
from PIL import Image

# ============================================================
# CATVTON PATH
# ============================================================

CATVTON_ROOT = os.environ.get(
    "CATVTON_ROOT",
    "/opt/CatVTON"
)

if CATVTON_ROOT not in sys.path:
    sys.path.insert(0, CATVTON_ROOT)


# ============================================================
# CATVTON IMPORTS
# ============================================================

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
# CONFIGURATION
# ============================================================

DEVICE = "cuda"

WIDTH = 768
HEIGHT = 1024

MODEL_REPO = "zhengchong/CatVTON"

BASE_MODEL = "booksforcharlie/stable-diffusion-inpainting"

DEFAULT_STEPS = 40
DEFAULT_GUIDANCE = 2.5


# ============================================================
# GPU CHECK
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required. "
        "This handler must run on a RunPod GPU."
    )

print("========================================")
print("VASUNDHARA CATVTON")
print("========================================")

print(
    "GPU:",
    torch.cuda.get_device_name(0),
    flush=True
)


# ============================================================
# DOWNLOAD MODEL
# ============================================================

print("Downloading CatVTON model...", flush=True)

REPO_PATH = snapshot_download(
    repo_id=MODEL_REPO
)

print(
    "CatVTON model:",
    REPO_PATH,
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

print("Pipeline loaded.", flush=True)


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
        "DensePose",
    ),
    schp_ckpt=os.path.join(
        REPO_PATH,
        "SCHP",
    ),
    device=DEVICE,
)

print("========================================")
print("CATVTON READY")
print("========================================", flush=True)


# ============================================================
# IMAGE DECODING
# ============================================================

def decode_image(value):

    if value is None:
        raise ValueError(
            "Image value is missing."
        )

    # Already PIL image
    if isinstance(value, Image.Image):
        return value.convert("RGB")

    # Data URI / base64
    if isinstance(value, str):

        if value.startswith("data:"):
            value = value.split(
                ",",
                1
            )[1]

        try:

            raw = base64.b64decode(
                value
            )

            return Image.open(
                io.BytesIO(raw)
            ).convert("RGB")

        except Exception as exc:

            raise ValueError(
                f"Could not decode base64 image: {exc}"
            )

    raise ValueError(
        "Unsupported image format."
    )


# ============================================================
# IMAGE ENCODING
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
# CATVTON TRY-ON
# ============================================================

def run_tryon(
    person_image,
    garment_image,
    cloth_type="overall",
    steps=40,
    guidance_scale=2.5,
    seed=-1,
):

    print("Preparing person image...", flush=True)

    person_image = resize_and_crop(
        person_image,
        (WIDTH, HEIGHT)
    )

    print("Preparing garment image...", flush=True)

    garment_image = resize_and_padding(
        garment_image,
        (WIDTH, HEIGHT)
    )

    # --------------------------------------------------------
    # MASK
    # --------------------------------------------------------

    print(
        "Generating clothing mask...",
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
        "Running CatVTON...",
        flush=True
    )

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

        job_input = job.get(
            "input",
            {}
        )

        print(
            "========================================",
            flush=True
        )

        print(
            "NEW VTON REQUEST",
            flush=True
        )

        # ----------------------------------------------------
        # PERSON
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
        # GARMENT / SAREE
        # ----------------------------------------------------

        garment_value = (
            job_input.get("garment_image")
            or job_input.get("cloth_image")
            or job_input.get("saree")
            or job_input.get("product_image")
        )

        if not garment_value:

            raise ValueError(
                "Saree/garment image is required. "
                "Use garment_image or cloth_image."
            )

        # ----------------------------------------------------
        # OPTIONS
        # ----------------------------------------------------

        cloth_type = job_input.get(
            "cloth_type",
            "overall"
        )

        steps = job_input.get(
            "steps",
            DEFAULT_STEPS
        )

        guidance_scale = job_input.get(
            "guidance_scale",
            DEFAULT_GUIDANCE
        )

        seed = job_input.get(
            "seed",
            -1
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
            "Decoding saree image...",
            flush=True
        )

        garment_image = decode_image(
            garment_value
        )

        # ----------------------------------------------------
        # GENERATE
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
        # ENCODE
        # ----------------------------------------------------

        result_base64 = encode_image(
            result
        )

        print(
            "VTON SUCCESS",
            flush=True
        )

        print(
            "========================================",
            flush=True
        )

        return {
            "success": True,
            "provider": "runpod",
            "model": "CatVTON",
            "image_base64": result_base64,
            "width": WIDTH,
            "height": HEIGHT,
            "steps": int(steps),
            "guidance_scale": float(
                guidance_scale
            ),
            "seed": int(seed),
        }

    except torch.cuda.OutOfMemoryError:

        torch.cuda.empty_cache()

        return {
            "success": False,
            "error": (
                "GPU out of memory. "
                "Use a larger RunPod GPU or reduce "
                "the image resolution."
            ),
        }

    except Exception as exc:

        print(
            "VTON ERROR:",
            repr(exc),
            flush=True
        )

        return {
            "success": False,
            "error": str(exc),
        }


# ============================================================
# START RUNPOD SERVERLESS
# ============================================================

print(
    "Starting RunPod Serverless worker...",
    flush=True
)

runpod.serverless.start({
    "handler": handler
})
