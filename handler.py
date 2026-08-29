import os
import sys
import io
import base64
import traceback

import runpod
import torch

from PIL import Image

# ============================================================
# CATVTON PATH
# ============================================================

CATVTON_ROOT = os.environ.get(
    "CATVTON_ROOT",
    "/workspace/CatVTON"
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
        "This worker must run on a RunPod NVIDIA GPU."
    )

print("=" * 60, flush=True)
print("VASUNDHARA FREE VTON")
print("CATVTON SERVERLESS WORKER")
print("=" * 60, flush=True)

print(
    "GPU:",
    torch.cuda.get_device_name(0),
    flush=True
)

print(
    "CUDA:",
    torch.version.cuda,
    flush=True
)

print(
    "PyTorch:",
    torch.__version__,
    flush=True
)

# ============================================================
# DOWNLOAD CATVTON MODEL
# ============================================================

print("=" * 60, flush=True)
print("Downloading/loading CatVTON model...", flush=True)
print("=" * 60, flush=True)

REPO_PATH = snapshot_download(
    repo_id=MODEL_REPO
)

print(
    "CatVTON model path:",
    REPO_PATH,
    flush=True
)

# ============================================================
# LOAD CATVTON PIPELINE
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
        "DensePose",
    ),
    schp_ckpt=os.path.join(
        REPO_PATH,
        "SCHP",
    ),
    device=DEVICE,
)

print("=" * 60, flush=True)
print("VASUNDHARA CATVTON READY")
print("=" * 60, flush=True)

# ============================================================
# IMAGE DECODING
# ============================================================

def decode_image(value):

    if value is None:
        raise ValueError("Image value is missing.")

    # PIL image
    if isinstance(value, Image.Image):
        return value.convert("RGB")

    # Base64 / data URI
    if isinstance(value, str):

        if value.startswith("data:"):
            try:
                value = value.split(",", 1)[1]
            except Exception:
                raise ValueError("Invalid data URI image.")

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

    # Raw bytes
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
    # CLOTHING MASK
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
    # RANDOM SEED
    # --------------------------------------------------------

    generator = None

    if seed is not None:

        seed = int(seed)

        if seed >= 0:

            print(
                f"Using seed: {seed}",
                flush=True
            )

            generator = torch.Generator(
                device=DEVICE
            ).manual_seed(seed)

    # --------------------------------------------------------
    # CATVTON INFERENCE
    # --------------------------------------------------------

    print("=" * 60, flush=True)
    print("RUNNING CATVTON...", flush=True)
    print("=" * 60, flush=True)

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

        print("=" * 60, flush=True)
        print("NEW VASUNDHARA VTON REQUEST", flush=True)
        print("=" * 60, flush=True)

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
            or job_input.get("saree")
            or job_input.get("product_image")
        )

        if not garment_value:

            raise ValueError(
                "Garment image is required. "
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
        # DECODE PERSON
        # ----------------------------------------------------

        print(
            "Decoding person image...",
            flush=True
        )

        person_image = decode_image(
            person_value
        )

        # ----------------------------------------------------
        # DECODE GARMENT
        # ----------------------------------------------------

        print(
            "Decoding garment image...",
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
        # ENCODE RESULT
        # ----------------------------------------------------

        print(
            "Encoding result...",
            flush=True
        )

        result_base64 = encode_image(
            result
        )

        print("=" * 60, flush=True)
        print("VTON SUCCESS", flush=True)
        print("=" * 60, flush=True)

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

        print(
            "GPU OUT OF MEMORY",
            flush=True
        )

        return {
            "success": False,
            "error": (
                "GPU out of memory. "
                "Reduce resolution or use a larger GPU."
            ),
        }

    except Exception as exc:

        print(
            "=" * 60,
            flush=True
        )

        print(
            "VTON ERROR:",
            repr(exc),
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
# START RUNPOD SERVERLESS
# ============================================================

print(
    "Starting RunPod Serverless worker...",
    flush=True
)

runpod.serverless.start({
    "handler": handler
})
