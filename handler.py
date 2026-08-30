import os
import sys
import io
import base64
import traceback

# ============================================================
# STARTUP DIAGNOSTICS
# ============================================================

print("=" * 70, flush=True)
print("VASUNDHARA VTON SERVERLESS WORKER STARTING", flush=True)
print("=" * 70, flush=True)

print("Python:", sys.version, flush=True)

# ============================================================
# CATVTON PATH
# ============================================================

CATVTON_ROOT = os.environ.get(
    "CATVTON_ROOT",
    "/workspace/CatVTON",
)

if CATVTON_ROOT not in sys.path:
    sys.path.insert(0, CATVTON_ROOT)

print("CATVTON_ROOT:", CATVTON_ROOT, flush=True)
print("Python path configured.", flush=True)

# ============================================================
# IMPORT TORCH
# ============================================================

try:
    import torch

    print("PyTorch:", torch.__version__, flush=True)
    print("Torch CUDA:", torch.version.cuda, flush=True)
    print(
        "CUDA available:",
        torch.cuda.is_available(),
        flush=True,
    )

except Exception:
    print("FAILED TO IMPORT PYTORCH", flush=True)
    traceback.print_exc()
    raise

# ============================================================
# CUDA CHECK
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required. "
        "RunPod worker does not have a usable NVIDIA GPU."
    )

print(
    "GPU:",
    torch.cuda.get_device_name(0),
    flush=True,
)

# ============================================================
# OTHER IMPORTS
# ============================================================

try:
    import runpod

    from PIL import Image

    from diffusers.image_processor import VaeImageProcessor
    from huggingface_hub import snapshot_download

    print("Basic imports: OK", flush=True)

except Exception:
    print("FAILED BASIC IMPORTS", flush=True)
    traceback.print_exc()
    raise

# ============================================================
# CATVTON IMPORTS
# ============================================================

try:
    print("Importing CatVTON...", flush=True)

    from model.pipeline import CatVTONPipeline
    from model.cloth_masker import AutoMasker

    from utils import (
        resize_and_crop,
        resize_and_padding,
        init_weight_dtype,
    )

    print("CatVTON imports: OK", flush=True)

except Exception:
    print("FAILED CATVTON IMPORTS", flush=True)
    traceback.print_exc()
    raise

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
# MODEL VARIABLES
# ============================================================

pipeline = None
automasker = None
mask_processor = None


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    global pipeline
    global automasker
    global mask_processor

    print("=" * 70, flush=True)
    print("LOADING CATVTON MODELS", flush=True)
    print("=" * 70, flush=True)

    # --------------------------------------------------------
    # Download CatVTON checkpoint
    # --------------------------------------------------------

    print(
        "Downloading/loading:",
        MODEL_REPO,
        flush=True,
    )

    repo_path = snapshot_download(
        repo_id=MODEL_REPO,
    )

    print(
        "CatVTON checkpoint:",
        repo_path,
        flush=True,
    )

    # --------------------------------------------------------
    # Verify required checkpoint directories
    # --------------------------------------------------------

    densepose_path = os.path.join(
        repo_path,
        "DensePose",
    )

    schp_path = os.path.join(
        repo_path,
        "SCHP",
    )

    print(
        "DensePose path:",
        densepose_path,
        flush=True,
    )

    print(
        "SCHP path:",
        schp_path,
        flush=True,
    )

    if not os.path.isdir(densepose_path):
        raise FileNotFoundError(
            f"DensePose checkpoint directory not found: "
            f"{densepose_path}"
        )

    if not os.path.isdir(schp_path):
        raise FileNotFoundError(
            f"SCHP checkpoint directory not found: "
            f"{schp_path}"
        )

    # --------------------------------------------------------
    # Weight type
    # --------------------------------------------------------

    print(
        "Initializing BF16 weight dtype...",
        flush=True,
    )

    weight_dtype = init_weight_dtype(
        "bf16"
    )

    print(
        "Weight dtype:",
        weight_dtype,
        flush=True,
    )

    # --------------------------------------------------------
    # CatVTON pipeline
    # --------------------------------------------------------

    print(
        "Loading CatVTON pipeline...",
        flush=True,
    )

    pipeline = CatVTONPipeline(
        base_ckpt=BASE_MODEL,
        attn_ckpt=repo_path,
        attn_ckpt_version="mix",
        weight_dtype=weight_dtype,
        use_tf32=True,
        device=DEVICE,
    )

    print(
        "CatVTON pipeline loaded successfully.",
        flush=True,
    )

    # --------------------------------------------------------
    # VAE mask processor
    # --------------------------------------------------------

    mask_processor = VaeImageProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    )

    print(
        "Mask processor loaded.",
        flush=True,
    )

    # --------------------------------------------------------
    # AutoMasker
    # --------------------------------------------------------

    print(
        "Loading AutoMasker...",
        flush=True,
    )

    automasker = AutoMasker(
        densepose_ckpt=densepose_path,
        schp_ckpt=schp_path,
        device=DEVICE,
    )

    print(
        "AutoMasker loaded successfully.",
        flush=True,
    )

    # --------------------------------------------------------
    # GPU memory
    # --------------------------------------------------------

    try:
        allocated = (
            torch.cuda.memory_allocated(0)
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved(0)
            / 1024**3
        )

        print(
            f"GPU memory allocated: {allocated:.2f} GB",
            flush=True,
        )

        print(
            f"GPU memory reserved: {reserved:.2f} GB",
            flush=True,
        )

    except Exception:
        traceback.print_exc()

    print("=" * 70, flush=True)
    print("VASUNDHARA CATVTON READY", flush=True)
    print("=" * 70, flush=True)


# ============================================================
# IMAGE DECODING
# ============================================================

def decode_image(value):

    if value is None:
        raise ValueError(
            "Image value is missing."
        )

    # --------------------------------------------------------
    # PIL
    # --------------------------------------------------------

    if isinstance(value, Image.Image):
        return value.convert("RGB")

    # --------------------------------------------------------
    # Base64 / Data URI
    # --------------------------------------------------------

    if isinstance(value, str):

        if value.startswith("data:"):

            try:
                value = value.split(
                    ",",
                    1,
                )[1]

            except Exception as exc:
                raise ValueError(
                    "Invalid data URI image."
                ) from exc

        try:

            raw = base64.b64decode(
                value,
                validate=True,
            )

            return Image.open(
                io.BytesIO(raw)
            ).convert("RGB")

        except Exception as exc:

            raise ValueError(
                f"Could not decode base64 image: {exc}"
            ) from exc

    # --------------------------------------------------------
    # Raw bytes
    # --------------------------------------------------------

    if isinstance(
        value,
        (bytes, bytearray),
    ):

        try:

            return Image.open(
                io.BytesIO(value)
            ).convert("RGB")

        except Exception as exc:

            raise ValueError(
                f"Could not decode image bytes: {exc}"
            ) from exc

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
        format="PNG",
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

    if pipeline is None:
        raise RuntimeError(
            "CatVTON pipeline is not loaded."
        )

    if automasker is None:
        raise RuntimeError(
            "AutoMasker is not loaded."
        )

    # --------------------------------------------------------
    # Person
    # --------------------------------------------------------

    print(
        "Preparing person image...",
        flush=True,
    )

    person_image = resize_and_crop(
        person_image,
        (WIDTH, HEIGHT),
    )

    # --------------------------------------------------------
    # Garment
    # --------------------------------------------------------

    print(
        "Preparing garment image...",
        flush=True,
    )

    garment_image = resize_and_padding(
        garment_image,
        (WIDTH, HEIGHT),
    )

    # --------------------------------------------------------
    # Clothing mask
    # --------------------------------------------------------

    print(
        "Generating clothing mask...",
        flush=True,
    )

    print(
        "Cloth type:",
        cloth_type,
        flush=True,
    )

    mask_result = automasker(
        person_image,
        cloth_type,
    )

    mask = mask_result["mask"]

    mask = mask_processor.blur(
        mask,
        blur_factor=9,
    )

    print(
        "Clothing mask generated.",
        flush=True,
    )

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    generator = None

    if seed is not None:

        seed = int(seed)

        if seed >= 0:

            print(
                f"Using seed: {seed}",
                flush=True,
            )

            generator = torch.Generator(
                device=DEVICE
            ).manual_seed(seed)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print("=" * 70, flush=True)
    print("RUNNING CATVTON", flush=True)
    print(
        f"Resolution: {WIDTH}x{HEIGHT}",
        flush=True,
    )
    print(
        f"Steps: {steps}",
        flush=True,
    )
    print(
        f"Guidance: {guidance_scale}",
        flush=True,
    )
    print("=" * 70, flush=True)

    result = pipeline(
        image=person_image,
        condition_image=garment_image,
        mask=mask,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        height=HEIGHT,
        width=WIDTH,
        generator=generator,
    )[0]

    print(
        "Generation complete.",
        flush=True,
    )

    return result


# ============================================================
# RUNPOD HANDLER
# ============================================================

def handler(job):

    try:

        job_input = job.get(
            "input",
            {},
        )

        print("=" * 70, flush=True)
        print(
            "NEW VASUNDHARA VTON REQUEST",
            flush=True,
        )
        print("=" * 70, flush=True)

        # ----------------------------------------------------
        # Person image
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
        # Garment image
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
        # Options
        # ----------------------------------------------------

        cloth_type = job_input.get(
            "cloth_type",
            "overall",
        )

        steps = job_input.get(
            "steps",
            DEFAULT_STEPS,
        )

        guidance_scale = job_input.get(
            "guidance_scale",
            DEFAULT_GUIDANCE,
        )

        seed = job_input.get(
            "seed",
            -1,
        )

        # ----------------------------------------------------
        # Decode person
        # ----------------------------------------------------

        print(
            "Decoding person image...",
            flush=True,
        )

        person_image = decode_image(
            person_value
        )

        # ----------------------------------------------------
        # Decode garment
        # ----------------------------------------------------

        print(
            "Decoding garment image...",
            flush=True,
        )

        garment_image = decode_image(
            garment_value
        )

        # ----------------------------------------------------
        # Generate
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
        # Encode
        # ----------------------------------------------------

        print(
            "Encoding result...",
            flush=True,
        )

        result_base64 = encode_image(
            result
        )

        print("=" * 70, flush=True)
        print(
            "VTON SUCCESS",
            flush=True,
        )
        print("=" * 70, flush=True)

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
            flush=True,
        )

        traceback.print_exc()

        return {
            "success": False,
            "error": (
                "GPU out of memory. "
                "Reduce resolution or use a larger GPU."
            ),
        }

    except Exception as exc:

        print("=" * 70, flush=True)
        print(
            "VTON ERROR:",
            repr(exc),
            flush=True,
        )

        traceback.print_exc()

        print("=" * 70, flush=True)

        return {
            "success": False,
            "error": str(exc),
        }


# ============================================================
# LOAD MODELS BEFORE SERVERLESS START
# ============================================================

try:

    load_models()

except Exception as exc:

    print("=" * 70, flush=True)
    print(
        "FATAL STARTUP ERROR",
        flush=True,
    )
    print(
        repr(exc),
        flush=True,
    )
    traceback.print_exc()
    print("=" * 70, flush=True)

    # Make the container exit with a clear error.
    raise


# ============================================================
# START RUNPOD SERVERLESS
# ============================================================

print("=" * 70, flush=True)
print(
    "Starting RunPod Serverless worker...",
    flush=True,
)
print("=" * 70, flush=True)

runpod.serverless.start(
    {
        "handler": handler,
    }
)
