import base64
import io
import os
import traceback
from pathlib import Path

import numpy as np
import runpod
import torch
from PIL import Image

from vton.model import create_model, load_checkpoint


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/workspace/checkpoints/vasundhara-vton/best.pt",
)
DEFAULT_WIDTH = int(os.environ.get("VTON_WIDTH", "384"))
DEFAULT_HEIGHT = int(os.environ.get("VTON_HEIGHT", "512"))

model = None
MODEL_READY = False


def decode_image(value):
    if value is None:
        raise ValueError("Image value is missing")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, str):
        if value.startswith("data:"):
            value = value.split(",", 1)[1]
        try:
            raw = base64.b64decode(value)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Could not decode image: {exc}") from exc
    raise ValueError("Unsupported image format")


def decode_mask(value, size):
    if value is None:
        return None
    if isinstance(value, Image.Image):
        image = value.convert("L")
    elif isinstance(value, (bytes, bytearray)):
        image = Image.open(io.BytesIO(value)).convert("L")
    elif isinstance(value, str):
        if value.startswith("data:"):
            value = value.split(",", 1)[1]
        try:
            image = Image.open(io.BytesIO(base64.b64decode(value))).convert("L")
        except Exception as exc:
            raise ValueError(f"Could not decode mask: {exc}") from exc
    else:
        raise ValueError("Unsupported mask format")
    return image.resize(size, Image.Resampling.NEAREST)


def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def image_to_tensor(image, size):
    image = image.resize(size, Image.Resampling.BICUBIC)
    values = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = values.reshape(image.height, image.width, 3).permute(2, 0, 1)
    tensor = tensor.float().div(255.0).mul(2.0).sub(1.0)
    return tensor.unsqueeze(0)


def mask_to_tensor(mask, size):
    mask = mask.resize(size, Image.Resampling.NEAREST)
    values = torch.frombuffer(bytearray(mask.tobytes()), dtype=torch.uint8)
    tensor = values.reshape(mask.height, mask.width).float().div(255.0)
    return tensor.unsqueeze(0).unsqueeze(0)


def fallback_masks(size):
    """Simple fallback only for endpoint smoke tests.

    Production-quality results should send trained segmentation masks from
    the saree/person segmentation pipeline. These masks are intentionally
    conservative and are not a substitute for trained segmentation.
    """
    width, height = size
    clothing = Image.new("L", size, 0)
    face = Image.new("L", size, 0)

    clothing_pixels = clothing.load()
    for y in range(int(height * 0.25), int(height * 0.92)):
        for x in range(int(width * 0.12), int(width * 0.88)):
            clothing_pixels[x, y] = 255

    face_pixels = face.load()
    for y in range(0, int(height * 0.24)):
        for x in range(int(width * 0.25), int(width * 0.75)):
            face_pixels[x, y] = 255

    return clothing, face


def load_model():
    global model, MODEL_READY

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for VASUNDHARA inference")

    print("=" * 60, flush=True)
    print("VASUNDHARA VTON WORKER", flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("PyTorch:", torch.__version__, flush=True)
    print("CUDA:", torch.version.cuda, flush=True)
    print("Model path:", MODEL_PATH, flush=True)
    print("=" * 60, flush=True)

    model = create_model(device=DEVICE)

    if Path(MODEL_PATH).is_file():
        print("Loading trained VASUNDHARA checkpoint...", flush=True)
        model = load_checkpoint(model, MODEL_PATH, device=DEVICE)
        MODEL_READY = True
        print("VASUNDHARA checkpoint loaded.", flush=True)
    else:
        MODEL_READY = False
        model.eval()
        print(
            "WARNING: trained checkpoint not found. "
            "Worker will start, but inference requests will return a clear error.",
            flush=True,
        )


load_model()


def handler(job):
    try:
        if not MODEL_READY:
            raise RuntimeError(
                f"VASUNDHARA trained checkpoint not found at {MODEL_PATH}. "
                "Train the model first and place best.pt at that path."
            )

        job_input = job.get("input", {})

        # The public VASUNDHARA API uses person_image + saree_image.
        # garment_image/cloth_image remain accepted for backwards compatibility.
        person_value = (
            job_input.get("person_image")
            or job_input.get("model_image")
            or job_input.get("person")
        )
        saree_value = (
            job_input.get("saree_image")
            or job_input.get("product_image")
            or job_input.get("garment_image")
            or job_input.get("cloth_image")
            or job_input.get("saree")
            or job_input.get("garment")
            or job_input.get("cloth")
        )

        if not person_value:
            raise ValueError("Person image is required")
        if not saree_value:
            raise ValueError("Saree image is required")

        width = max(128, min(int(job_input.get("width", DEFAULT_WIDTH)), 768))
        height = max(128, min(int(job_input.get("height", DEFAULT_HEIGHT)), 1024))
        size = (width, height)

        person = decode_image(person_value)
        saree = decode_image(saree_value)

        clothing_mask_value = (
            job_input.get("clothing_mask")
            or job_input.get("cloth_mask")
            or job_input.get("mask")
        )
        face_mask_value = job_input.get("face_mask")

        clothing_mask = decode_mask(clothing_mask_value, size) if clothing_mask_value else None
        face_mask = decode_mask(face_mask_value, size) if face_mask_value else None

        if clothing_mask is None or face_mask is None:
            fallback_clothing, fallback_face = fallback_masks(size)
            if clothing_mask is None:
                clothing_mask = fallback_clothing
            if face_mask is None:
                face_mask = fallback_face
            print(
                "Using fallback person masks; trained segmentation masks are recommended for production quality.",
                flush=True,
            )

        person_tensor = image_to_tensor(person, size).to(DEVICE)
        # A flat-lay saree product photo is intentionally passed as the garment
        # tensor without treating it as a shirt/dress crop. Saree-specific
        # draping/segmentation is part of the training pipeline.
        saree_tensor = image_to_tensor(saree, size).to(DEVICE)
        clothing_tensor = mask_to_tensor(clothing_mask, size).to(DEVICE)
        face_tensor = mask_to_tensor(face_mask, size).to(DEVICE)

        seed = int(job_input.get("seed", -1))
        if seed >= 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        print(f"Running VASUNDHARA saree inference at {width}x{height}...", flush=True)
        with torch.inference_mode():
            output = model(
                person_tensor,
                saree_tensor,
                clothing_tensor,
                face_tensor,
            )

        output = ((output.clamp(-1, 1) + 1.0) * 127.5).byte()
        output = output[0].permute(1, 2, 0).cpu().numpy()
        result = Image.fromarray(np.asarray(output), mode="RGB")

        return {
            "success": True,
            "image": encode_image(result),
            "format": "png",
            "width": result.width,
            "height": result.height,
            "model": "VASUNDHARA-VTON",
            "input_type": "flat_lay_saree",
            "checkpoint": MODEL_PATH,
            "seed": seed,
        }

    except Exception as exc:
        traceback.print_exc()
        return {
            "success": False,
            "error": str(exc),
        }


print("Starting RunPod Serverless worker...", flush=True)
runpod.serverless.start({"handler": handler})
