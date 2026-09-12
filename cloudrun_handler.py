import base64
import io
import os
import traceback
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

from vton.model import create_model, load_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/checkpoints/vasundhara-vton/best.pt")
DEFAULT_WIDTH = int(os.environ.get("VTON_WIDTH", "384"))
DEFAULT_HEIGHT = int(os.environ.get("VTON_HEIGHT", "512"))

app = FastAPI(title="VASUNDHARA VTON")
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
            return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")
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
        image = Image.open(io.BytesIO(base64.b64decode(value))).convert("L")
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
    return tensor.float().div(255.0).mul(2.0).sub(1.0).unsqueeze(0)


def mask_to_tensor(mask, size):
    mask = mask.resize(size, Image.Resampling.NEAREST)
    values = torch.frombuffer(bytearray(mask.tobytes()), dtype=torch.uint8)
    tensor = values.reshape(mask.height, mask.width).float().div(255.0)
    return tensor.unsqueeze(0).unsqueeze(0)


def fallback_masks(size):
    width, height = size
    clothing = Image.new("L", size, 0)
    face = Image.new("L", size, 0)
    cp = clothing.load()
    fp = face.load()
    for y in range(int(height * 0.25), int(height * 0.92)):
        for x in range(int(width * 0.12), int(width * 0.88)):
            cp[x, y] = 255
    for y in range(0, int(height * 0.24)):
        for x in range(int(width * 0.25), int(width * 0.75)):
            fp[x, y] = 255
    return clothing, face


def load_model():
    global model, MODEL_READY
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for VASUNDHARA inference")
    print("VASUNDHARA VTON", flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, flush=True)
    print("Model path:", MODEL_PATH, flush=True)
    if not Path(MODEL_PATH).is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {MODEL_PATH}. Upload/mount best.pt before starting Cloud Run."
        )
    model = create_model(device=DEVICE)
    model = load_checkpoint(model, MODEL_PATH, device=DEVICE)
    model.eval()
    MODEL_READY = True
    print("VASUNDHARA checkpoint loaded.", flush=True)


load_model()


class VTONRequest(BaseModel):
    person_image: str | None = None
    saree_image: str | None = None
    model_image: str | None = None
    person: str | None = None
    product_image: str | None = None
    garment_image: str | None = None
    cloth_image: str | None = None
    saree: str | None = None
    garment: str | None = None
    cloth: str | None = None
    clothing_mask: str | None = None
    cloth_mask: str | None = None
    mask: str | None = None
    face_mask: str | None = None
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    seed: int = -1


@app.get("/health")
def health():
    return {"status": "ok", "model": "VASUNDHARA-VTON", "gpu": torch.cuda.get_device_name(0), "ready": MODEL_READY}


@app.post("/")
def inference(request: VTONRequest):
    try:
        person_value = request.person_image or request.model_image or request.person
        saree_value = (
            request.saree_image or request.product_image or request.garment_image
            or request.cloth_image or request.saree or request.garment or request.cloth
        )
        if not person_value:
            raise ValueError("Person image is required")
        if not saree_value:
            raise ValueError("Saree image is required")

        width = max(128, min(int(request.width), 768))
        height = max(128, min(int(request.height), 1024))
        size = (width, height)
        person = decode_image(person_value)
        saree = decode_image(saree_value)

        clothing_mask_value = request.clothing_mask or request.cloth_mask or request.mask
        clothing_mask = decode_mask(clothing_mask_value, size) if clothing_mask_value else None
        face_mask = decode_mask(request.face_mask, size) if request.face_mask else None
        if clothing_mask is None or face_mask is None:
            fallback_clothing, fallback_face = fallback_masks(size)
            clothing_mask = clothing_mask or fallback_clothing
            face_mask = face_mask or fallback_face

        person_tensor = image_to_tensor(person, size).to(DEVICE)
        saree_tensor = image_to_tensor(saree, size).to(DEVICE)
        clothing_tensor = mask_to_tensor(clothing_mask, size).to(DEVICE)
        face_tensor = mask_to_tensor(face_mask, size).to(DEVICE)

        if request.seed >= 0:
            torch.manual_seed(request.seed)
            torch.cuda.manual_seed_all(request.seed)

        with torch.inference_mode():
            output = model(person_tensor, saree_tensor, clothing_tensor, face_tensor)

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
            "seed": request.seed,
        }
    except Exception as exc:
        traceback.print_exc()
        return {"success": False, "error": str(exc)}
