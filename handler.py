```python
import base64
import io
import traceback

import numpy as np
import runpod
import torch

from PIL import Image

from vton.model import create_model


# =========================================================
# GLOBAL MODEL
# =========================================================

MODEL = None


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():
    global MODEL

    if MODEL is None:

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("========================================", flush=True)
        print("VASUNDHARA VTON V7", flush=True)
        print(f"Device: {device}", flush=True)

        if torch.cuda.is_available():
            print(
                f"GPU: {torch.cuda.get_device_name(0)}",
                flush=True,
            )

        print("Creating model...", flush=True)

        MODEL = create_model(device)

        MODEL.eval()

        print("VASUNDHARA MODEL LOADED", flush=True)
        print("========================================", flush=True)

    return MODEL


# =========================================================
# DECODE BASE64 IMAGE
# =========================================================

def decode_image(value):

    if not value:
        raise ValueError("Image is missing")

    if value.startswith("data:image"):
        value = value.split(",", 1)[1]

    try:
        data = base64.b64decode(value)
    except Exception as exc:
        raise ValueError(
            f"Invalid base64 image: {exc}"
        )

    try:
        image = Image.open(
            io.BytesIO(data)
        ).convert("RGB")
    except Exception as exc:
        raise ValueError(
            f"Invalid image data: {exc}"
        )

    return image


# =========================================================
# ENCODE IMAGE
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
# HEALTH TEST
# =========================================================

def health_test():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    gpu = None

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)

    return {
        "status": "ok",
        "service": "VASUNDHARA VTON",
        "version": "V7",
        "device": device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
    }


# =========================================================
# MAIN HANDLER
# =========================================================

def handler(job):

    print("========================================", flush=True)
    print("JOB RECEIVED", flush=True)
    print("========================================", flush=True)

    try:

        # -------------------------------------------------
        # JOB INPUT
        # -------------------------------------------------

        data = job.get("input", {})

        if not isinstance(data, dict):
            raise ValueError(
                "input must be an object"
            )

        # -------------------------------------------------
        # HEALTH CHECK
        # -------------------------------------------------

        if data.get("test") is True:

            print(
                "Running worker health test...",
                flush=True,
            )

            result = health_test()

            print(
                "Health test successful.",
                flush=True,
            )

            return result

        # -------------------------------------------------
        # REQUIRED INPUTS
        # -------------------------------------------------

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

        # -------------------------------------------------
        # DECODE PERSON
        # -------------------------------------------------

        print(
            "Decoding person image...",
            flush=True,
        )

        person = decode_image(
            person_data
        )

        print(
            f"Person image: {person.size}",
            flush=True,
        )

        # -------------------------------------------------
        # DECODE GARMENT
        # -------------------------------------------------

        print(
            "Decoding garment image...",
            flush=True,
        )

        garment = decode_image(
            garment_data
        )

        print(
            f"Garment image: {garment.size}",
            flush=True,
        )

        # -------------------------------------------------
        # LOAD MODEL
        # -------------------------------------------------

        print(
            "Loading VASUNDHARA model...",
            flush=True,
        )

        model = load_model()

        # -------------------------------------------------
        # MODEL SIZE
        # -------------------------------------------------

        size = (512, 768)

        print(
            f"Resizing images to {size}...",
            flush=True,
        )

        person = person.resize(
            size,
            Image.Resampling.LANCZOS,
        )

        garment = garment.resize(
            size,
            Image.Resampling.LANCZOS,
        )

        # -------------------------------------------------
        # NUMPY
        # -------------------------------------------------

        person_array = np.asarray(
            person,
            dtype=np.float32,
        )

        garment_array = np.asarray(
            garment,
            dtype=np.float32,
        )

        # -------------------------------------------------
        # PERSON TENSOR
        # -------------------------------------------------

        person_tensor = (
            torch.from_numpy(
                person_array
            )
            .permute(2, 0, 1)
            .contiguous()
            / 255.0
        )

        # -------------------------------------------------
        # GARMENT TENSOR
        # -------------------------------------------------

        garment_tensor = (
            torch.from_numpy(
                garment_array
            )
            .permute(2, 0, 1)
            .contiguous()
            / 255.0
        )

        # -------------------------------------------------
        # NORMALIZATION
        # -------------------------------------------------

        person_tensor = (
            person_tensor * 2.0 - 1.0
        )

        garment_tensor = (
            garment_tensor * 2.0 - 1.0
        )

        # -------------------------------------------------
        # TEMPORARY MASKS
        #
        # These are placeholders for the current V7
        # model interface.
        # -------------------------------------------------

        clothing_mask = torch.ones(
            1,
            768,
            512,
            dtype=torch.float32,
        )

        face_mask = torch.ones(
            1,
            768,
            512,
            dtype=torch.float32,
        )

        # -------------------------------------------------
        # MODEL DEVICE
        # -------------------------------------------------

        try:
            device = next(
                model.parameters()
            ).device
        except StopIteration:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        print(
            f"Model device: {device}",
            flush=True,
        )

        # -------------------------------------------------
        # ADD BATCH DIMENSION
        # -------------------------------------------------

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

        # -------------------------------------------------
        # INFERENCE
        # -------------------------------------------------

        print(
            "Starting VTON inference...",
            flush=True,
        )

        with torch.no_grad():

            output = model(
                person_tensor,
                garment_tensor,
                clothing_mask,
                face_mask,
            )

        print(
            "VTON inference completed.",
            flush=True,
        )

        # -------------------------------------------------
        # HANDLE MODEL OUTPUT
        # -------------------------------------------------

        if isinstance(
            output,
            (tuple, list)
        ):
            output = output[0]

        if not isinstance(
            output,
            torch.Tensor
        ):
            raise TypeError(
                "Model output is not a torch.Tensor"
            )

        # -------------------------------------------------
        # REMOVE BATCH DIMENSION
        # -------------------------------------------------

        if output.ndim == 4:

            output = output[0]

        elif output.ndim != 3:

            raise ValueError(
                f"Unexpected model output shape: "
                f"{tuple(output.shape)}"
            )

        # -------------------------------------------------
        # CONVERT [-1, 1] -> [0, 255]
        # -------------------------------------------------

        output = (
            output
            .clamp(-1, 1)
            .add(1)
            .div(2)
            .mul(255)
            .byte()
        )

        # -------------------------------------------------
        # CHW -> HWC
        # -------------------------------------------------

        output = (
            output
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )

        # -------------------------------------------------
        # PIL IMAGE
        # -------------------------------------------------

        result = Image.fromarray(
            output,
            mode="RGB",
        )

        print(
            f"Output image: {result.size}",
            flush=True,
        )

        # -------------------------------------------------
        # RETURN RESULT
        # -------------------------------------------------

        return {
            "status": "success",
            "service": "VASUNDHARA VTON",
            "version": "V7",
            "image": encode_image(
                result
            ),
        }

    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as exc:

        print("========================================", flush=True)
        print("VASUNDHARA ERROR", flush=True)
        print("========================================", flush=True)

        traceback.print_exc()

        return {
            "status": "error",
            "service": "VASUNDHARA VTON",
            "version": "V7",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


# =========================================================
# START WORKER
# =========================================================

print("========================================", flush=True)
print("VASUNDHARA VTON V7 WORKER", flush=True)
print("Starting RunPod worker...", flush=True)
print("========================================", flush=True)

print(
    f"Python/PyTorch: {torch.__version__}",
    flush=True,
)

print(
    f"CUDA version: {torch.version.cuda}",
    flush=True,
)

print(
    f"CUDA available: {torch.cuda.is_available()}",
    flush=True,
)

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}",
        flush=True,
    )


print(
    "Starting serverless handler...",
    flush=True,
)


runpod.serverless.start(
    {
        "handler": handler
    }
)


print(
    "WARNING: RunPod serverless handler exited.",
    flush=True,
)
```
