import os
import sys
import io
import time
import base64
import threading
from uuid import uuid4

from flask import Flask, jsonify, request, render_template
from PIL import Image

import torch

# CatVTON is cloned into /opt/CatVTON by the Dockerfile.
CATVTON_ROOT = os.environ.get("CATVTON_ROOT", "/opt/CatVTON")
if CATVTON_ROOT not in sys.path:
    sys.path.insert(0, CATVTON_ROOT)

from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from model.cloth_masker import AutoMasker
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding


app = Flask(__name__)

MODEL_REPO = os.environ.get("CATVTON_MODEL_REPO", "zhengchong/CatVTON")
BASE_MODEL = os.environ.get(
    "CATVTON_BASE_MODEL",
    "booksforcharlie/stable-diffusion-inpainting",
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/vton-output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WIDTH = int(os.environ.get("VTON_WIDTH", "768"))
HEIGHT = int(os.environ.get("VTON_HEIGHT", "1024"))
STEPS = int(os.environ.get("VTON_STEPS", "50"))
GUIDANCE = float(os.environ.get("VTON_GUIDANCE", "2.5"))
MIXED_PRECISION = os.environ.get("VTON_PRECISION", "bf16")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = None
automasker = None
mask_processor = None
model_lock = threading.Lock()
model_error = None


def load_models():
    global pipeline, automasker, mask_processor, model_error

    if DEVICE != "cuda":
        model_error = "GPU is required. Deploy this service with an NVIDIA GPU."
        return

    try:
        print("========================================")
        print("VASUNDHARA OWN VTON")
        print("ENGINE: CatVTON")
        print("DEVICE:", DEVICE)
        print("GPU:", torch.cuda.get_device_name(0))
        print("========================================", flush=True)

        repo_path = snapshot_download(repo_id=MODEL_REPO)

        weight_dtype = init_weight_dtype(MIXED_PRECISION)

        pipeline = CatVTONPipeline(
            base_ckpt=BASE_MODEL,
            attn_ckpt=repo_path,
            attn_ckpt_version="mix",
            weight_dtype=weight_dtype,
            use_tf32=True,
            device="cuda",
        )

        mask_processor = VaeImageProcessor(
            vae_scale_factor=8,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )

        automasker = AutoMasker(
            densepose_ckpt=os.path.join(repo_path, "DensePose"),
            schp_ckpt=os.path.join(repo_path, "SCHP"),
            device="cuda",
        )

        print("MODEL READY", flush=True)

    except Exception as exc:
        model_error = repr(exc)
        print("MODEL LOAD ERROR:", model_error, flush=True)


# Load once when the container starts.
load_models()


def read_image(field_names):
    for name in field_names:
        f = request.files.get(name)
        if f is not None:
            raw = f.read()
            if raw:
                return Image.open(io.BytesIO(raw)).convert("RGB"), f.filename
    return None, None


def image_to_data_uri(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return "data:image/png;base64," + encoded


def make_result(person, garment, seed):
    # CatVTON expects portrait-oriented 768x1024 input.
    person = resize_and_crop(person, (WIDTH, HEIGHT))
    garment = resize_and_padding(garment, (WIDTH, HEIGHT))

    # "overall" is the closest built-in mode for a full saree.
    mask = automasker(person, "overall")["mask"]
    mask = mask_processor.blur(mask, blur_factor=9)

    generator = None
    if seed >= 0:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipeline(
        image=person,
        condition_image=garment,
        mask=mask,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        height=HEIGHT,
        width=WIDTH,
        generator=generator,
    )[0]

    return result


@app.route("/", methods=["GET"])
def home():
    try:
        return render_template("index.html")
    except Exception as exc:
        return f"""
        <!doctype html>
        <html><body>
        <h1>VASUNDHARA OWN VTON API</h1>
        <p>Backend is running.</p>
        <pre>{exc}</pre>
        </body></html>
        """


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "message": "VASUNDHARA OWN VTON SERVER IS RUNNING",
        "provider": "vasundhara-own",
        "model": "CatVTON",
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "model_ready": pipeline is not None,
        "model_error": model_error,
    })


@app.route("/debug", methods=["GET"])
def debug():
    return jsonify({
        "ok": True,
        "provider": "vasundhara-own",
        "model": "CatVTON",
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "model_ready": pipeline is not None,
        "model_error": model_error,
        "routes": [str(rule) for rule in app.url_map.iter_rules()],
    })


@app.route("/api/tryon", methods=["POST"])
@app.route("/api/fashn-tryon", methods=["POST"])
def tryon():
    started = time.time()

    if pipeline is None or automasker is None:
        return jsonify({
            "success": False,
            "error": "VTON model is not ready.",
            "details": model_error,
        }), 503

    person, person_name = read_image(
        ["model_image", "person", "person_image"]
    )
    garment, garment_name = read_image(
        ["garment_image", "garment", "product_image", "saree"]
    )

    if person is None:
        return jsonify({
            "success": False,
            "error": "model_image/person image is missing.",
        }), 400

    if garment is None:
        return jsonify({
            "success": False,
            "error": "garment_image/saree image is missing.",
        }), 400

    try:
        seed_text = request.form.get("seed", "42")
        try:
            seed = int(seed_text)
        except Exception:
            seed = 42

        # Keep compatibility with the existing website.
        hairstyle = request.form.get("hairstyle", "Original")
        background = request.form.get("background", "Original")

        print("========================================", flush=True)
        print("NEW OWN VTON REQUEST", flush=True)
        print("PERSON:", person_name, flush=True)
        print("SAREE:", garment_name, flush=True)
        print("HAIRSTYLE:", hairstyle, flush=True)
        print("BACKGROUND:", background, flush=True)
        print("SEED:", seed, flush=True)
        print("========================================", flush=True)

        # A GPU instance should process one generation at a time.
        with model_lock:
            result = make_result(person, garment, seed)

        output_id = str(uuid4())
        output_path = os.path.join(OUTPUT_DIR, output_id + ".png")
        result.save(output_path)

        data_uri = image_to_data_uri(result)
        elapsed = round(time.time() - started, 2)

        return jsonify({
            "success": True,
            "provider": "vasundhara-own",
            "model": "CatVTON",
            "prediction_id": output_id,
            "image_url": data_uri,
            "images": [data_uri],
            "hairstyle": hairstyle,
            "background": background,
            "resolution": f"{WIDTH}x{HEIGHT}",
            "generation_mode": "quality",
            "num_images": 1,
            "elapsed_seconds": elapsed,
        })

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return jsonify({
            "success": False,
            "error": "GPU out of memory. Reduce VTON_WIDTH/VTON_HEIGHT or use a larger GPU.",
        }), 507

    except Exception as exc:
        print("VTON ERROR:", repr(exc), flush=True)
        return jsonify({
            "success": False,
            "error": "Own VTON generation failed.",
            "details": repr(exc),
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Flask 404",
        "requested_path": request.path,
    }), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, threaded=True)
