```python
import os
import sys
import subprocess

import torch
import gradio as gr
import spaces

# ============================================================
# 1. CatVTON SOURCE
# ============================================================

CATVTON_ROOT = "/home/user/app/CatVTON"

if not os.path.exists(os.path.join(CATVTON_ROOT, "model")):
    print("CatVTON source not found. Cloning...")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/Zheng-Chong/CatVTON.git",
            CATVTON_ROOT,
        ],
        check=True,
    )

if CATVTON_ROOT not in sys.path:
    sys.path.insert(0, CATVTON_ROOT)


# ============================================================
# 2. CATVTON IMPORTS
# ============================================================

from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download

from model.cloth_masker import AutoMasker
from model.pipeline import CatVTONPipeline
from utils import (
    init_weight_dtype,
    resize_and_crop,
    resize_and_padding,
)


# ============================================================
# 3. MODEL CONFIGURATION
# ============================================================

MODEL_REPO = "zhengchong/CatVTON"

BASE_MODEL = "booksforcharlie/stable-diffusion-inpainting"

WIDTH = 768
HEIGHT = 1024

STEPS = 40
GUIDANCE = 2.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("========================================")
print("VASUNDHARA SAREE VIRTUAL TRY-ON")
print("CatVTON")
print("Device:", DEVICE)
print("========================================")


# ============================================================
# 4. MODEL LOADING
# ============================================================

pipeline = None
automasker = None
mask_processor = None
model_error = None


def load_models():

    global pipeline
    global automasker
    global mask_processor
    global model_error

    if DEVICE != "cuda":
        model_error = (
            "CUDA GPU is required. "
            "This application must run inside a Hugging Face "
            "ZeroGPU-enabled Space."
        )

        print(model_error)

        return

    try:

        print("Downloading CatVTON model...")

        repo_path = snapshot_download(
            repo_id=MODEL_REPO
        )

        print("Initializing precision...")

        weight_dtype = init_weight_dtype("bf16")

        print("Loading CatVTON pipeline...")

        pipeline = CatVTONPipeline(
            base_ckpt=BASE_MODEL,
            attn_ckpt=repo_path,
            attn_ckpt_version="mix",
            weight_dtype=weight_dtype,
            use_tf32=True,
            device="cuda",
        )

        print("Creating mask processor...")

        mask_processor = VaeImageProcessor(
            vae_scale_factor=8,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )

        print("Loading AutoMasker...")

        automasker = AutoMasker(
            densepose_ckpt=os.path.join(
                repo_path,
                "DensePose",
            ),
            schp_ckpt=os.path.join(
                repo_path,
                "SCHP",
            ),
            device="cuda",
        )

        print("========================================")
        print("MODEL READY")
        print("========================================")

    except Exception as exc:

        model_error = repr(exc)

        print("MODEL LOAD ERROR:")
        print(model_error)


# Load models when Space starts
load_models()


# ============================================================
# 5. SAREE TRY-ON
# ============================================================

@spaces.GPU(duration=120)
def saree_tryon(
    person_img,
    saree_img,
    seed=42,
):

    if person_img is None:
        raise gr.Error(
            "Please upload a person photo."
        )

    if saree_img is None:
        raise gr.Error(
            "Please upload a saree photo."
        )

    if pipeline is None or automasker is None:
        raise gr.Error(
            "VTON model is not ready. "
            f"{model_error or ''}"
        )

    try:

        # ----------------------------------------------------
        # Convert to RGB
        # ----------------------------------------------------

        person_img = person_img.convert("RGB")
        saree_img = saree_img.convert("RGB")

        # ----------------------------------------------------
        # Prepare person image
        # ----------------------------------------------------

        person = resize_and_crop(
            person_img,
            (WIDTH, HEIGHT),
        )

        # ----------------------------------------------------
        # Prepare saree image
        # ----------------------------------------------------

        garment = resize_and_padding(
            saree_img,
            (WIDTH, HEIGHT),
        )

        # ----------------------------------------------------
        # Generate clothing mask
        # ----------------------------------------------------

        mask_data = automasker(
            person,
            "overall",
        )

        mask = mask_data["mask"]

        mask = mask_processor.blur(
            mask,
            blur_factor=9,
        )

        # ----------------------------------------------------
        # Random/selected seed
        # ----------------------------------------------------

        seed = int(seed)

        generator = torch.Generator(
            device="cuda"
        ).manual_seed(seed)

        # ----------------------------------------------------
        # CatVTON generation
        # ----------------------------------------------------

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

    except torch.cuda.OutOfMemoryError:

        torch.cuda.empty_cache()

        raise gr.Error(
            "GPU memory is insufficient. "
            "Try a smaller image or fewer inference steps."
        )

    except Exception as exc:

        print(
            "TRY-ON ERROR:",
            repr(exc),
        )

        raise gr.Error(
            f"Try-on failed: {exc}"
        )


# ============================================================
# 6. GRADIO USER INTERFACE
# ============================================================

with gr.Blocks(
    title="Vasundhara Saree Virtual Try-On"
) as interface:

    gr.Markdown(
        """
        # 👗 Vasundhara Saree Virtual Try-On

        Upload a **person photo** and a **saree/product photo**.

        CatVTON will generate a virtual try-on result.
        """
    )

    with gr.Row():

        with gr.Column():

            person_input = gr.Image(
                type="pil",
                label="👩 Person Photo",
            )

            saree_input = gr.Image(
                type="pil",
                label="🥻 Saree Photo",
            )

            seed_input = gr.Number(
                value=42,
                precision=0,
                label="Seed",
            )

            try_button = gr.Button(
                "✨ Generate Saree Try-On",
                variant="primary",
            )

        with gr.Column():

            result_output = gr.Image(
                type="pil",
                label="✨ Virtual Saree Result",
            )

    try_button.click(
        fn=saree_tryon,
        inputs=[
            person_input,
            saree_input,
            seed_input,
        ],
        outputs=result_output,
    )


# ============================================================
# 7. START SPACE
# ============================================================

if __name__ == "__main__":

    interface.launch()
```
