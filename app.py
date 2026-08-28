import os
import sys
import torch
import gradio as gr
import spaces  # <--- This is the magic Hugging Face library for free GPUs

# 1. Setup CatVTON paths
CATVTON_ROOT = "/home/user/app/CatVTON"
if not os.path.exists(CATVTON_ROOT):
    os.system("git clone https://github.com")
if CATVTON_ROOT not in sys.path:
    sys.path.insert(0, CATVTON_ROOT)

from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from model.cloth_masker import AutoMasker
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

# 2. Download and initialize models globally
repo_path = snapshot_download(repo_id="zhengchong/CatVTON")
weight_dtype = init_weight_dtype("bf16")

pipeline = CatVTONPipeline(
    base_ckpt="booksforcharlie/stable-diffusion-inpainting",
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=weight_dtype,
    use_tf32=True,
    device="cuda",
)

mask_processor = VaeImageProcessor(
    vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True
)

automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device="cuda",
)

# 3. The Core Try-On Function (Wrapped with free GPU decorator)
@spaces.GPU(duration=120)  # Tells Hugging Face to give you a free A100 GPU for this function
def saree_tryon(person_img, saree_img, seed=42):
    if person_img is None or saree_img is None:
        return None
        
    # Process images for Saree sizing
    person = resize_and_crop(person_img.convert("RGB"), (768, 1024))
    garment = resize_and_padding(saree_img.convert("RGB"), (768, 1024))

    # Generate layout mask for the Saree drape
    mask = automasker(person, "overall")["mask"]
    mask = mask_processor.blur(mask, blur_factor=9)

    generator = torch.Generator(device="cuda").manual_seed(int(seed))

    # Run AI Generation
    result = pipeline(
        image=person,
        condition_image=garment,
        mask=mask,
        num_inference_steps=40,
        guidance_scale=2.5,
        height=1024,
        width=768,
        generator=generator,
    )[0]

    return result

# 4. Build a beautiful UI web dashboard automatically
interface = gr.Interface(
    fn=saree_tryon,
    inputs=[
        gr.Image(type="pil", label="Upload Person Photo"),
        gr.Image(type="pil", label="Upload Saree Photo"),
        gr.Number(value=42, label="Seed (Change for different draping styles)")
    ],
    outputs=gr.Image(type="pil", label="Virtual Saree Try-On Result"),
    title="Vasundhara Saree Virtual Try-On",
    description="Upload a full-body photo and a saree image to test the drape completely for free using Hugging Face ZeroGPU!"
)

if __name__ == "__main__":
    interface.launch()
