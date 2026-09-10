# Vasundhara VTON

Vasundhara's virtual try-on project. The production inference path is a **RunPod Serverless GPU worker running the own VASUNDHARA-VTON PyTorch model**.

## Production API

The RunPod worker starts from `handler.py` and exposes a Serverless job handler.

The public API accepts:

- `person_image` — person/model image as base64 or a data URI
- `saree_image` — flat-lay saree/product image as base64 or a data URI
- optional `clothing_mask` / `face_mask`
- optional `width`, `height`, and `seed`

Legacy aliases such as `model_image`, `garment_image`, `cloth_image`, `person`, `saree`, `garment`, and `cloth` are also accepted.

Example input:

```json
{
  "input": {
    "person_image": "<base64>",
    "saree_image": "<base64>",
    "width": 384,
    "height": 512,
    "seed": 42
  }
}
```

Successful responses contain a PNG encoded as base64:

```json
{
  "success": true,
  "image": "<base64-png>",
  "format": "png",
  "width": 384,
  "height": 512,
  "model": "VASUNDHARA-VTON"
}
```

Errors are returned as:

```json
{
  "success": false,
  "error": "<message>"
}
```

## RunPod deployment

`Dockerfile` builds the GPU runtime and starts:

```text
python -u /workspace/handler.py
```

The serverless image intentionally contains only the dependencies required by the own VASUNDHARA worker. **CatVTON, Detectron2, DensePose, and Gradio are not part of the production worker.**

The trained checkpoint is expected at:

```text
/workspace/checkpoints/vasundhara-vton/best.pt
```

Set `MODEL_PATH` if the checkpoint is mounted at a different location. For production, keep the checkpoint on the RunPod persistent/network volume rather than committing model weights to GitHub.

## Custom Vasundhara model

`vton/model.py` contains the trainable VASUNDHARA-VTON architecture and `train.py` contains the CUDA training pipeline. Training produces `last.pt` and `best.pt` under `checkpoints/vasundhara-vton/`.

`best.pt` is compatible with the production handler's checkpoint loader because the training checkpoint stores the model weights under the `model` key.

Dataset/training instructions are in `TRAINING.md`.

## Masks

The worker accepts trained `clothing_mask` and `face_mask` inputs. If they are omitted, conservative fallback masks are used so the endpoint can be smoke-tested; these fallbacks are **not** a substitute for production-quality segmentation.

## Important

Gemini can help with dataset QA, filtering and synthetic-data workflows, but it is not the trainer. The actual VTON weights are trained with PyTorch on a CUDA GPU.

Never commit API keys, tokens, private credentials, or large model checkpoints to this repository.
