# Vasundhara VTON

Vasundhara's virtual try-on project. The production inference path is a **RunPod Serverless GPU worker** using CatVTON with DensePose/SCHP automatic masking.

## Production inference

The RunPod worker starts from `handler.py`.

It accepts a person image and a garment/saree image as base64 (or a data URI) and returns a generated PNG as base64.

Example input:

```json
{
  "input": {
    "person_image": "<base64>",
    "garment_image": "<base64>",
    "cloth_type": "overall",
    "steps": 40,
    "guidance_scale": 2.5,
    "seed": 42
  }
}
```

Supported image aliases include `model_image` / `person_image` and `garment_image` / `cloth_image`.

## Docker / RunPod

`Dockerfile` builds the GPU runtime and starts:

```text
python -u /workspace/handler.py
```

The image contains CatVTON plus Detectron2/DensePose dependencies. Model weights are downloaded at worker startup and cached under `/workspace/huggingface`.

## Custom Vasundhara model

`vton/model.py` contains the trainable Vasundhara VTON architecture and `train.py` contains the CUDA training pipeline. Training is separate from production CatVTON inference until a trained Vasundhara checkpoint has been produced and validated.

Dataset/training instructions are in `TRAINING.md`.

## Important

Gemini can help with dataset QA, filtering and synthetic-data workflows, but it is not the trainer. The actual VTON weights are trained with PyTorch on a CUDA GPU.

Never commit API keys, tokens or private credentials to this repository.
