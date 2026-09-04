# Vasundhara VTON training

This repository now contains a trainable Vasundhara VTON baseline in `vton/model.py` plus a paired-data training pipeline.

## Dataset

Create:

```text
data/
  train/
    person/
    garment/
    target/
    clothing_mask/
    face_mask/
```

Each sample must use the same filename stem in all five folders, for example:

```text
person/0001.jpg
garment/0001.jpg
target/0001.jpg
clothing_mask/0001.png
face_mask/0001.png
```

`target` is the ground-truth image of the person wearing the garment. Masks should be binary/grayscale PNGs.

## Train

On the GPU machine/container:

```bash
pip install -r requirements-train.txt
python train.py --data-root data/train --epochs 50 --batch-size 2 --width 384 --height 512
```

Checkpoints are written to `checkpoints/vasundhara-vton/last.pt` and `best.pt`.

## Important

This is the actual PyTorch training stage; Gemini is not the trainer. Gemini can assist with dataset QA, descriptions, filtering and synthetic-data workflows, while these files train the Vasundhara model weights on a CUDA GPU.

For a production saree model, first establish a properly licensed paired saree dataset. General apparel data can be used for pretraining, followed by saree-specific fine-tuning. Do not treat generated images as perfect ground truth without QA.
