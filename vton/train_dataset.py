import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision import transforms


class VTONDataset(Dataset):
    """Paired VTON dataset.

    Expected directory layout:
      root/person, root/garment, root/target,
      root/clothing_mask, root/face_mask

    Files are matched by filename stem. A manifest.csv/json can instead
    provide person, garment, target, clothing_mask and face_mask paths.
    """

    REQUIRED_FIELDS = ["person", "garment", "target", "clothing_mask", "face_mask"]
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

    def __init__(self, root: str, size=(384, 512), manifest: Optional[str] = None):
        self.root = Path(root)
        self.width, self.height = size
        self.samples: List[Dict[str, str]] = []
        if manifest:
            self._load_manifest(Path(manifest))
        else:
            self._load_directories()
        if not self.samples:
            raise RuntimeError(f"No paired samples found under {self.root}")

        self.rgb = transforms.Compose([
            transforms.Resize((self.height, self.width), InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        # Masks must be single-channel. Converting them to RGB would produce
        # 3 channels and break the model's 5-channel person input.
        self.mask = transforms.Compose([
            transforms.Resize((self.height, self.width), InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def _load_manifest(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".json":
            rows = json.loads(path.read_text())
        else:
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        if not isinstance(rows, list):
            raise ValueError("Manifest JSON must contain a list of sample objects")

        for row in rows:
            if not all(row.get(k) for k in self.REQUIRED_FIELDS):
                raise ValueError(f"Manifest row missing required fields: {self.REQUIRED_FIELDS}")
            self.samples.append({k: str(self.root / row[k]) for k in self.REQUIRED_FIELDS})

    def _load_directories(self):
        dirs = {k: self.root / k for k in self.REQUIRED_FIELDS}
        if not all(p.is_dir() for p in dirs.values()):
            raise FileNotFoundError(
                "Dataset must contain person, garment, target, clothing_mask and face_mask directories"
            )

        targets = {
            p.stem: p
            for p in dirs["target"].iterdir()
            if p.suffix.lower() in self.IMAGE_EXTS
        }
        for stem, target in sorted(targets.items()):
            row = {"target": str(target)}
            ok = True
            for k in ["person", "garment", "clothing_mask", "face_mask"]:
                found = next(
                    (dirs[k] / (stem + ext) for ext in self.IMAGE_EXTS if (dirs[k] / (stem + ext)).exists()),
                    None,
                )
                if found is None:
                    ok = False
                    break
                row[k] = str(found)
            if ok:
                self.samples.append(row)

    @staticmethod
    def _open_rgb(path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    @staticmethod
    def _open_mask(path: str) -> Image.Image:
        # Preserve a true single-channel mask regardless of source PNG mode.
        return Image.open(path).convert("L")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        s = self.samples[index]
        return {
            "person": self.rgb(self._open_rgb(s["person"])),
            "garment": self.rgb(self._open_rgb(s["garment"])),
            "target": self.rgb(self._open_rgb(s["target"])),
            "clothing_mask": self.mask(self._open_mask(s["clothing_mask"])),
            "face_mask": self.mask(self._open_mask(s["face_mask"])),
        }
