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
        self.mask = transforms.Compose([
            transforms.Resize((self.height, self.width), InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def _load_manifest(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(path)
        rows = json.loads(path.read_text()) if path.suffix.lower() == ".json" else list(csv.DictReader(path.open()))
        for row in rows:
            required = ["person", "garment", "target", "clothing_mask", "face_mask"]
            if not all(row.get(k) for k in required):
                raise ValueError(f"Manifest row missing required fields: {required}")
            self.samples.append({k: str(self.root / row[k]) for k in required})

    def _load_directories(self):
        dirs = {k: self.root / k for k in ["person", "garment", "target", "clothing_mask", "face_mask"]}
        if not all(p.is_dir() for p in dirs.values()):
            raise FileNotFoundError("Dataset must contain person, garment, target, clothing_mask and face_mask directories")
        target_exts = {".png", ".jpg", ".jpeg", ".webp"}
        targets = {p.stem: p for p in dirs["target"].iterdir() if p.suffix.lower() in target_exts}
        for stem, target in sorted(targets.items()):
            row = {"target": str(target)}
            ok = True
            for k in ["person", "garment", "clothing_mask", "face_mask"]:
                matches = [dirs[k] / (stem + ext) for ext in target_exts]
                found = next((p for p in matches if p.exists()), None)
                if found is None:
                    ok = False
                    break
                row[k] = str(found)
            if ok:
                self.samples.append(row)

    @staticmethod
    def _open(path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        s = self.samples[index]
        return {
            "person": self.rgb(self._open(s["person"])),
            "garment": self.rgb(self._open(s["garment"])),
            "target": self.rgb(self._open(s["target"])),
            "clothing_mask": self.mask(self._open(s["clothing_mask"])),
            "face_mask": self.mask(self._open(s["face_mask"])),
        }
