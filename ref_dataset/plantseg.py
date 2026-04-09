import json
from pathlib import Path

import numpy as np
import torch.utils.data as data
from PIL import Image


class PlantSegDataset(data.Dataset):
    def __init__(self, data_root, split="train", image_transforms=None, caption_index=3):
        self.root = Path(data_root).expanduser()
        self.split = split
        self.image_transforms = image_transforms
        self.caption_index = caption_index

        metadata_path = self.root / "main.json"
        records = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.samples = [sample for sample in records if sample["split"] == split]

        if not self.samples:
            raise ValueError(f"No PlantSeg samples found for split={split} under {metadata_path}")

        first_caption_count = len(self.samples[0]["caption"])
        if caption_index < 0 or caption_index >= first_caption_count:
            raise ValueError(
                f"caption_index={caption_index} is out of range for PlantSeg captions of length {first_caption_count}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = self.root / sample["image"]
        mask_path = self.root / sample["mask"]

        img = Image.open(image_path).convert("RGB")
        mask = np.array(Image.open(mask_path))
        mask = (mask > 0).astype(np.uint8)
        org_gt = mask.copy()
        target = Image.fromarray(mask, mode="P")

        if self.image_transforms is not None:
            img, target = self.image_transforms(img, target)
            target = target.unsqueeze(0)

        sentence = sample["caption"][self.caption_index]
        if self.split != "train":
            sentence = [sentence]

        return {
            "query_img": img,
            "query_mask": target,
            "query_idx": index,
            "sentence": sentence,
            "category_id": 0,
            "class_name": sample["disease_label"],
            "image_filename": image_path.name,
            "mask_path": str(mask_path),
            "org_gt": org_gt,
        }
