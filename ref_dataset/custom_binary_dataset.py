import json
from pathlib import Path

import numpy as np
import torch.utils.data as data
from PIL import Image


class CustomBinaryDataset(data.Dataset):
    def __init__(
        self,
        data_root="../dataset",
        split="train",
        image_transforms=None,
        caption_index=2,
    ):
        self.root = Path(data_root)
        self.split = split
        self.image_transforms = image_transforms
        self.caption_index = caption_index

        index_file = self.root / f"{split}.json"
        if not index_file.exists():
            raise FileNotFoundError(f"Missing dataset index: {index_file}")

        self.samples = json.loads(index_file.read_text(encoding="utf-8"))
        if not isinstance(self.samples, list):
            raise ValueError(f"Dataset index must be a list: {index_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = self.root / sample["image"]
        mask_path = self.root / sample["mask"]

        image = Image.open(image_path).convert("RGB")
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask.max(axis=-1)
        mask = (mask > 0).astype(np.uint8)
        org_gt = mask.astype(bool)
        target = Image.fromarray(mask, mode="P")

        if self.image_transforms is not None:
            image, target = self.image_transforms(image, target)
            target = target.unsqueeze(0)

        captions = sample["caption"]
        if len(captions) <= self.caption_index:
            raise IndexError(
                f"Sample {sample['id']} does not have caption[{self.caption_index}]"
            )
        selected_caption = captions[self.caption_index]

        sentence = selected_caption if self.split == "train" else [selected_caption]

        return {
            "query_img": image,
            "query_mask": target,
            "query_idx": index,
            "sentence": sentence,
            "category_id": 1,
            "class_name": "foreground",
            "image_filename": sample["image"],
            "org_gt": org_gt,
            "sample_id": sample["id"],
        }
