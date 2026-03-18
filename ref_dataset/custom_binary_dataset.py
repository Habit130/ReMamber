import json
import os

import numpy as np
import torch.utils.data as data
from PIL import Image


class CustomBinaryDataset(data.Dataset):
    def __init__(self, data_root, split="train", image_transforms=None, caption_index=2):
        self.data_root = data_root
        self.split = split
        self.image_transforms = image_transforms
        self.caption_index = caption_index

        annotation_path = os.path.join(self.data_root, f"{self.split}.json")
        with open(annotation_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        self.class_name = "foreground"
        self.category_id = 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = os.path.join(self.data_root, sample["image"])
        mask_path = os.path.join(self.data_root, sample["mask"])

        img = Image.open(image_path).convert("RGB")
        mask = np.array(Image.open(mask_path))
        annot = (mask > 0).astype(np.uint8)
        org_gt = annot.copy()
        annot = Image.fromarray(annot, mode="P")

        if self.image_transforms is not None:
            img, target = self.image_transforms(img, annot)
            target = target.unsqueeze(0)
        else:
            target = annot

        sentence = sample["caption"][self.caption_index]
        if self.split != "train":
            sentence = [sentence]

        return {
            "query_img": img,
            "query_mask": target,
            "query_idx": index,
            "sentence": sentence,
            "category_id": self.category_id,
            "class_name": self.class_name,
            "image_filename": os.path.basename(image_path),
            "org_gt": org_gt,
        }
