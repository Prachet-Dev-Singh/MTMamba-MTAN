"""
NYUv2MTAN dataset adapter for MTMamba.
Supports MTAN pre-processed NYUv2 benchmark (288x384 .npy format).
Tasks:
  - semseg: 13 semantic classes
  - depth: 1-channel metric depth
  - normals: 3-channel surface normal vectors
"""

import os
import glob
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as t_transforms

NYU_MTAN_CATEGORY_NAMES = [
    "bed", "books", "ceiling", "chair", "floor", "furniture",
    "objects", "painting", "sofa", "table", "tv", "wall", "window"
]

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


class NYUv2MTAN(data.Dataset):
    N_CLASSES = 13
    DEPTH_INVALID_IN = 0.0
    DEPTH_INVALID_OUT = 255.0  # Remap to 255.0 so L1Loss(ignore_index=255) ignores it

    def __init__(self, root, split="train", task_list=None, ignore_index=255, augment=False):
        super().__init__()
        self.root = root
        self.split = split
        self.task_list = task_list or ["semseg", "depth"]
        self.ignore_index = ignore_index
        self.augment = augment and (split == "train")

        sd = os.path.join(root, split)
        if not os.path.isdir(sd):
            raise FileNotFoundError(f"NYUv2 split folder not found at: {sd}")

        # Find image files
        self.image_files = sorted(glob.glob(os.path.join(sd, "image", "*.npy")))
        if not self.image_files:
            raise FileNotFoundError(f"No image .npy files found in {sd}/image/")

        # Find label files (support 'label' or 'label_13')
        lbl_dir = os.path.join(sd, "label")
        if not os.path.exists(lbl_dir):
            lbl_dir = os.path.join(sd, "label_13")
        self.label_files = sorted(glob.glob(os.path.join(lbl_dir, "*.npy"))) if os.path.exists(lbl_dir) else []

        # Find depth files
        depth_dir = os.path.join(sd, "depth")
        self.depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.npy"))) if os.path.exists(depth_dir) else []

        # Find normal files (support 'normal' or 'normals')
        normal_dir = os.path.join(sd, "normal")
        if not os.path.exists(normal_dir):
            normal_dir = os.path.join(sd, "normals")
        self.normal_files = sorted(glob.glob(os.path.join(normal_dir, "*.npy"))) if os.path.exists(normal_dir) else []

        self.img_transform = t_transforms.Compose([
            t_transforms.ToTensor(),
            t_transforms.Normalize(mean=_MEAN, std=_STD)
        ])

        print(f"[NYUv2MTAN] split={split}, n={len(self.image_files)}, tasks={self.task_list}, augment={self.augment}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = self.image_files[index]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img = np.load(img_path)

        # Handle float [0, 1] vs uint8 [0, 255]
        if img.dtype != np.uint8:
            if img.max() <= 1.5:
                img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        # Transpose to (H, W, 3) if stored as (3, H, W)
        if img.ndim == 3 and img.shape[0] == 3 and img.shape[2] != 3:
            img = np.moveaxis(img, 0, -1)

        H, W = img.shape[:2]
        lbl = dep = nor = None

        if "semseg" in self.task_list and self.label_files:
            lbl = np.load(self.label_files[index]).astype(np.int64)
            # MTAN uses -1 for unlabeled / background pixels; map to ignore_index (255)
            lbl[lbl == -1] = self.ignore_index

        if "depth" in self.task_list and self.depth_files:
            dep = np.load(self.depth_files[index]).astype(np.float32)
            if dep.ndim == 3:
                dep = dep.squeeze(0) if dep.shape[0] == 1 else dep.squeeze(-1)
            # Map invalid depth (<= 0) to ignore_index
            dep[dep <= self.DEPTH_INVALID_IN] = self.DEPTH_INVALID_OUT

        if "normals" in self.task_list and self.normal_files:
            nor = np.load(self.normal_files[index]).astype(np.float32)
            # Normalize to (3, H, W)
            if nor.ndim == 3 and nor.shape[-1] == 3:
                nor = np.moveaxis(nor, -1, 0)

        # Horizontal flip data augmentation
        if self.augment and np.random.rand() < 0.5:
            img = np.fliplr(img).copy()
            if lbl is not None:
                lbl = np.fliplr(lbl).copy()
            if dep is not None:
                dep = np.fliplr(dep).copy()
            if nor is not None:
                nor = np.flip(nor, axis=-1).copy()
                nor[0] = -nor[0]  # Flip x-component of surface normal

        sample = {"image": self.img_transform(img)}
        if "semseg" in self.task_list and lbl is not None:
            sample["semseg"] = torch.from_numpy(lbl).long()
        if "depth" in self.task_list and dep is not None:
            sample["depth"] = torch.from_numpy(dep).float().unsqueeze(0)
        if "normals" in self.task_list and nor is not None:
            sample["normals"] = torch.from_numpy(nor).float()

        sample["meta"] = {"img_name": stem, "img_size": (H, W), "scale_factor": np.array([1.0, 1.0])}
        return sample
