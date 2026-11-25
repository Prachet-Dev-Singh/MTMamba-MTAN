"""
Reliable NYUv2 MTAN Dataset Downloader
Fetches the exact MTAN preprocessed NYUv2 benchmark (288x384, 13 classes, depth, normals)
from Hugging Face and exports it to the standard .npy folder structure at /kaggle/working/data/nyuv2.
"""

import os
import argparse
import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def prepare_nyuv2(output_dir="/kaggle/working/data/nyuv2"):
    train_dir = os.path.join(output_dir, "train", "image")
    val_dir = os.path.join(output_dir, "val", "image")
    
    # Check if already exported
    if os.path.exists(train_dir) and len(os.listdir(train_dir)) == 795:
        print(f"[NYUv2] Dataset already present at: {output_dir}")
        return output_dir

    print(f"\n[NYUv2] Loading official MTAN NYUv2 dataset from Hugging Face...")
    ds = load_dataset("tanganke/nyuv2")

    for split in ["train", "val"]:
        for sub in ["image", "label", "depth", "normal"]:
            os.makedirs(os.path.join(output_dir, split, sub), exist_ok=True)

        n_samples = len(ds[split])
        print(f"\n[NYUv2] Exporting {split} split ({n_samples} scenes) to .npy...")
        for idx in tqdm(range(n_samples)):
            item = ds[split][idx]
            
            # 1. Image: (3, 288, 384) float -> (288, 384, 3) uint8
            img = np.array(item["image"], dtype=np.float32)
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.moveaxis(img, 0, -1)
            if img.max() <= 1.5:
                img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

            # 2. Semantic segmentation: (288, 384) int64
            lbl = np.array(item["segmentation"], dtype=np.int64)

            # 3. Depth map: (1, 288, 384) float32
            dep = np.array(item["depth"], dtype=np.float32)
            if dep.ndim == 3 and dep.shape[0] == 1:
                dep = dep.squeeze(0)

            # 4. Surface normals: (3, 288, 384) float32
            nor = np.array(item["normal"], dtype=np.float32)
            if nor.ndim == 3 and nor.shape[-1] == 3:
                nor = np.moveaxis(nor, -1, 0)

            np.save(os.path.join(output_dir, split, "image", f"{idx}.npy"), img)
            np.save(os.path.join(output_dir, split, "label", f"{idx}.npy"), lbl)
            np.save(os.path.join(output_dir, split, "depth", f"{idx}.npy"), dep)
            np.save(os.path.join(output_dir, split, "normal", f"{idx}.npy"), nor)

    print(f"\n[NYUv2] Successfully exported MTAN NYUv2 dataset to: {output_dir}")
    print(f"  - Train scenes: {len(os.listdir(train_dir))}")
    print(f"  - Val scenes:   {len(os.listdir(val_dir))}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/data/nyuv2")
    args = parser.parse_args()
    
    prepare_nyuv2(args.output_dir)
