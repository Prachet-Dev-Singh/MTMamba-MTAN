"""
Standalone Evaluation Script for MTMamba (Swin-Large & ResNet-50)

Computes the full set of authentic metrics matching the CNN U-Net and Mamba-UNet tables:
  - Loss (Linear Multi-Task Loss = L_semseg + L_depth, identical to Part 1 & 2 benchmarks)
  - mIoU (Mean Intersection over Union / Macro-average)
  - Overall IoU (Global Intersection over Union / Micro-average)
  - Pixel Accuracy (Global Correct Pixel Percentage)
  - Abs Rel (%) (Absolute Relative Depth Error)
  - RMSE (Root Mean Squared Depth Error)
  - Log RMSE & Sq Rel

Usage on Kaggle:
    python evaluate_complete.py --config configs/mtmamba_cityscapes_mtan_resnet.yml
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.config import create_config
from utils.common_config import get_model, get_test_dataset, get_test_dataloader, get_criterion
from evaluation.evaluate_utils import PerformanceMeter
from utils.utils import get_output


def auto_detect_dataset(p):
    """Locate CityscapesMTAN dataset directory containing val/image/."""
    candidates = [
        p.get('db_paths', {}).get('CityscapesMTAN', ''),
        '/data',
        '/kaggle/input/cityscapes-mtan-dataset',
        '/kaggle/input/datasets/prachetdevsingh/cityscapes-mtan-dataset',
    ]
    for c in candidates:
        if c and os.path.exists(os.path.join(c, 'val', 'image')):
            p.db_paths['CityscapesMTAN'] = c
            print(f"[Evaluate] Detected CityscapesMTAN dataset at: {c}")
            return c

    # Recursive search under /kaggle/input if not in standard candidates
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            if 'image' in dirs and os.path.basename(root) == 'val':
                dataset_dir = os.path.dirname(root)
                p.db_paths['CityscapesMTAN'] = dataset_dir
                print(f"[Evaluate] Auto-discovered CityscapesMTAN dataset at: {dataset_dir}")
                return dataset_dir

    print(f"[Evaluate] Warning: Using default dataset path '{p.db_paths.get('CityscapesMTAN')}'")
    return p.db_paths.get('CityscapesMTAN')


def auto_detect_checkpoint(checkpoint_path, config_path):
    """Find the checkpoint (.pth.tar) across /kaggle/working and /kaggle/input."""
    if checkpoint_path and os.path.isfile(checkpoint_path):
        return checkpoint_path

    print(f"[Evaluate] Checkpoint '{checkpoint_path}' not found directly. Searching /kaggle...")
    search_dirs = ['/kaggle/working', '/kaggle/input', '.', 'runs']
    found = []
    for sdir in search_dirs:
        if os.path.exists(sdir):
            for root, dirs, files in os.walk(sdir):
                for f in files:
                    if f.endswith('.pth.tar'):
                        found.append(os.path.join(root, f))

    if not found:
        raise FileNotFoundError(f"No .pth.tar checkpoints found under {search_dirs}")

    # Prioritize matching backbone name
    is_resnet = 'resnet' in config_path.lower()
    matching = [x for x in found if ('resnet' in x.lower()) == is_resnet]
    if not matching:
        matching = found

    # Prefer 'checkpoint.pth.tar' or 'best'
    prioritized = [x for x in matching if 'checkpoint' in os.path.basename(x) or 'best' in os.path.basename(x)]
    chosen = prioritized[0] if prioritized else matching[0]
    print(f"[Evaluate] Auto-detected checkpoint: {chosen}")
    return chosen


@torch.no_grad()
def evaluate_checkpoint(config_path, checkpoint_path=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    params = {'run_mode': 'test'}
    p = create_config(config_path, params)
    
    # 1. Dataset path
    auto_detect_dataset(p)
    
    # 2. Checkpoint path
    checkpoint_path = auto_detect_checkpoint(checkpoint_path, config_path)
    
    print(f"\n[Evaluate] Loading config from: {config_path}")
    print(f"[Evaluate] Loading checkpoint from: {checkpoint_path}")
    print(f"[Evaluate] Compute device: {device}")
    
    # 3. Dataset & Dataloader
    test_dataset = get_test_dataset(p)
    test_loader = get_test_dataloader(p, test_dataset)
    print(f"[Evaluate] Validation images count: {len(test_dataset)}")
    
    # 4. Model & Checkpoint weights
    model = get_model(p)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "")
        clean_state_dict[clean_k] = v
    model.load_state_dict(clean_state_dict, strict=True)
    print("[Evaluate] Checkpoint loaded successfully into model!")
    
    model = model.to(device)
    model.eval()
    
    # 5. Loss criterion & Evaluation Meters
    criterion = get_criterion(p).to(device)
    performance_meter = PerformanceMeter(p, p.TASKS.NAMES)
    
    total_linear_loss = 0.0
    total_scheme_loss = 0.0
    n_batches = 0
    
    # Segmentation pixel confusion accumulator for Overall IoU and Pixel Accuracy
    n_classes = 7
    class_names = ["flat", "construction", "object", "nature", "sky", "human", "vehicle"]
    total_tp = np.zeros(n_classes, dtype=np.int64)
    total_fp = np.zeros(n_classes, dtype=np.int64)
    total_fn = np.zeros(n_classes, dtype=np.int64)
    
    print("[Evaluate] Running validation pass across all 500 images...")
    for batch in tqdm(test_loader):
        images = batch['image'].to(device, non_blocking=True)
        targets = {task: batch[task].to(device, non_blocking=True) for task in p.TASKS.NAMES}
        
        output = model(images)
        loss_dict = criterion(output, targets, tasks=p.TASKS.NAMES)
        
        # Linear multi-task loss (L_semseg + L_depth) matching CNN U-Net & Mamba-UNet benchmark standard
        l_sem = loss_dict['semseg'].item() if 'semseg' in loss_dict else 0.0
        l_dep = loss_dict['depth'].item() if 'depth' in loss_dict else 0.0
        total_linear_loss += (l_sem + l_dep)
        
        if 'total' in loss_dict:
            total_scheme_loss += loss_dict['total'].item()
        n_batches += 1
        
        # Update standard performance meter (for mIoU and depth metrics)
        performance_meter.update(
            {t: get_output(output[t], t) for t in p.TASKS.NAMES},
            {t: targets[t] for t in p.TASKS.NAMES}
        )
        
        # Accumulate pixel-level stats for segmentation
        pred_seg = get_output(output['semseg'], 'semseg').squeeze()
        gt_seg = targets['semseg'].squeeze()
        valid = (gt_seg != p.ignore_index)
        
        for c in range(n_classes):
            c_gt = (gt_seg == c)
            c_pred = (pred_seg == c)
            total_tp[c] += torch.sum(c_gt & c_pred & valid).item()
            total_fp[c] += torch.sum(~c_gt & c_pred & valid).item()
            total_fn[c] += torch.sum(c_gt & ~c_pred & valid).item()
            
    # Calculate aggregated metrics
    val_linear_loss = total_linear_loss / max(n_batches, 1)
    val_scheme_loss = total_scheme_loss / max(n_batches, 1)
    
    # Segmentation Metrics: mIoU (macro), Overall IoU (micro), Pixel Accuracy
    class_ious = [float(total_tp[c]) / max(float(total_tp[c] + total_fp[c] + total_fn[c]), 1e-8) for c in range(n_classes)]
    m_iou = float(np.mean(class_ious))
    overall_iou = float(np.sum(total_tp)) / max(float(np.sum(total_tp + total_fp + total_fn)), 1e-8)
    pixel_acc = float(np.sum(total_tp)) / max(float(np.sum(total_tp + total_fn)), 1e-8)
    
    # Depth metrics from performance meter
    depth_res = performance_meter.get_score(verbose=False)['depth']
    
    summary = {
        "model": p.get("version_name", "mtmamba"),
        "linear_loss": round(val_linear_loss, 4),
        "scheme_loss": round(val_scheme_loss, 4),
        "mIoU": round(m_iou, 4),
        "overall_iou": round(overall_iou, 4),
        "pixel_accuracy": round(pixel_acc, 4),
        "abs_rel": round(depth_res['abs_rel'], 4),
        "rmse": round(depth_res['rmse'], 4),
        "log_rmse": round(depth_res['log_rmse'], 4),
        "sq_rel": round(depth_res['sq_rel'], 4),
    }
    
    print("\n" + "=" * 70)
    print(f"AUTHENTIC EVALUATION RESULTS: {p.get('version_name')}")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:20}: {v}")
    print("=" * 70)
    
    print("\nClass-wise IoUs:")
    for c, name in enumerate(class_names):
        print(f"  {name:15}: {class_ious[c] * 100:.2f}%")
        
    out_json = checkpoint_path.replace(".pth.tar", "_complete_metrics.json")
    try:
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"\n[Evaluate] Saved complete metrics to: {out_json}")
    except Exception as e:
        print(f"\n[Evaluate] Could not write json to checkpoint dir ({e}), writing to ./evaluation_results.json")
        with open("evaluation_results.json", "w") as f:
            json.dump(summary, f, indent=4)
            
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint .pth.tar (optional, auto-searches if omitted)")
    args = parser.parse_args()
    
    evaluate_checkpoint(args.config, args.checkpoint if args.checkpoint else None)
