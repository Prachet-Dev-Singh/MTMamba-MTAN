"""
Standalone Evaluation Script for MTMamba (Swin-Large & ResNet-50)

Computes the full set of metrics matching the CNN U-Net and Mamba-UNet tables:
  - Loss (Validation Multi-Task Loss)
  - mIoU (Mean Intersection over Union / Macro-average)
  - Overall IoU (Global Intersection over Union / Micro-average)
  - Pixel Accuracy (Global Correct Pixel Percentage)
  - Abs Rel (%) (Absolute Relative Depth Error)
  - RMSE (Root Mean Squared Depth Error)
  - Log RMSE & Sq Rel

Usage on Kaggle:
    python evaluate_complete.py --config configs/mtmamba_cityscapes_mtan.yml --checkpoint /kaggle/working/runs/mtmamba_city_mtan/best_model.pth.tar
    python evaluate_complete.py --config configs/mtmamba_cityscapes_mtan_resnet.yml --checkpoint /kaggle/working/runs/mtmamba_city_mtan_resnet50/best_model.pth.tar
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.config import create_config
from utils.common_config import get_model, get_test_dataset, get_test_dataloader, get_criterion
from evaluation.evaluate_utils import PerformanceMeter
from utils.utils import get_output


@torch.no_grad()
def evaluate_checkpoint(config_path, checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    p = create_config(config_path)
    p['db_paths']['CityscapesMTAN'] = p.db_paths['CityscapesMTAN']
    
    print(f"\n[Evaluate] Loading config from: {config_path}")
    print(f"[Evaluate] Loading checkpoint from: {checkpoint_path}")
    
    # 1. Dataset & Dataloader
    test_dataset = get_test_dataset(p)
    test_loader = get_test_dataloader(p, test_dataset)
    print(f"[Evaluate] Validation images: {len(test_dataset)}")
    
    # 2. Model & Checkpoint
    model = get_model(p)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        # Strip potential DDP 'module.' prefix
        clean_state_dict = {}
        for k, v in state_dict.items():
            clean_k = k.replace("module.", "")
            clean_state_dict[clean_k] = v
        model.load_state_dict(clean_state_dict)
        print("[Evaluate] Checkpoint loaded successfully!")
    else:
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    
    # 3. Loss criterion & Meters
    criterion = get_criterion(p).to(device)
    performance_meter = PerformanceMeter(p, p.TASKS.NAMES)
    
    total_loss = 0.0
    n_batches = 0
    
    # Semantic segmentation accumulation for Overall IoU & Pixel Accuracy
    n_classes = 7
    total_tp = np.zeros(n_classes, dtype=np.int64)
    total_fp = np.zeros(n_classes, dtype=np.int64)
    total_fn = np.zeros(n_classes, dtype=np.int64)
    
    print("[Evaluate] Running validation pass across all 500 images...")
    for batch in tqdm(test_loader):
        images = batch['image'].to(device, non_blocking=True)
        targets = {task: batch[task].to(device, non_blocking=True) for task in p.TASKS.NAMES}
        
        output = model(images)
        loss, _ = criterion(output, targets)
        total_loss += loss.item()
        n_batches += 1
        
        # Update standard performance meter
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
    val_loss = total_loss / max(n_batches, 1)
    
    # mIoU (macro), Overall IoU (micro), Pixel Accuracy
    class_ious = [float(total_tp[c]) / max(float(total_tp[c] + total_fp[c] + total_fn[c]), 1e-8) for c in range(n_classes)]
    m_iou = float(np.mean(class_ious))
    overall_iou = float(np.sum(total_tp)) / max(float(np.sum(total_tp + total_fp + total_fn)), 1e-8)
    pixel_acc = float(np.sum(total_tp)) / max(float(np.sum(total_tp + total_fn)), 1e-8)
    
    # Depth metrics from performance meter
    depth_res = performance_meter.get_score(verbose=False)['depth']
    
    summary = {
        "model": p.get("version_name", "mtmamba"),
        "loss": round(val_loss, 4),
        "mIoU": round(m_iou, 4),
        "overall_iou": round(overall_iou, 4),
        "pixel_accuracy": round(pixel_acc, 4),
        "abs_rel": round(depth_res['abs_rel'], 4),
        "rmse": round(depth_res['rmse'], 4),
        "log_rmse": round(depth_res['log_rmse'], 4),
        "sq_rel": round(depth_res['sq_rel'], 4),
    }
    
    print("\n" + "=" * 60)
    print(f"COMPLETE EVALUATION RESULTS: {p.get('version_name')}")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:20}: {v}")
    print("=" * 60)
    
    out_json = checkpoint_path.replace(".pth.tar", "_complete_metrics.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"[Evaluate] Saved complete metrics to: {out_json}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth.tar")
    args = parser.parse_args()
    
    evaluate_checkpoint(args.config, args.checkpoint)
