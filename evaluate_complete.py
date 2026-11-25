"""
Standalone Evaluation Script for MTMamba (Swin-Large & ResNet-50)
Supports both CityscapesMTAN (7 classes) and NYUv2MTAN (13 classes).

Computes the full set of authentic metrics:
  - Loss (Linear Multi-Task Loss = sum(L_task), identical to benchmarks)
  - mIoU (Mean Intersection over Union / Macro-average)
  - Overall IoU (Global Intersection over Union / Micro-average)
  - Pixel Accuracy (Global Correct Pixel Percentage)
  - Abs Rel (%) (Absolute Relative Depth Error)
  - RMSE (Root Mean Squared Depth Error)
  - Log RMSE & Sq Rel
  - Surface Normal Errors (Mean / Median degree error, % < 11.25°, 22.5°, 30°)
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
    """Locate dataset directory containing val/image/."""
    db_name = p.get('val_db_name', 'CityscapesMTAN')
    candidates = [
        p.get('db_paths', {}).get(db_name, ''),
        f'/kaggle/working/data/{db_name.lower().replace("mtan", "")}',
        '/kaggle/working/data/nyuv2',
        '/data',
        '/kaggle/input/nyuv2',
        '/kaggle/input/nyu-depth-v2',
        '/kaggle/input/cityscapes-mtan-dataset',
        '/kaggle/input/datasets/prachetdevsingh/cityscapes-mtan-dataset',
    ]
    for c in candidates:
        if c and os.path.exists(os.path.join(c, 'val', 'image')):
            p.db_paths[db_name] = c
            print(f"[Evaluate] Detected {db_name} dataset at: {c}")
            return c

    # Recursive search under /kaggle/input and /kaggle/working
    for sroot in ['/kaggle/working', '/kaggle/input']:
        if os.path.exists(sroot):
            for root, dirs, files in os.walk(sroot):
                if 'image' in dirs and os.path.basename(root) == 'val':
                    dataset_dir = os.path.dirname(root)
                    p.db_paths[db_name] = dataset_dir
                    print(f"[Evaluate] Auto-discovered {db_name} dataset at: {dataset_dir}")
                    return dataset_dir

    print(f"[Evaluate] Warning: Using default dataset path '{p.db_paths.get(db_name)}'")
    return p.db_paths.get(db_name)


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

    # Prioritize matching backbone name and dataset name
    is_resnet = 'resnet' in config_path.lower()
    is_nyu = 'nyu' in config_path.lower()

    matching = [x for x in found if ('resnet' in x.lower()) == is_resnet and ('nyu' in x.lower()) == is_nyu]
    if not matching:
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
    
    # Semantic class names
    if p.get('val_db_name') == 'NYUv2MTAN':
        class_names = [
            "bed", "books", "ceiling", "chair", "floor", "furniture",
            "objects", "painting", "sofa", "table", "tv", "wall", "window"
        ]
    else:
        class_names = ["flat", "construction", "object", "nature", "sky", "human", "vehicle"]

    n_classes = len(class_names)
    total_tp = np.zeros(n_classes, dtype=np.int64)
    total_fp = np.zeros(n_classes, dtype=np.int64)
    total_fn = np.zeros(n_classes, dtype=np.int64)
    
    print(f"[Evaluate] Running validation pass across all {len(test_dataset)} images...")
    for batch in tqdm(test_loader):
        images = batch['image'].to(device, non_blocking=True)
        targets = {task: batch[task].to(device, non_blocking=True) for task in p.TASKS.NAMES}
        
        output = model(images)
        loss_dict = criterion(output, targets, tasks=p.TASKS.NAMES)
        
        # Linear multi-task loss (sum of individual task losses)
        batch_linear = sum(loss_dict[t].item() for t in p.TASKS.NAMES if t in loss_dict)
        total_linear_loss += batch_linear
        
        if 'total' in loss_dict:
            total_scheme_loss += loss_dict['total'].item()
        n_batches += 1
        
        # Update standard performance meter
        performance_meter.update(
            {t: get_output(output[t], t) for t in p.TASKS.NAMES},
            {t: targets[t] for t in p.TASKS.NAMES}
        )
        
        # Accumulate pixel-level stats for segmentation if present
        if 'semseg' in p.TASKS.NAMES:
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
    
    summary = {
        "model": p.get("version_name", "mtmamba"),
        "linear_loss": round(val_linear_loss, 4),
        "scheme_loss": round(val_scheme_loss, 4),
    }

    # Segmentation Metrics
    if 'semseg' in p.TASKS.NAMES:
        class_ious = [float(total_tp[c]) / max(float(total_tp[c] + total_fp[c] + total_fn[c]), 1e-8) for c in range(n_classes)]
        m_iou = float(np.mean(class_ious))
        overall_iou = float(np.sum(total_tp)) / max(float(np.sum(total_tp + total_fp + total_fn)), 1e-8)
        pixel_acc = float(np.sum(total_tp)) / max(float(np.sum(total_tp + total_fn)), 1e-8)
        summary["mIoU"] = round(m_iou, 4)
        summary["overall_iou"] = round(overall_iou, 4)
        summary["pixel_accuracy"] = round(pixel_acc, 4)
    
    # Depth metrics
    perf_scores = performance_meter.get_score(verbose=False)
    if 'depth' in p.TASKS.NAMES and 'depth' in perf_scores:
        depth_res = perf_scores['depth']
        summary["abs_rel"] = round(depth_res.get('abs_rel', 0.0), 4)
        summary["rmse"] = round(depth_res.get('rmse', 0.0), 4)
        summary["log_rmse"] = round(depth_res.get('log_rmse', 0.0), 4)
        summary["sq_rel"] = round(depth_res.get('sq_rel', 0.0), 4)

    # Normals metrics
    if 'normals' in p.TASKS.NAMES and 'normals' in perf_scores:
        norm_res = perf_scores['normals']
        summary["normals_mean_deg"] = round(norm_res.get('mean', 0.0), 4)
        summary["normals_median_deg"] = round(norm_res.get('median', 0.0), 4)
        summary["normals_11deg"] = round(norm_res.get('deg_11.25', 0.0), 4)
        summary["normals_22deg"] = round(norm_res.get('deg_22.5', 0.0), 4)
        summary["normals_30deg"] = round(norm_res.get('deg_30', 0.0), 4)
    
    print("\n" + "=" * 70)
    print(f"AUTHENTIC EVALUATION RESULTS: {p.get('version_name')}")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:20}: {v}")
    print("=" * 70)
    
    if 'semseg' in p.TASKS.NAMES:
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
