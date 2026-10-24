"""
CityscapesMTAN dataset adapter for MTMamba.
"""
import os, glob, numpy as np, torch, torch.utils.data as data
import torchvision.transforms as t_transforms

MTAN_CLASS_NAMES = ["flat", "construction", "object", "nature", "sky", "human", "vehicle"]
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

class CityscapesMTAN(data.Dataset):
    N_CLASSES = 7
    DEPTH_INVALID_IN  = 0
    DEPTH_INVALID_OUT = 255.0  # Remap to 255.0 so L1Loss(ignore_index=255) ignores it

    def __init__(self, root, split="train", task_list=None, ignore_index=255, augment=False):
        super().__init__()
        self.root = root
        self.split = split
        self.task_list = task_list or ["semseg", "depth"]
        self.ignore_index = ignore_index
        self.augment = augment and (split == "train")
        sd = os.path.join(root, split)
        self.image_files = sorted(glob.glob(os.path.join(sd, "image", "*.npy")))
        self.label_files = sorted(glob.glob(os.path.join(sd, "label_7", "*.npy")))
        self.depth_files = sorted(glob.glob(os.path.join(sd, "depth", "*.npy")))
        if not self.image_files:
            raise FileNotFoundError(f"No image .npy files in {sd}/image/")
        assert len(self.image_files) == len(self.label_files) == len(self.depth_files), \
            f"File count mismatch: {len(self.image_files)} {len(self.label_files)} {len(self.depth_files)}"
        for a, b, c in zip(self.image_files, self.label_files, self.depth_files):
            sa, sb, sc = [os.path.splitext(os.path.basename(x))[0] for x in (a, b, c)]
            assert sa == sb == sc, f"Filename mismatch: {sa} vs {sb} vs {sc}"
            
        self.img_transform = t_transforms.Compose([
            t_transforms.ToTensor(),
            t_transforms.Normalize(mean=_MEAN, std=_STD)
        ])
        print(f"[CityscapesMTAN] split={split}, n={len(self.image_files)}, tasks={self.task_list}, augment={self.augment}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = self.image_files[index]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img = np.load(img_path)
        
        # CRITICAL FIX 1: Proper float [0, 1] -> uint8 [0, 255] conversion
        # MTAN dumps store images as float in [0.0, 1.0].
        # If cast directly without multiplying by 255, 99% of pixels become 0 (black).
        if img.dtype != np.uint8:
            if img.max() <= 1.5:
                img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
                
        H, W = img.shape[:2]
        lbl = dep = None
        
        if "semseg" in self.task_list:
            lbl = np.load(self.label_files[index]).astype(np.int64)
            # MTAN uses -1 for unlabeled pixels; map to ignore_index (255)
            lbl[lbl == -1] = self.ignore_index
            
        if "depth" in self.task_list:
            dep = np.load(self.depth_files[index]).astype(np.float32)
            if dep.ndim == 3:
                dep = dep.squeeze(0) if dep.shape[0] == 1 else dep.squeeze(-1)
            # CRITICAL FIX 2: Map invalid depth (<= 0) to ignore_index (255.0) so L1Loss ignores them
            dep[dep <= self.DEPTH_INVALID_IN] = self.DEPTH_INVALID_OUT
            
        if self.augment and np.random.rand() < 0.5:
            img = np.fliplr(img).copy()
            if lbl is not None:
                lbl = np.fliplr(lbl).copy()
            if dep is not None:
                dep = np.fliplr(dep).copy()
                
        sample = {"image": self.img_transform(img)}
        if "semseg" in self.task_list:
            sample["semseg"] = torch.from_numpy(lbl).long()
        if "depth" in self.task_list:
            sample["depth"] = torch.from_numpy(dep).float().unsqueeze(0)
        sample["meta"] = {"img_name": stem, "img_size": (H, W), "scale_factor": np.array([1.0, 1.0])}
        return sample
