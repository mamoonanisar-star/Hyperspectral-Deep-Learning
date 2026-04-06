"""
Project 2: Crop Stress / Disease Detection
===========================================
Detects drought stress, fungal disease, and nutrient deficiency in
hyperspectral crop imagery using:
  - Vegetation indices (NDVI, NDRE, MCARI)
  - A fine-tuned U-Net segmentation model
  - A per-patch "health score" (0–100 %)

The module can be used standalone or imported into other projects.

Usage
-----
    python crop_stress_detection.py

The default run uses synthetic imagery. To use real data (e.g. PlantVillage
hyperspectral scans), pass your (H, W, B) datacube to `run_pipeline()`.
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import jaccard_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Vegetation index computation
# ---------------------------------------------------------------------------

# Band-index mapping (0-based) for a Sentinel-2-style 13-band datacube
# matching nd_crop_classifier.py ordering:
# [B02, B03, B04, B05, B06, B07, B08, B08A, B09, B11, B12, NDVI_raw, NDRE_raw]
BAND_IDX = {
    "B02": 0,   # Blue     490 nm
    "B03": 1,   # Green    560 nm
    "B04": 2,   # Red      665 nm
    "B05": 3,   # RE1      705 nm
    "B07": 5,   # RE3      783 nm
    "B08": 6,   # NIR      842 nm
    "B11": 9,   # SWIR1   1610 nm
}


def compute_ndvi(data: np.ndarray) -> np.ndarray:
    """
    NDVI = (NIR – Red) / (NIR + Red)
    Higher is healthier vegetation.
    """
    nir = data[..., BAND_IDX["B08"]].astype(np.float32)
    red = data[..., BAND_IDX["B04"]].astype(np.float32)
    denom = nir + red + 1e-8
    return np.clip((nir - red) / denom, -1.0, 1.0)


def compute_ndre(data: np.ndarray) -> np.ndarray:
    """
    NDRE = (NIR – RedEdge1) / (NIR + RedEdge1)
    More sensitive to chlorophyll content than NDVI.
    """
    nir = data[..., BAND_IDX["B08"]].astype(np.float32)
    re1 = data[..., BAND_IDX["B05"]].astype(np.float32)
    denom = nir + re1 + 1e-8
    return np.clip((nir - re1) / denom, -1.0, 1.0)


def compute_mcari(data: np.ndarray) -> np.ndarray:
    """
    MCARI = [(RE1 – Red) – 0.2*(RE1 – Green)] * (RE1 / Red)
    Modified Chlorophyll Absorption Ratio Index.
    Sensitive to disease-induced chlorophyll loss.
    """
    re1 = data[..., BAND_IDX["B05"]].astype(np.float32)
    red = data[..., BAND_IDX["B04"]].astype(np.float32)
    grn = data[..., BAND_IDX["B03"]].astype(np.float32)
    ratio = re1 / (red + 1e-8)
    return ((re1 - red) - 0.2 * (re1 - grn)) * ratio


def append_vegetation_indices(data: np.ndarray) -> np.ndarray:
    """
    Concatenate NDVI, NDRE, MCARI as extra bands.

    Parameters
    ----------
    data : (H, W, B) float32 array

    Returns
    -------
    (H, W, B+3) float32 array
    """
    ndvi = compute_ndvi(data)[..., None]
    ndre = compute_ndre(data)[..., None]
    mcari = compute_mcari(data)[..., None]
    return np.concatenate([data, ndvi, ndre, mcari], axis=-1)


# ---------------------------------------------------------------------------
# Synthetic stress dataset generator
# ---------------------------------------------------------------------------

STRESS_LABELS = {0: "Healthy", 1: "Drought Stress", 2: "Disease / Fungal", 3: "Nutrient Deficiency"}
NUM_STRESS_CLASSES = len(STRESS_LABELS)


class CropStressDataGenerator:
    """
    Generates a (H, W, B) datacube with spatially structured stress regions.
    Designed to mimic a field where different blocks suffer different stress types.
    """

    def __init__(self, height: int = 160, width: int = 160, num_bands: int = 13, seed: int = 7):
        self.H, self.W, self.B = height, width, num_bands
        self.rng = np.random.default_rng(seed)

    # Spectral means per stress class (13 Sentinel-2-style bands)
    SPECTRAL = {
        0: [0.05, 0.09, 0.07, 0.06, 0.38, 0.44, 0.48, 0.46, 0.49, 0.50, 0.20, 0.13, 0.09],  # Healthy
        1: [0.06, 0.10, 0.10, 0.08, 0.25, 0.28, 0.30, 0.29, 0.31, 0.32, 0.29, 0.22, 0.18],  # Drought
        2: [0.07, 0.11, 0.12, 0.09, 0.20, 0.22, 0.24, 0.23, 0.25, 0.26, 0.26, 0.20, 0.17],  # Disease
        3: [0.05, 0.08, 0.08, 0.07, 0.30, 0.34, 0.36, 0.35, 0.37, 0.38, 0.24, 0.17, 0.13],  # Nutrient def
    }

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        data : (H, W, B) float32
        gt   : (H, W)    int32  (0=Healthy, 1=Drought, 2=Disease, 3=Nutrient)
        """
        data = np.zeros((self.H, self.W, self.B), dtype=np.float32)
        gt = np.zeros((self.H, self.W), dtype=np.int32)

        # Divide image into 2x2 quadrants, one class each
        half_h, half_w = self.H // 2, self.W // 2
        quadrants = [
            (slice(0, half_h), slice(0, half_w), 0),           # top-left  = Healthy
            (slice(0, half_h), slice(half_w, self.W), 1),      # top-right = Drought
            (slice(half_h, self.H), slice(0, half_w), 2),      # bot-left  = Disease
            (slice(half_h, self.H), slice(half_w, self.W), 3), # bot-right = Nutrient
        ]
        for rs, cs, cls in quadrants:
            h_slice = rs.stop - rs.start
            w_slice = cs.stop - cs.start
            mean_vec = np.array(self.SPECTRAL[cls], dtype=np.float32)
            noise = self.rng.normal(0, 0.025, (h_slice, w_slice, self.B)).astype(np.float32)
            tile = np.clip(mean_vec[None, None, :] + noise, 0.0, 1.0)
            data[rs, cs, :] = tile
            gt[rs, cs] = cls

        return data, gt


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StressPatchDataset(Dataset):
    """Yields (patch_tensor, label) pairs for segmentation / classification."""

    def __init__(
        self,
        data: np.ndarray,
        gt: np.ndarray,
        patch_size: int = 13,
        augment: bool = False,
    ):
        self.patch_size = patch_size
        self.augment = augment
        pad = patch_size // 2

        # Append vegetation indices as extra channels
        data_vi = append_vegetation_indices(data)          # (H, W, B+3)
        padded = np.pad(data_vi, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

        ys, xs = np.where(gt >= 0)    # all pixels
        patches, labels = [], []
        for y, x in zip(ys, xs):
            p = padded[y: y + patch_size, x: x + patch_size, :]   # (P, P, B+3)
            patches.append(p.transpose(2, 0, 1))                   # (B+3, P, P)
            labels.append(int(gt[y, x]))

        self.X = torch.tensor(np.array(patches, dtype=np.float32))
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        if self.augment and torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[2])   # horizontal flip
        if self.augment and torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[1])   # vertical flip
        return x, self.y[idx]


# ---------------------------------------------------------------------------
# U-Net for crop stress segmentation
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CropStressUNet(nn.Module):
    """
    Lightweight U-Net for patch-level stress segmentation.
    Input : (B, in_channels, patch_size, patch_size)
    Output: (B, num_classes, patch_size, patch_size)
    """

    def __init__(self, in_channels: int = 16, num_classes: int = 4, base: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool = nn.MaxPool2d(2, 2)
        # Bottleneck
        self.bottleneck = DoubleConv(base * 2, base * 4)
        # Decoder
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        # Head
        self.head = nn.Conv2d(base, num_classes, kernel_size=1)

    @staticmethod
    def _pad_to_match(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Interpolate `upsampled` to the spatial size of `skip` if they differ."""
        if upsampled.shape[-2:] != skip.shape[-2:]:
            upsampled = F.interpolate(
                upsampled, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return upsampled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        # Bottleneck
        b = self.bottleneck(self.pool(e2))
        # Decode – align spatial sizes before concatenation
        d2 = self.dec2(torch.cat([self._pad_to_match(self.up2(b), e2),  e2], dim=1))
        d1 = self.dec1(torch.cat([self._pad_to_match(self.up1(d2), e1), e1], dim=1))
        return self.head(d1)


# ---------------------------------------------------------------------------
# Health score computation
# ---------------------------------------------------------------------------

def compute_health_score(ndvi: np.ndarray, ndre: np.ndarray) -> np.ndarray:
    """
    Combines NDVI and NDRE into a single health score in [0, 100].
    Higher = healthier.

    score = 50 * (NDVI_norm + NDRE_norm)
    where NDVI_norm, NDRE_norm are rescaled from [-1, 1] to [0, 1].
    """
    ndvi_norm = (ndvi + 1.0) / 2.0
    ndre_norm = (ndre + 1.0) / 2.0
    score = 50.0 * (ndvi_norm + ndre_norm)
    return np.clip(score, 0.0, 100.0)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

@dataclass
class StressConfig:
    in_channels: int = 16        # 13 bands + 3 VI bands
    num_classes: int = 4
    patch_size: int = 13
    base_filters: int = 32
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_unet(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: StressConfig,
) -> Dict:
    device = torch.device(cfg.device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    # Weighted cross-entropy (disease / nutrient classes are less common)
    class_weights = torch.tensor([1.0, 1.5, 2.0, 1.5], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
    }

    for epoch in range(1, cfg.num_epochs + 1):
        # --- train ---
        model.train()
        t_loss, t_corr, t_tot = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            # Patch-level: model outputs (B, C, P, P), we use per-pixel labels
            out = model(xb)                         # (B, C, P, P)
            # Expand scalar label to spatial map
            B, C, H, W = out.shape
            yb_map = yb.view(B, 1, 1).expand(B, H, W)
            loss = criterion(out, yb_map)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * B
            preds = out.argmax(1)                   # (B, H, W)
            t_corr += (preds == yb_map).float().mean().item() * B
            t_tot += B
        scheduler.step()

        # --- val ---
        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                B, C, H, W = out.shape
                yb_map = yb.view(B, 1, 1).expand(B, H, W)
                v_loss += criterion(out, yb_map).item() * B
                preds = out.argmax(1)
                v_corr += (preds == yb_map).float().mean().item() * B
                v_tot += B

        history["train_loss"].append(t_loss / max(t_tot, 1))
        history["val_loss"].append(v_loss / max(v_tot, 1))
        history["train_acc"].append(t_corr / max(t_tot, 1))
        history["val_acc"].append(v_corr / max(v_tot, 1))

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [U-Net] Epoch {epoch:3d}/{cfg.num_epochs}"
                  f"  train_loss={t_loss/max(t_tot,1):.4f}"
                  f"  val_acc={v_corr/max(v_tot,1):.4f}")

    return history


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise_vi_maps(data: np.ndarray, gt: np.ndarray):
    ndvi = compute_ndvi(data)
    ndre = compute_ndre(data)
    mcari = compute_mcari(data)
    health = compute_health_score(ndvi, ndre)

    # False-colour composite (NIR, Red, Green)
    r = data[..., BAND_IDX["B08"]]
    g = data[..., BAND_IDX["B04"]]
    b = data[..., BAND_IDX["B03"]]
    rgb = np.stack([r, g, b], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    cmap_stress = mcolors.LinearSegmentedColormap.from_list(
        "stress", ["red", "orange", "yellow", "limegreen", "darkgreen"], N=256
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Crop Stress / Vegetation Index Maps", fontsize=14)

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("False-colour (NIR-Red-Green)")

    axes[0, 1].imshow(ndvi, cmap=cmap_stress, vmin=-0.2, vmax=0.8)
    axes[0, 1].set_title("NDVI")
    plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1])

    axes[0, 2].imshow(ndre, cmap=cmap_stress, vmin=-0.2, vmax=0.7)
    axes[0, 2].set_title("NDRE")
    plt.colorbar(axes[0, 2].images[0], ax=axes[0, 2])

    axes[1, 0].imshow(mcari, cmap="PiYG", vmin=-0.5, vmax=0.5)
    axes[1, 0].set_title("MCARI")
    plt.colorbar(axes[1, 0].images[0], ax=axes[1, 0])

    axes[1, 1].imshow(health, cmap=cmap_stress, vmin=0, vmax=100)
    axes[1, 1].set_title("Health Score (0–100%)")
    plt.colorbar(axes[1, 1].images[0], ax=axes[1, 1])

    label_colors = ["limegreen", "orange", "firebrick", "royalblue"]
    from matplotlib.colors import ListedColormap
    gt_cmap = ListedColormap(label_colors)
    im = axes[1, 2].imshow(gt, cmap=gt_cmap, vmin=-0.5, vmax=3.5)
    axes[1, 2].set_title("Ground Truth (stress labels)")
    cbar = plt.colorbar(im, ax=axes[1, 2], ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(list(STRESS_LABELS.values()))

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs("outputs/crop_stress", exist_ok=True)
    plt.savefig("outputs/crop_stress/vi_maps.png", dpi=150)
    plt.show()
    print("Saved → outputs/crop_stress/vi_maps.png")

    return health


def visualise_health_score(health: np.ndarray, gt: np.ndarray):
    """Per-class health score distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    label_colors = ["limegreen", "orange", "firebrick", "royalblue"]
    for cls_idx, (label, color) in enumerate(zip(STRESS_LABELS.values(), label_colors)):
        mask = gt == cls_idx
        if not mask.any():
            continue
        scores = health[mask]
        ax.hist(scores, bins=30, alpha=0.6, color=color, label=label, edgecolor="white")
    ax.set_xlabel("Health Score (%)")
    ax.set_ylabel("Pixel Count")
    ax.set_title("Per-class Health Score Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/crop_stress/health_score_dist.png", dpi=150)
    plt.show()
    print("Saved → outputs/crop_stress/health_score_dist.png")


def visualise_segmentation(
    model: nn.Module,
    data: np.ndarray,
    gt: np.ndarray,
    cfg: StressConfig,
):
    """Run the U-Net on the full image and display predicted stress map."""
    device = torch.device(cfg.device)
    model.eval()
    model = model.to(device)

    data_vi = append_vegetation_indices(data)          # (H, W, B+3)
    H, W, B = data_vi.shape
    patch = cfg.patch_size

    # Pad image
    pad = patch // 2
    padded = np.pad(data_vi, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

    pred_map = np.zeros((H, W), dtype=np.int32)

    # Slide over every pixel
    batch_x, batch_yx = [], []
    batch_size = 512
    count = 0

    with torch.no_grad():
        for y in range(H):
            for x in range(W):
                p = padded[y: y + patch, x: x + patch, :].transpose(2, 0, 1)
                batch_x.append(p)
                batch_yx.append((y, x))
                count += 1
                if count == batch_size:
                    t = torch.tensor(np.array(batch_x, dtype=np.float32)).to(device)
                    out = model(t)           # (B, C, P, P)
                    centre = out[:, :, patch // 2, patch // 2]   # (B, C)
                    preds = centre.argmax(1).cpu().numpy()
                    for (py, px), pred in zip(batch_yx, preds):
                        pred_map[py, px] = pred
                    batch_x, batch_yx = [], []
                    count = 0
        if batch_x:
            t = torch.tensor(np.array(batch_x, dtype=np.float32)).to(device)
            out = model(t)
            centre = out[:, :, patch // 2, patch // 2]
            preds = centre.argmax(1).cpu().numpy()
            for (py, px), pred in zip(batch_yx, preds):
                pred_map[py, px] = pred

    from matplotlib.colors import ListedColormap
    label_colors = ["limegreen", "orange", "firebrick", "royalblue"]
    gt_cmap = ListedColormap(label_colors)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(gt, cmap=gt_cmap, vmin=-0.5, vmax=3.5)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")
    im = axes[1].imshow(pred_map, cmap=gt_cmap, vmin=-0.5, vmax=3.5)
    axes[1].set_title("U-Net Prediction")
    axes[1].axis("off")
    cbar = plt.colorbar(im, ax=axes[1], ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
    cbar.set_ticklabels(list(STRESS_LABELS.values()))
    plt.suptitle("Crop Stress Segmentation", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/crop_stress/segmentation_map.png", dpi=150)
    plt.show()
    print("Saved → outputs/crop_stress/segmentation_map.png")

    oa = accuracy_score(gt.flatten(), pred_map.flatten())
    print(f"\n  Pixel-level OA: {oa*100:.2f}%")
    return pred_map


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    data: Optional[np.ndarray] = None,
    gt: Optional[np.ndarray] = None,
    cfg: Optional[StressConfig] = None,
):
    if cfg is None:
        cfg = StressConfig()

    if data is None or gt is None:
        print("\n[INFO] Generating synthetic crop stress data…")
        gen = CropStressDataGenerator()
        data, gt = gen.generate()
        print(f"  Data: {data.shape}, GT unique: {np.unique(gt)}")

    os.makedirs("outputs/crop_stress", exist_ok=True)

    # --- Vegetation index maps ---
    print("\n[INFO] Computing vegetation indices & health score…")
    health = visualise_vi_maps(data, gt)
    visualise_health_score(health, gt)

    # --- Dataset ---
    print("\n[INFO] Building patch dataset…")
    full_ds = StressPatchDataset(data, gt, patch_size=cfg.patch_size, augment=False)
    aug_ds = StressPatchDataset(data, gt, patch_size=cfg.patch_size, augment=True)

    n = len(full_ds)
    idx = np.arange(n)
    i_tv, i_test = train_test_split(idx, test_size=0.10, stratify=gt.flatten(), random_state=42)
    i_train, i_val = train_test_split(i_tv, test_size=0.111, stratify=gt.flatten()[i_tv], random_state=42)

    def subset_loader(dataset, indices, shuffle):
        from torch.utils.data import Subset
        return DataLoader(Subset(dataset, indices), batch_size=cfg.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = subset_loader(aug_ds, i_train, shuffle=True)
    val_loader   = subset_loader(full_ds, i_val,   shuffle=False)
    test_loader  = subset_loader(full_ds, i_test,  shuffle=False)

    print(f"  Train={len(i_train)}  Val={len(i_val)}  Test={len(i_test)}")

    # --- Train U-Net ---
    print("\n[INFO] Training U-Net for crop stress segmentation…")
    model = CropStressUNet(
        in_channels=data.shape[-1] + 3,    # bands + 3 VI bands
        num_classes=cfg.num_classes,
        base=cfg.base_filters,
    )
    history = train_unet(model, train_loader, val_loader, cfg)

    # --- Inference on full image ---
    print("\n[INFO] Running full-image segmentation…")
    visualise_segmentation(model, data, gt, cfg)

    # --- Save model ---
    os.makedirs("checkpoints/crop_stress", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/crop_stress/unet.pth")

    # --- Training curves ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.suptitle("U-Net Training Curves – Crop Stress")
    plt.tight_layout()
    plt.savefig("outputs/crop_stress/training_curves.png", dpi=150)
    plt.show()
    print("Saved → outputs/crop_stress/training_curves.png")

    print("\n[INFO] Project 2 complete. Outputs saved to outputs/crop_stress/")
    return model, history


if __name__ == "__main__":
    run_pipeline()
