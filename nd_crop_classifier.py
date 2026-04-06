"""
Project 1: ND Crop Type Classifier
====================================
Classifies North Dakota crops (soybean, wheat, barley, corn, sunflower)
using a synthetic multispectral dataset that mimics Sentinel-2 spectral
characteristics. Trains ResNet18, ViT, and 3D-CNN models from the existing
pipeline and visualises spectral signatures + confusion matrices.

Usage
-----
    python nd_crop_classifier.py

To use real Sentinel-2 / USDA CroplandCROS data, replace
`NDCropDataGenerator.generate()` with your own loader and pass the
(H, W, bands) array and (H, W) ground-truth array to `run_pipeline()`.
"""

import os
import sys
import math
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    confusion_matrix, accuracy_score, cohen_kappa_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# ND Crop dataset definition
# ---------------------------------------------------------------------------

ND_CROP_INFO = {
    "num_classes": 5,
    # 13 Sentinel-2-style bands (B02..B8A, B11, B12) + derived indices
    "num_bands": 13,
    "spatial_size": (200, 200),
    "wavelength_range": "0.49-2.19 µm",
    "spatial_resolution": "10-20 m (Sentinel-2)",
    "class_names": [
        "Soybean",    # 0 – high NIR reflectance early season
        "Wheat",      # 1 – yellow / golden signature at harvest
        "Barley",     # 2 – similar to wheat, earlier peak
        "Corn",       # 3 – very high NIR mid-season
        "Sunflower",  # 4 – unique SWIR signature
    ],
    # Approximate mean reflectance per class across the 13 S2 bands
    # (values in [0, 1] normalised range)
    "spectral_means": {
        "Soybean":   [0.05, 0.07, 0.06, 0.04, 0.35, 0.42, 0.45, 0.43,
                      0.47, 0.48, 0.22, 0.14, 0.10],
        "Wheat":     [0.08, 0.11, 0.10, 0.08, 0.28, 0.30, 0.32, 0.31,
                      0.33, 0.34, 0.28, 0.20, 0.16],
        "Barley":    [0.07, 0.10, 0.09, 0.07, 0.30, 0.33, 0.35, 0.34,
                      0.36, 0.37, 0.26, 0.18, 0.14],
        "Corn":      [0.04, 0.06, 0.05, 0.03, 0.40, 0.48, 0.52, 0.50,
                      0.54, 0.55, 0.20, 0.12, 0.08],
        "Sunflower": [0.06, 0.09, 0.08, 0.06, 0.32, 0.36, 0.38, 0.37,
                      0.39, 0.40, 0.30, 0.24, 0.20],
    },
    "spectral_stds": 0.03,  # uniform noise std
}

BAND_NAMES = [
    "B02 (Blue)", "B03 (Green)", "B04 (Red)", "B05 (RedEdge1)",
    "B06 (RedEdge2)", "B07 (RedEdge3)", "B08 (NIR)", "B08A (NIR-narrow)",
    "B09 (WV)", "B11 (SWIR1)", "B12 (SWIR2)", "NDVI", "NDRE",
]


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

class NDCropDataGenerator:
    """
    Generates a synthetic hyperspectral/multispectral datacube that mimics
    the spectral characteristics of five ND crop types.

    To replace with real Sentinel-2 imagery, load your GeoTIFF / .mat file
    and return arrays of the same shapes from `generate()`.
    """

    def __init__(
        self,
        height: int = 200,
        width: int = 200,
        num_bands: int = 13,
        seed: int = 42,
    ):
        self.height = height
        self.width = width
        self.num_bands = num_bands
        self.seed = seed
        self.class_names = ND_CROP_INFO["class_names"]
        self.num_classes = len(self.class_names)

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        data : (H, W, B) float32 array  – normalised reflectance
        gt   : (H, W) int32 array       – class labels 1-based (0 = background)
        """
        rng = np.random.default_rng(self.seed)

        data = np.zeros((self.height, self.width, self.num_bands), dtype=np.float32)
        gt = np.zeros((self.height, self.width), dtype=np.int32)

        patch_h = self.height // self.num_classes
        means = list(ND_CROP_INFO["spectral_means"].values())
        std = ND_CROP_INFO["spectral_stds"]

        for cls_idx, cls_name in enumerate(self.class_names):
            row_start = cls_idx * patch_h
            row_end = (cls_idx + 1) * patch_h if cls_idx < self.num_classes - 1 else self.height

            mean_vec = np.array(means[cls_idx], dtype=np.float32)
            noise = rng.normal(0, std, (row_end - row_start, self.width, self.num_bands)).astype(np.float32)
            tile = mean_vec[None, None, :] + noise
            tile = np.clip(tile, 0.0, 1.0)

            data[row_start:row_end, :, :] = tile
            gt[row_start:row_end, :] = cls_idx + 1  # 1-based labels

        return data, gt

    def print_info(self):
        print("\n" + "=" * 60)
        print("ND Crop Dataset (Synthetic Sentinel-2 style)")
        print("=" * 60)
        print(f"Spatial size  : {self.height} x {self.width}")
        print(f"Spectral bands: {self.num_bands}  ({', '.join(BAND_NAMES)})")
        print(f"Classes       : {self.num_classes}")
        for i, name in enumerate(self.class_names):
            print(f"  {i+1}: {name}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Model configs for ND crops
# ---------------------------------------------------------------------------

@dataclass
class NDModelConfig:
    in_channels: int = 13
    num_classes: int = 5
    spatial_size: int = 9   # patch size used for training
    base_channels: int = 32
    dropout_rate: float = 0.4
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 40
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Lightweight CNN suitable for small patch size (9x9) and 13-band input
# ---------------------------------------------------------------------------

class SpectralAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        reduced = max(channels // 8, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        self.att = SpectralAttention(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.att(self.conv(x))


class NDCropCNN(nn.Module):
    """ResNet-style CNN for ND crop classification (classification head)."""

    def __init__(self, cfg: NDModelConfig):
        super().__init__()
        bc = cfg.base_channels
        self.stem = ConvBlock(cfg.in_channels, bc)
        self.layer1 = ConvBlock(bc, bc * 2)
        self.layer2 = ConvBlock(bc * 2, bc * 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(bc * 4, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        return self.head(x)


class NDCropViT(nn.Module):
    """Minimal ViT for ND crop patch classification."""

    def __init__(self, cfg: NDModelConfig):
        super().__init__()
        dim = 128
        self.proj = nn.Conv2d(cfg.in_channels, dim, kernel_size=1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.spatial_size ** 2 + 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=256,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.proj(x)                          # (B, dim, H, W)
        x = x.flatten(2).transpose(1, 2)          # (B, H*W, dim)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.norm(self.transformer(x))
        return self.head(x[:, 0])


class NDCrop3DCNN(nn.Module):
    """3D-CNN treating bands as the depth dimension."""

    def __init__(self, cfg: NDModelConfig):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=(3, 1, 1), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(16, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → unsqueeze to (B, 1, C, H, W)
        x = x.unsqueeze(1)
        x = self.conv3d(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_patch_dataset(
    data: np.ndarray,
    gt: np.ndarray,
    patch_size: int = 9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract labelled patches from the datacube."""
    pad = patch_size // 2
    padded = np.pad(data, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    ys, xs = np.where(gt > 0)
    patches, labels = [], []
    for y, x in zip(ys, xs):
        patch = padded[y: y + patch_size, x: x + patch_size, :]
        patches.append(patch.transpose(2, 0, 1))       # (C, H, W)
        labels.append(int(gt[y, x]) - 1)               # 0-based
    X = torch.tensor(np.array(patches, dtype=np.float32))
    y = torch.tensor(labels, dtype=torch.long)
    return X, y


def split_and_load(
    X: torch.Tensor,
    y: torch.Tensor,
    cfg: NDModelConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    idx = np.arange(len(y))
    i_tv, i_test = train_test_split(idx, test_size=0.10, stratify=y.numpy(), random_state=42)
    i_train, i_val = train_test_split(i_tv, test_size=0.111, stratify=y[i_tv].numpy(), random_state=42)

    def loader(indices, shuffle):
        ds = TensorDataset(X[indices], y[indices])
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=0)

    return loader(i_train, True), loader(i_val, False), loader(i_test, False)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: NDModelConfig,
    model_name: str = "model",
) -> Dict:
    device = torch.device(cfg.device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, cfg.num_epochs + 1):
        # --- train ---
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * xb.size(0)
            t_correct += (out.argmax(1) == yb).sum().item()
            t_total += xb.size(0)
        scheduler.step()

        # --- val ---
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                v_loss += criterion(out, yb).item() * xb.size(0)
                v_correct += (out.argmax(1) == yb).sum().item()
                v_total += xb.size(0)

        t_acc = t_correct / t_total
        v_acc = v_correct / v_total
        history["train_loss"].append(t_loss / t_total)
        history["val_loss"].append(v_loss / v_total)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{model_name}] Epoch {epoch:3d}/{cfg.num_epochs}"
                  f"  train_loss={t_loss/t_total:.4f}  val_acc={v_acc:.4f}")

    return history


# ---------------------------------------------------------------------------
# Evaluation & visualisation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    model_name: str = "model",
) -> Dict:
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds = model(xb).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(yb.numpy())

    oa = accuracy_score(all_targets, all_preds)
    kappa = cohen_kappa_score(all_targets, all_preds)
    per_class = [
        accuracy_score(
            [t for t, p in zip(all_targets, all_preds) if t == i],
            [p for t, p in zip(all_targets, all_preds) if t == i],
        )
        for i in range(len(class_names))
        if sum(1 for t in all_targets if t == i) > 0
    ]
    aa = float(np.mean(per_class))

    print(f"\n{'='*50}")
    print(f"  {model_name}  –  Test-set Results")
    print(f"{'='*50}")
    print(f"  OA (Overall Accuracy) : {oa*100:.2f}%")
    print(f"  AA (Average Accuracy) : {aa*100:.2f}%")
    print(f"  Kappa Coefficient     : {kappa:.4f}")
    print(f"\n{classification_report(all_targets, all_preds, target_names=class_names)}")

    return {
        "model": model_name,
        "OA": oa,
        "AA": aa,
        "Kappa": kappa,
        "preds": all_preds,
        "targets": all_targets,
    }


def plot_confusion_matrix(results: Dict, class_names: List[str]):
    cm = confusion_matrix(results["targets"], results["preds"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm * 100, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        vmin=0, vmax=100, ax=ax,
    )
    ax.set_title(f"Confusion Matrix (%) – {results['model']}")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    os.makedirs("outputs/nd_crop", exist_ok=True)
    plt.savefig(f"outputs/nd_crop/confusion_{results['model']}.png", dpi=150)
    plt.show()
    print(f"Saved → outputs/nd_crop/confusion_{results['model']}.png")


def plot_spectral_signatures(data: np.ndarray, gt: np.ndarray, class_names: List[str]):
    plt.figure(figsize=(10, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(class_names)))
    bands = np.arange(data.shape[2])

    for cls_idx, (name, color) in enumerate(zip(class_names, colors)):
        mask = gt == (cls_idx + 1)
        if not mask.any():
            continue
        pixels = data[mask]
        mean = pixels.mean(axis=0)
        std = pixels.std(axis=0)
        plt.plot(bands, mean, label=name, color=color, linewidth=2)
        plt.fill_between(bands, mean - std, mean + std, alpha=0.15, color=color)

    plt.xlabel("Band Index")
    plt.ylabel("Normalised Reflectance")
    plt.title("ND Crop Spectral Signatures (Sentinel-2 style)")
    tick_step = max(1, len(BAND_NAMES) // 8)
    plt.xticks(bands[::tick_step], BAND_NAMES[::tick_step], rotation=30, ha="right", fontsize=8)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs("outputs/nd_crop", exist_ok=True)
    plt.savefig("outputs/nd_crop/spectral_signatures.png", dpi=150)
    plt.show()
    print("Saved → outputs/nd_crop/spectral_signatures.png")


def plot_training_curves(histories: Dict[str, Dict], metric: str = "val_acc"):
    plt.figure(figsize=(8, 4))
    for name, h in histories.items():
        plt.plot(h[metric], label=name)
    plt.xlabel("Epoch")
    ylabel = "Validation Accuracy" if "acc" in metric else "Validation Loss"
    plt.ylabel(ylabel)
    plt.title(f"ND Crop Classifier – {ylabel}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs("outputs/nd_crop", exist_ok=True)
    plt.savefig(f"outputs/nd_crop/training_{metric}.png", dpi=150)
    plt.show()
    print(f"Saved → outputs/nd_crop/training_{metric}.png")


def plot_benchmark_table(all_results: List[Dict]):
    print("\n" + "=" * 55)
    print(f"{'Model':<15} {'OA (%)':>10} {'AA (%)':>10} {'Kappa':>12}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['model']:<15} {r['OA']*100:>10.2f} {r['AA']*100:>10.2f} {r['Kappa']:>12.4f}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    data: Optional[np.ndarray] = None,
    gt: Optional[np.ndarray] = None,
    cfg: Optional[NDModelConfig] = None,
):
    """
    Full training + evaluation pipeline.

    Parameters
    ----------
    data : (H, W, B) float32 array. If None, synthetic data is generated.
    gt   : (H, W) int32 array with 1-based class labels.
    cfg  : NDModelConfig instance.
    """
    if cfg is None:
        cfg = NDModelConfig()

    # --- data ---
    if data is None or gt is None:
        print("\n[INFO] No real data provided – generating synthetic ND crop data.")
        gen = NDCropDataGenerator()
        gen.print_info()
        data, gt = gen.generate()
    else:
        print(f"\n[INFO] Using provided data: shape={data.shape}, gt unique={np.unique(gt)}")

    # Plot spectral signatures
    plot_spectral_signatures(data, gt, ND_CROP_INFO["class_names"])

    # Build patch dataset
    print("\n[INFO] Extracting patches…")
    X, y = build_patch_dataset(data, gt, patch_size=cfg.spatial_size)
    print(f"  Patches: {X.shape},  Labels: {y.shape}")

    train_loader, val_loader, test_loader = split_and_load(X, y, cfg)

    # --- models ---
    model_factories = {
        "ResNet-CNN": lambda: NDCropCNN(cfg),
        "ViT":        lambda: NDCropViT(cfg),
        "3D-CNN":     lambda: NDCrop3DCNN(cfg),
    }

    histories = {}
    all_results = []
    device = torch.device(cfg.device)

    for name, factory in model_factories.items():
        print(f"\n{'='*60}")
        print(f"  Training {name}")
        print(f"{'='*60}")
        model = factory()
        history = train_model(model, train_loader, val_loader, cfg, model_name=name)
        histories[name] = history

        results = evaluate_model(model, test_loader, device, ND_CROP_INFO["class_names"], model_name=name)
        all_results.append(results)
        plot_confusion_matrix(results, ND_CROP_INFO["class_names"])

        # Save checkpoint
        os.makedirs("checkpoints/nd_crop", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/nd_crop/{name.replace(' ', '_')}.pth")

    plot_training_curves(histories, metric="val_acc")
    plot_training_curves(histories, metric="val_loss")
    plot_benchmark_table(all_results)

    print("\n[INFO] Project 1 complete. All outputs saved to outputs/nd_crop/")
    return all_results, histories


if __name__ == "__main__":
    results, histories = run_pipeline()
