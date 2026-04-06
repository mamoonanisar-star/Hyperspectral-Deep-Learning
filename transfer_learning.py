"""
Project 5: Transfer Learning from Public Datasets to ND Crops
==============================================================
Demonstrates three training strategies on a small ND crop dataset:

  1. Full training    – train from scratch on ND crop data
  2. Transfer learning – pre-train on Indian Pines / Salinas,
                         then fine-tune the last N layers on ND crop data
  3. Few-shot learning – fine-tune with only 10 labelled samples per class

Outcome: accuracy comparison table + training-curve plots.

Usage
-----
    python transfer_learning.py

All training runs use synthetic data by default. Set `real_data=True` in
`run_pipeline()` to load the real .mat files (requires prior download).
"""

import os
import copy
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Shared model – ResNet-style CNN usable for any number of bands / classes
# ---------------------------------------------------------------------------

class SpectralAttention(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        r = max(c // 8, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c, r), nn.ReLU(inplace=True),
            nn.Linear(r, c), nn.Sigmoid(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x).view(x.size(0), -1, 1, 1)


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        self.att = SpectralAttention(out_c)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.att(self.conv(x))


class FeatureExtractor(nn.Module):
    """
    Shared backbone (stem + 3 conv blocks).
    This is the part that gets pre-trained on source datasets and then
    frozen / fine-tuned on the target ND-crop dataset.
    """
    def __init__(self, in_channels: int, base: int = 64):
        super().__init__()
        self.stem   = ConvBlock(in_channels, base)
        self.layer1 = ConvBlock(base,     base * 2)
        self.layer2 = ConvBlock(base * 2, base * 4)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.out_dim = base * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.pool(x).flatten(1)          # (B, out_dim)


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class TransferNet(nn.Module):
    """Full model = FeatureExtractor + ClassifierHead."""
    def __init__(self, in_channels: int, num_classes: int, base: int = 64):
        super().__init__()
        self.backbone = FeatureExtractor(in_channels, base)
        self.head     = ClassifierHead(self.backbone.out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def freeze_backbone(self):
        """Freeze all backbone parameters (transfer / few-shot modes)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_layers(self, n: int = 2):
        """
        Unfreeze the last n ConvBlock layers of the backbone for fine-tuning.
        n=1 → unfreeze layer2, n=2 → unfreeze layer1+layer2, etc.
        """
        layers = [self.backbone.stem, self.backbone.layer1, self.backbone.layer2]
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------

SOURCE_DATASETS = {
    "indian_pines":  {"nc": 16, "nb": 180, "h": 100, "w": 100},
    "salinas":       {"nc": 16, "nb": 204, "h": 100, "w": 100},
}

ND_CROP_CONFIG = {
    "nc": 5, "nb": 13, "h": 150, "w": 150,
    "class_names": ["Soybean", "Wheat", "Barley", "Corn", "Sunflower"],
}

ND_SPECTRAL_MEANS = [
    [0.05, 0.07, 0.06, 0.04, 0.35, 0.42, 0.45, 0.43, 0.47, 0.48, 0.22, 0.14, 0.10],
    [0.08, 0.11, 0.10, 0.08, 0.28, 0.30, 0.32, 0.31, 0.33, 0.34, 0.28, 0.20, 0.16],
    [0.07, 0.10, 0.09, 0.07, 0.30, 0.33, 0.35, 0.34, 0.36, 0.37, 0.26, 0.18, 0.14],
    [0.04, 0.06, 0.05, 0.03, 0.40, 0.48, 0.52, 0.50, 0.54, 0.55, 0.20, 0.12, 0.08],
    [0.06, 0.09, 0.08, 0.06, 0.32, 0.36, 0.38, 0.37, 0.39, 0.40, 0.30, 0.24, 0.20],
]


def _synth_source(name: str, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    p = SOURCE_DATASETS[name]
    nc, nb, H, W = p["nc"], p["nb"], p["h"], p["w"]
    rng = np.random.default_rng(seed)
    data = np.zeros((H, W, nb), dtype=np.float32)
    gt   = np.zeros((H, W),     dtype=np.int32)
    ph   = H // nc
    for cls in range(nc):
        rs = cls * ph
        re = (cls + 1) * ph if cls < nc - 1 else H
        mean = rng.uniform(0.05, 0.6, nb).astype(np.float32)
        noise = rng.normal(0, 0.03, (re - rs, W, nb)).astype(np.float32)
        data[rs:re, :, :] = np.clip(mean + noise, 0, 1)
        gt[rs:re, :] = cls + 1
    return data, gt


def _synth_nd_crops(seed: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    cfg = ND_CROP_CONFIG
    nc, nb, H, W = cfg["nc"], cfg["nb"], cfg["h"], cfg["w"]
    rng = np.random.default_rng(seed)
    data = np.zeros((H, W, nb), dtype=np.float32)
    gt   = np.zeros((H, W),     dtype=np.int32)
    ph   = H // nc
    for cls in range(nc):
        rs = cls * ph
        re = (cls + 1) * ph if cls < nc - 1 else H
        mean = np.array(ND_SPECTRAL_MEANS[cls], dtype=np.float32)
        noise = rng.normal(0, 0.02, (re - rs, W, nb)).astype(np.float32)
        data[rs:re, :, :] = np.clip(mean + noise, 0, 1)
        gt[rs:re, :] = cls + 1
    return data, gt


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_patches(
    data: np.ndarray,
    gt: np.ndarray,
    patch_size: int = 9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w, b = data.shape
    flat = data.reshape(-1, b).astype(np.float32)
    flat = (flat - flat.mean(0)) / (flat.std(0) + 1e-8)
    data_n = flat.reshape(h, w, b)

    pad = patch_size // 2
    padded = np.pad(data_n, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    ys, xs = np.where(gt > 0)
    patches, labels = [], []
    for y, x in zip(ys, xs):
        p = padded[y: y + patch_size, x: x + patch_size, :].transpose(2, 0, 1)
        patches.append(p)
        labels.append(int(gt[y, x]) - 1)
    X = torch.tensor(np.array(patches, dtype=np.float32))
    y = torch.tensor(labels, dtype=torch.long)
    return X, y


def make_loaders(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    idx = np.arange(len(y))
    i_tv, i_te = train_test_split(idx, test_size=0.10, stratify=y.numpy(), random_state=42)
    i_tr, i_va = train_test_split(i_tv, test_size=0.111, stratify=y[i_tv].numpy(), random_state=42)

    def loader(ind, shuffle):
        return DataLoader(TensorDataset(X[ind], y[ind]),
                          batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader(i_tr, True), loader(i_va, False), loader(i_te, False)


def make_few_shot_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    k_shot: int = 10,
    batch_size: int = 64,
) -> DataLoader:
    """Return a DataLoader with at most k_shot samples per class."""
    rng = np.random.default_rng(42)
    selected = []
    for cls in range(int(y.max().item()) + 1):
        cls_idx = (y == cls).nonzero(as_tuple=True)[0].numpy()
        chosen  = rng.choice(cls_idx, size=min(k_shot, len(cls_idx)), replace=False)
        selected.extend(chosen.tolist())
    ds = TensorDataset(X[selected], y[selected])
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)


# ---------------------------------------------------------------------------
# Generic training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainCfg:
    num_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    patience: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _run_epoch(
    model, loader, criterion, optimizer, device, train: bool
):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct    += (out.argmax(1) == yb).sum().item()
            total      += xb.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainCfg,
    label: str = "",
) -> Dict:
    device = torch.device(cfg.device)
    model = model.to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = 0.0
    patience_counter = 0

    for epoch in range(1, cfg.num_epochs + 1):
        tl, ta = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        scheduler.step()
        vl, va = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)

        if va > best_val:
            best_val = va
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    [{label}] Epoch {epoch:3d}/{cfg.num_epochs}"
                  f"  val_acc={va:.4f}  val_loss={vl:.4f}")

        if patience_counter >= cfg.patience:
            print(f"    [{label}] Early stopping at epoch {epoch}")
            break

    return history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
) -> Dict:
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(dev)
            preds = model(xb).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(yb.numpy())
    oa = accuracy_score(all_targets, all_preds)
    kappa = cohen_kappa_score(all_targets, all_preds)
    unique = sorted(set(all_targets))
    per_class = []
    for c in unique:
        mask = [t == c for t in all_targets]
        if sum(mask) == 0:
            continue
        per_class.append(accuracy_score(
            [t for t, m in zip(all_targets, mask) if m],
            [p for p, m in zip(all_preds, mask) if m],
        ))
    aa = float(np.mean(per_class)) if per_class else 0.0
    return {"OA": oa, "AA": aa, "Kappa": kappa}


# ---------------------------------------------------------------------------
# Pre-training on source dataset
# ---------------------------------------------------------------------------

def pretrain_on_source(
    source_name: str,
    nb_target: int,
    base_channels: int,
    cfg: TrainCfg,
    real_data: bool = False,
    patch_size: int = 9,
) -> FeatureExtractor:
    """
    Pre-train a FeatureExtractor on a source (Indian Pines / Salinas) dataset.
    Because the source has a different number of bands, we use a band projection
    layer (1x1 conv) to map source bands → target bands before the shared backbone.

    Returns the pre-trained FeatureExtractor ready for transfer to ND crops.
    """
    print(f"\n  [Pre-train] Source: {source_name}")
    if real_data:
        try:
            from scipy.io import loadmat
            keys = {
                "indian_pines":  ("datasets/indian_pines_data.mat",   "indian_pines_corrected",
                                   "datasets/indian_pines_gt.mat",     "indian_pines_gt"),
                "salinas":       ("datasets/salinas_data.mat",         "salinas_corrected",
                                   "datasets/salinas_gt.mat",           "salinas_gt"),
            }
            dp, dk, gp, gk = keys[source_name]
            src_data = loadmat(dp)[dk].astype(np.float32)
            src_gt   = loadmat(gp)[gk].astype(np.int32).squeeze()
            print(f"    Loaded real {source_name}: {src_data.shape}")
        except Exception as e:
            print(f"    [WARN] Could not load real data ({e}); using synthetic fallback.")
            src_data, src_gt = _synth_source(source_name)
    else:
        src_data, src_gt = _synth_source(source_name)
        print(f"    Using synthetic {source_name}: {src_data.shape}")

    src_nb = src_data.shape[-1]
    src_nc = len(np.unique(src_gt[src_gt > 0]))

    # Project source bands → target bands using a 1×1 conv (trained jointly)
    class SourceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.band_proj = nn.Conv2d(src_nb, nb_target, 1, bias=False)
            self.backbone  = FeatureExtractor(nb_target, base_channels)
            self.head      = ClassifierHead(self.backbone.out_dim, src_nc)
        def forward(self, x):
            return self.head(self.backbone(self.band_proj(x)))

    model = SourceModel()
    X, y = extract_patches(src_data, src_gt, patch_size)
    tr, va, _ = make_loaders(X, y, cfg.batch_size)
    train_model(model, tr, va, cfg, label=f"pretrain-{source_name}")

    return model.backbone


# ---------------------------------------------------------------------------
# Adapter to reuse a pre-trained backbone on different band counts
# ---------------------------------------------------------------------------

class AdaptedTransferNet(nn.Module):
    """
    Wraps a pre-trained backbone (trained on source bands) with a
    band-projection layer and a new head for the target dataset.
    """
    def __init__(
        self,
        pretrained_backbone: FeatureExtractor,
        nb_source: int,
        nb_target: int,
        num_classes_target: int,
    ):
        super().__init__()
        self.band_proj = nn.Conv2d(nb_target, nb_target, 1, bias=False)
        self.backbone  = pretrained_backbone
        self.head      = ClassifierHead(pretrained_backbone.out_dim, num_classes_target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(self.band_proj(x)))

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_layers(self, n: int = 2):
        layers = [self.backbone.stem, self.backbone.layer1, self.backbone.layer2]
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    real_data: bool = False,
    k_shot: int = 10,
    patch_size: int = 9,
    base_channels: int = 32,
):
    os.makedirs("outputs/transfer_learning", exist_ok=True)
    os.makedirs("checkpoints/transfer_learning", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*65}")
    print(f"  Project 5: Transfer Learning")
    print(f"  Device: {device}  |  Few-shot k={k_shot}")
    print(f"{'='*65}")

    cfg_pretrain  = TrainCfg(num_epochs=20, learning_rate=1e-3, device=device)
    cfg_finetune  = TrainCfg(num_epochs=30, learning_rate=5e-4, device=device)
    cfg_full      = TrainCfg(num_epochs=30, learning_rate=1e-3, device=device)
    cfg_fewshot   = TrainCfg(num_epochs=30, learning_rate=5e-4, patience=5, device=device)

    # ---- ND Crops dataset ----
    nd_cfg = ND_CROP_CONFIG
    print("\n[INFO] Loading ND Crops data…")
    nd_data, nd_gt = _synth_nd_crops()
    nb_nd = nd_data.shape[-1]
    nc_nd = nd_cfg["nc"]

    X_nd, y_nd = extract_patches(nd_data, nd_gt, patch_size)
    tr_full, va_full, te_full = make_loaders(X_nd, y_nd, cfg_full.batch_size)
    fs_loader = make_few_shot_loader(X_nd, y_nd, k_shot=k_shot)
    print(f"  Full dataset : {len(X_nd)} patches")
    print(f"  Few-shot set : {len(fs_loader.dataset)} patches  ({k_shot} per class)")

    all_results: Dict[str, Dict] = {}
    all_histories: Dict[str, Dict] = {}

    # ================================================================
    # Strategy 1: Full Training from scratch
    # ================================================================
    print(f"\n{'─'*65}")
    print("  Strategy 1: Full Training (from scratch)")
    print(f"{'─'*65}")

    model_full = TransferNet(nb_nd, nc_nd, base_channels)
    h_full = train_model(model_full, tr_full, va_full, cfg_full, label="full-train")
    res_full = evaluate_model(model_full, te_full, device)
    all_results["Full Training"] = res_full
    all_histories["Full Training"] = h_full
    print(f"  OA={res_full['OA']*100:.2f}%  AA={res_full['AA']*100:.2f}%  κ={res_full['Kappa']:.4f}")
    torch.save(model_full.state_dict(), "checkpoints/transfer_learning/full_training.pth")

    # ================================================================
    # Strategy 2: Transfer Learning (pre-train on source → fine-tune on ND)
    # ================================================================
    for src_name in SOURCE_DATASETS:
        strat_name = f"Transfer ({src_name.replace('_', ' ').title()})"
        print(f"\n{'─'*65}")
        print(f"  Strategy 2: {strat_name}")
        print(f"{'─'*65}")

        pretrained_bb = pretrain_on_source(
            src_name, nb_nd, base_channels, cfg_pretrain,
            real_data=real_data, patch_size=patch_size,
        )

        model_tl = AdaptedTransferNet(
            pretrained_backbone=copy.deepcopy(pretrained_bb),
            nb_source=SOURCE_DATASETS[src_name]["nb"],
            nb_target=nb_nd,
            num_classes_target=nc_nd,
        )
        # Freeze backbone, train head only first
        model_tl.freeze_backbone()
        h_tl_head = train_model(model_tl, tr_full, va_full,
                                 TrainCfg(num_epochs=10, learning_rate=1e-3, device=device),
                                 label=f"tl-head-{src_name}")
        # Unfreeze last 2 backbone layers for fine-tuning
        model_tl.unfreeze_last_n_layers(n=2)
        h_tl_ft = train_model(model_tl, tr_full, va_full, cfg_finetune,
                               label=f"tl-finetune-{src_name}")

        # Merge histories
        h_combined = {
            "train_loss": h_tl_head["train_loss"] + h_tl_ft["train_loss"],
            "val_loss":   h_tl_head["val_loss"]   + h_tl_ft["val_loss"],
            "train_acc":  h_tl_head["train_acc"]  + h_tl_ft["train_acc"],
            "val_acc":    h_tl_head["val_acc"]     + h_tl_ft["val_acc"],
        }

        res_tl = evaluate_model(model_tl, te_full, device)
        all_results[strat_name]   = res_tl
        all_histories[strat_name] = h_combined
        print(f"  OA={res_tl['OA']*100:.2f}%  AA={res_tl['AA']*100:.2f}%  κ={res_tl['Kappa']:.4f}")
        torch.save(model_tl.state_dict(),
                   f"checkpoints/transfer_learning/tl_{src_name}.pth")

    # ================================================================
    # Strategy 3: Few-Shot Fine-Tuning (pre-train on Indian Pines, k-shot ND)
    # ================================================================
    print(f"\n{'─'*65}")
    print(f"  Strategy 3: Few-Shot Learning  (k={k_shot} per class)")
    print(f"{'─'*65}")

    best_src = "indian_pines"
    pretrained_bb_fs = pretrain_on_source(
        best_src, nb_nd, base_channels, cfg_pretrain,
        real_data=real_data, patch_size=patch_size,
    )
    model_fs = AdaptedTransferNet(
        pretrained_backbone=copy.deepcopy(pretrained_bb_fs),
        nb_source=SOURCE_DATASETS[best_src]["nb"],
        nb_target=nb_nd,
        num_classes_target=nc_nd,
    )
    model_fs.freeze_backbone()
    h_fs = train_model(model_fs, fs_loader, va_full, cfg_fewshot, label="few-shot")
    res_fs = evaluate_model(model_fs, te_full, device)
    all_results["Few-Shot (k=10)"]   = res_fs
    all_histories["Few-Shot (k=10)"] = h_fs
    print(f"  OA={res_fs['OA']*100:.2f}%  AA={res_fs['AA']*100:.2f}%  κ={res_fs['Kappa']:.4f}")
    torch.save(model_fs.state_dict(), "checkpoints/transfer_learning/few_shot.pth")

    # ================================================================
    # Summary table
    # ================================================================
    print(f"\n{'='*65}")
    print(f"{'Strategy':<40} {'OA (%)':>8} {'AA (%)':>8} {'Kappa':>8}")
    print(f"{'─'*65}")
    for name, res in all_results.items():
        print(f"{name:<40} {res['OA']*100:>8.2f} {res['AA']*100:>8.2f} {res['Kappa']:>8.4f}")
    print(f"{'='*65}")

    # ================================================================
    # Plots
    # ================================================================

    # -- Training curves comparison --
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, h in all_histories.items():
        axes[0].plot(h["val_loss"], label=name)
        axes[1].plot(h["val_acc"],  label=name)
    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=7)
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=7)
    plt.suptitle("Transfer Learning Strategy Comparison – ND Crops")
    plt.tight_layout()
    curves_path = "outputs/transfer_learning/strategy_comparison.png"
    plt.savefig(curves_path, dpi=150)
    plt.show()
    print(f"\nSaved → {curves_path}")

    # -- Bar chart OA comparison --
    names  = list(all_results.keys())
    oas    = [all_results[n]["OA"] * 100 for n in names]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(names, oas, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_ylabel("Test OA (%)")
    ax.set_title("Overall Accuracy by Training Strategy – ND Crops")
    plt.xticks(rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    bar_path = "outputs/transfer_learning/oa_bar_chart.png"
    plt.savefig(bar_path, dpi=150)
    plt.show()
    print(f"Saved → {bar_path}")

    print("\n[INFO] Project 5 complete. All outputs saved to outputs/transfer_learning/")
    return all_results, all_histories


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transfer Learning – ND Crops")
    parser.add_argument("--real-data", action="store_true",
                        help="Load real .mat files from datasets/ directory")
    parser.add_argument("--k-shot", type=int, default=10,
                        help="Samples per class for few-shot learning")
    args = parser.parse_args()
    run_pipeline(real_data=args.real_data, k_shot=args.k_shot)
