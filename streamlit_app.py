"""
Project 3: Interactive Streamlit Demo
=======================================
An interactive web application that lets users:
  1. Upload a hyperspectral .mat file **or** select a built-in sample
     (Indian Pines, Pavia University, Salinas, or the synthetic ND Crops)
  2. Choose a trained model (ResNet-CNN, ViT, 3D-CNN for classification;
     U-Net for segmentation)
  3. View: false-colour image → model prediction → colour-coded map

Run locally
-----------
    pip install streamlit scipy
    streamlit run streamlit_app.py

Deploy free
-----------
  Streamlit Cloud : https://streamlit.io/cloud
  Hugging Face Spaces : https://huggingface.co/spaces
"""

import io
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional, Tuple, Dict, List

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional import – app still works as a plain script for smoke-testing
# ---------------------------------------------------------------------------
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False

try:
    from scipy.io import loadmat as _loadmat
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Reuse model definitions from the other project files
# ---------------------------------------------------------------------------

# ---- lightweight inline definitions (no file-system dependency) -----------

class _SpAttn(nn.Module):
    def __init__(self, c):
        super().__init__()
        r = max(c // 8, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c, r), nn.ReLU(inplace=True),
            nn.Linear(r, c), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x).view(x.size(0), x.size(1), 1, 1)


class _CB(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False),
            nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.a = _SpAttn(o)
    def forward(self, x):
        return self.a(self.c(x))


class DemoResNet(nn.Module):
    def __init__(self, in_c, nc, base=32):
        super().__init__()
        self.net = nn.Sequential(
            _CB(in_c, base), _CB(base, base * 2), _CB(base * 2, base * 4),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(base * 4, nc))
    def forward(self, x):
        return self.net(x)


class DemoViT(nn.Module):
    def __init__(self, in_c, nc, sp=9, dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_c, dim, 1)
        self.cls = nn.Parameter(torch.randn(1, 1, dim))
        self.pe = nn.Parameter(torch.randn(1, sp ** 2 + 1, dim))
        el = nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=256,
                                        dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, num_layers=2)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, nc)
    def forward(self, x):
        B = x.size(0)
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1) + self.pe
        return self.head(self.norm(self.enc(x))[:, 0])


class Demo3DCNN(nn.Module):
    def __init__(self, in_c, nc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 8, (7, 3, 3), padding=(3, 1, 1), bias=False),
            nn.BatchNorm3d(8), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(8, nc))
    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1)))


# U-Net for segmentation
class _DC(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.b(x)


class DemoUNet(nn.Module):
    def __init__(self, in_c, nc, base=32):
        super().__init__()
        self.e1 = _DC(in_c, base);   self.p1 = nn.MaxPool2d(2)
        self.e2 = _DC(base, base*2); self.p2 = nn.MaxPool2d(2)
        self.bn = _DC(base*2, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.d2 = _DC(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.d1 = _DC(base*2, base)
        self.out = nn.Conv2d(base, nc, 1)
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        b  = self.bn(self.p2(e2))
        d2 = self.d2(torch.cat([self.u2(b), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "Indian Pines": {
        "num_classes": 16,
        "num_bands": 180,
        "patch_size": 9,
        "class_names": [
            "Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture",
            "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats",
            "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat",
            "Woods", "Buildings-Grass", "Stone-Steel-Towers",
        ],
        "type": "classification",
        "mat_data_key": "indian_pines_corrected",
        "mat_gt_key": "indian_pines_gt",
    },
    "Pavia University": {
        "num_classes": 9,
        "num_bands": 103,
        "patch_size": 9,
        "class_names": [
            "Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets",
            "Bare Soil", "Bitumen", "Self-Blocking Bricks", "Shadows",
        ],
        "type": "classification",
        "mat_data_key": "paviaU",
        "mat_gt_key": "paviaU_gt",
    },
    "Salinas": {
        "num_classes": 16,
        "num_bands": 204,
        "patch_size": 9,
        "class_names": [
            "Brocoli_1", "Brocoli_2", "Fallow", "Fallow_rough", "Fallow_smooth",
            "Stubble", "Celery", "Grapes", "Soil", "Corn", "Lettuce_4wk",
            "Lettuce_5wk", "Lettuce_6wk", "Lettuce_7wk", "Vinyard", "Vinyard_trellis",
        ],
        "type": "classification",
        "mat_data_key": "salinas_corrected",
        "mat_gt_key": "salinas_gt",
    },
    "ND Crops (Synthetic)": {
        "num_classes": 5,
        "num_bands": 13,
        "patch_size": 9,
        "class_names": ["Soybean", "Wheat", "Barley", "Corn", "Sunflower"],
        "type": "classification",
        "mat_data_key": None,
        "mat_gt_key": None,
    },
}

MODEL_CHOICES = {
    "ResNet-CNN (Classification)": "resnet",
    "ViT (Classification)":        "vit",
    "3D-CNN (Classification)":     "3dcnn",
    "U-Net (Segmentation)":        "unet",
}

COLORMAPS = [
    "#440154", "#3b528b", "#21918c", "#5ec962", "#fde725",  # viridis-like
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#e6194b", "#3cb44b", "#ffe119",
]


# ---------------------------------------------------------------------------
# Synthetic ND crop data (no external download needed)
# ---------------------------------------------------------------------------

def _make_nd_synthetic(h=120, w=120, seed=42):
    rng = np.random.default_rng(seed)
    num_bands = 13
    spectral = {
        0: [0.05, 0.07, 0.06, 0.04, 0.35, 0.42, 0.45, 0.43, 0.47, 0.48, 0.22, 0.14, 0.10],
        1: [0.08, 0.11, 0.10, 0.08, 0.28, 0.30, 0.32, 0.31, 0.33, 0.34, 0.28, 0.20, 0.16],
        2: [0.07, 0.10, 0.09, 0.07, 0.30, 0.33, 0.35, 0.34, 0.36, 0.37, 0.26, 0.18, 0.14],
        3: [0.04, 0.06, 0.05, 0.03, 0.40, 0.48, 0.52, 0.50, 0.54, 0.55, 0.20, 0.12, 0.08],
        4: [0.06, 0.09, 0.08, 0.06, 0.32, 0.36, 0.38, 0.37, 0.39, 0.40, 0.30, 0.24, 0.20],
    }
    data = np.zeros((h, w, num_bands), dtype=np.float32)
    gt   = np.zeros((h, w), dtype=np.int32)
    ph = h // 5
    for cls in range(5):
        rs = cls * ph
        re = (cls + 1) * ph if cls < 4 else h
        noise = rng.normal(0, 0.02, (re - rs, w, num_bands)).astype(np.float32)
        data[rs:re, :, :] = np.clip(np.array(spectral[cls]) + noise, 0, 1)
        gt[rs:re, :] = cls + 1
    return data, gt


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def normalise(data: np.ndarray) -> np.ndarray:
    """Per-band z-score normalisation."""
    h, w, b = data.shape
    flat = data.reshape(-1, b).astype(np.float32)
    mean = flat.mean(0)
    std  = flat.std(0) + 1e-8
    return ((flat - mean) / std).reshape(h, w, b)


def false_colour(data: np.ndarray, bands: Tuple[int, int, int] = None) -> np.ndarray:
    """Return (H, W, 3) uint8 false-colour image."""
    nb = data.shape[-1]
    if bands is None:
        r_idx = min(nb - 1, int(nb * 0.7))
        g_idx = min(nb - 1, int(nb * 0.4))
        b_idx = min(nb - 1, int(nb * 0.1))
        bands = (r_idx, g_idx, b_idx)
    rgb = data[..., list(bands)].astype(np.float32)
    lo, hi = rgb.min(), rgb.max()
    rgb = ((rgb - lo) / (hi - lo + 1e-8) * 255).clip(0, 255).astype(np.uint8)
    return rgb


def build_class_colormap(n_classes: int) -> np.ndarray:
    cmap = np.zeros((n_classes, 3), dtype=np.uint8)
    for i in range(n_classes):
        hex_color = COLORMAPS[i % len(COLORMAPS)]
        cmap[i] = [int(hex_color[j:j+2], 16) for j in (1, 3, 5)]
    return cmap


def predict_full_image(
    model: nn.Module,
    data: np.ndarray,
    patch_size: int = 9,
    device: str = "cpu",
    batch_size: int = 512,
) -> np.ndarray:
    """Slide a patch over every pixel and return per-pixel predictions."""
    pad = patch_size // 2
    padded = np.pad(data, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    H, W = data.shape[:2]

    model = model.to(device)
    model.eval()
    pred_map = np.zeros((H, W), dtype=np.int32)

    batch_x, batch_coords = [], []

    def _flush(bx, bc):
        t = torch.tensor(np.array(bx, dtype=np.float32)).to(device)
        with torch.no_grad():
            out = model(t).argmax(1).cpu().numpy()
        for (py, px), p in zip(bc, out):
            pred_map[py, px] = p

    for y in range(H):
        for x in range(W):
            p = padded[y: y + patch_size, x: x + patch_size, :].transpose(2, 0, 1)
            batch_x.append(p)
            batch_coords.append((y, x))
            if len(batch_x) == batch_size:
                _flush(batch_x, batch_coords)
                batch_x, batch_coords = [], []
    if batch_x:
        _flush(batch_x, batch_coords)

    return pred_map


def colorise_map(pred_map: np.ndarray, n_classes: int) -> np.ndarray:
    cmap = build_class_colormap(n_classes)
    h, w = pred_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(n_classes):
        rgb[pred_map == c] = cmap[c]
    return rgb


def render_legend(class_names: List[str]) -> plt.Figure:
    """Return a small matplotlib figure with a colour legend."""
    n = len(class_names)
    cmap = build_class_colormap(n)
    fig, ax = plt.subplots(figsize=(3, max(2, n * 0.35)))
    for i, name in enumerate(class_names):
        ax.add_patch(plt.Rectangle((0, i), 1, 1, color=cmap[i] / 255.0))
        ax.text(1.15, i + 0.5, name, va="center", fontsize=8)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, n)
    ax.axis("off")
    ax.set_title("Legend", fontsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

def run_streamlit_app():
    st.set_page_config(
        page_title="🌾 Hyperspectral Deep Learning Demo",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ---- sidebar ----
    st.sidebar.title("⚙️ Configuration")
    dataset_choice = st.sidebar.selectbox("Dataset", list(DATASET_CONFIGS.keys()))
    model_choice   = st.sidebar.selectbox("Model", list(MODEL_CHOICES.keys()))
    patch_size     = st.sidebar.slider("Patch size (px)", min_value=5, max_value=15, value=9, step=2)
    n_epochs       = st.sidebar.slider("Quick-train epochs", min_value=5, max_value=50, value=15)
    run_btn        = st.sidebar.button("🚀 Run Demo")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Upload your own .mat file** (optional):\n"
        "The file must contain arrays `data` and `gt` (or the dataset-specific keys)."
    )
    uploaded = st.sidebar.file_uploader("Upload .mat file", type=["mat"])

    # ---- main ----
    st.title("🌾 Hyperspectral Deep Learning – Interactive Demo")
    st.markdown(
        "Classify or segment hyperspectral imagery from Indian Pines, Pavia University, "
        "Salinas, or the synthetic ND Crops dataset. "
        "Upload your own `.mat` file or use the built-in samples."
    )

    if not run_btn:
        st.info("Configure the options in the sidebar and click **Run Demo**.")
        st.stop()

    cfg = DATASET_CONFIGS[dataset_choice]
    n_classes = cfg["num_classes"]
    n_bands   = cfg["num_bands"]
    sp        = patch_size
    class_names = cfg["class_names"]
    model_key   = MODEL_CHOICES[model_choice]

    # ---- load data ----
    with st.spinner("Loading data…"):
        if uploaded is not None and _HAS_SCIPY:
            mat = _loadmat(io.BytesIO(uploaded.read()))
            data_key = cfg["mat_data_key"] or "data"
            gt_key   = cfg["mat_gt_key"]   or "gt"
            data = mat[data_key].astype(np.float32)
            gt   = mat[gt_key].astype(np.int32).squeeze()
            n_bands = data.shape[-1]
            n_classes = len(np.unique(gt[gt > 0]))
            class_names = [f"Class {i}" for i in range(1, n_classes + 1)]
            st.success(f"Loaded custom data: {data.shape}, {n_classes} classes")
        else:
            if dataset_choice == "ND Crops (Synthetic)":
                data, gt = _make_nd_synthetic()
            else:
                st.warning(
                    "Real dataset files not found locally (requires prior download). "
                    "Falling back to synthetic ND Crops data."
                )
                data, gt = _make_nd_synthetic()
                n_classes = 5
                n_bands   = 13
                class_names = DATASET_CONFIGS["ND Crops (Synthetic)"]["class_names"]

    data_norm = normalise(data)

    # ---- false-colour image ----
    fc = false_colour(data)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📷 False-Colour Image")
        st.image(fc, caption=f"{dataset_choice} – false-colour (NIR/Red/Green)", use_container_width=True)

    # ---- build & train model (quick) ----
    with st.spinner(f"Training {model_choice} for {n_epochs} epochs…"):
        device = "cpu"

        if model_key == "resnet":
            model = DemoResNet(n_bands, n_classes)
        elif model_key == "vit":
            model = DemoViT(n_bands, n_classes, sp=sp)
        elif model_key == "3dcnn":
            model = Demo3DCNN(n_bands, n_classes)
        else:  # unet
            model = DemoUNet(n_bands, n_classes)

        # Build a small training set from the labelled pixels
        pad = sp // 2
        padded = np.pad(data_norm, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        ys, xs = np.where(gt > 0)

        # Sub-sample for speed in the demo (max 4000 pixels)
        if len(ys) > 4000:
            idx = np.random.default_rng(42).choice(len(ys), 4000, replace=False)
            ys, xs = ys[idx], xs[idx]

        patches, labels = [], []
        for y, x in zip(ys, xs):
            p = padded[y: y + sp, x: x + sp, :].transpose(2, 0, 1)
            patches.append(p)
            labels.append(int(gt[y, x]) - 1)

        X = torch.tensor(np.array(patches, dtype=np.float32))
        y_tensor = torch.tensor(labels, dtype=torch.long)
        ds = torch.utils.data.TensorDataset(X, y_tensor)
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(n_epochs):
            for xb, yb in loader:
                opt.zero_grad()
                out = model(xb)
                if model_key == "unet":
                    B, C, H_out, W_out = out.shape
                    yb_map = yb.view(B, 1, 1).expand(B, H_out, W_out)
                    loss = loss_fn(out, yb_map)
                else:
                    loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
        model.eval()

    # ---- predict full image ----
    with st.spinner("Running full-image prediction…"):
        H, W = data_norm.shape[:2]
        # For large images, downscale to max 80x80 for demo speed
        scale = 1.0
        if H > 80 or W > 80:
            scale = 80.0 / max(H, W)
            H2, W2 = max(1, int(H * scale)), max(1, int(W * scale))
            from PIL import Image as _PIL
            data_small = np.array(
                _PIL.fromarray(data_norm[:, :, 0]).resize((W2, H2), _PIL.BILINEAR)
            )[:, :, None]
            data_small = np.concatenate([
                np.array(_PIL.fromarray(data_norm[:, :, b]).resize((W2, H2), _PIL.BILINEAR))[:, :, None]
                for b in range(data_norm.shape[-1])
            ], axis=-1)
            pred_map = predict_full_image(model, data_small, patch_size=sp, device=device)
        else:
            pred_map = predict_full_image(model, data_norm, patch_size=sp, device=device)

    seg_rgb = colorise_map(pred_map, n_classes)

    with col2:
        st.subheader("🎨 Predicted Class Map")
        st.image(seg_rgb, caption=f"{model_choice} prediction", use_container_width=True)

    with col3:
        st.subheader("📊 Legend")
        leg_fig = render_legend(class_names)
        st.pyplot(leg_fig, use_container_width=True)

    # ---- class distribution bar chart ----
    st.subheader("📊 Predicted Class Distribution")
    unique, counts = np.unique(pred_map, return_counts=True)
    cmap_arr = build_class_colormap(n_classes)
    bar_colors = [
        f"#{cmap_arr[c, 0]:02x}{cmap_arr[c, 1]:02x}{cmap_arr[c, 2]:02x}"
        for c in unique if c < n_classes
    ]
    bar_labels = [class_names[c] if c < len(class_names) else f"Class {c}" for c in unique]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(bar_labels, counts[:len(bar_labels)], color=bar_colors)
    ax.set_xlabel("Class")
    ax.set_ylabel("Pixel Count")
    ax.set_title("Predicted Class Distribution")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    st.success("✅ Demo complete!")
    st.markdown(
        "**Next steps:**\n"
        "- Download real datasets to get proper predictions.\n"
        "- Increase epochs for better accuracy.\n"
        "- Deploy to [Streamlit Cloud](https://streamlit.io/cloud) or "
        "[Hugging Face Spaces](https://huggingface.co/spaces) for a public link."
    )


# ---------------------------------------------------------------------------
# Headless smoke-test (used when streamlit is not installed)
# ---------------------------------------------------------------------------

def smoke_test():
    """Run a headless test of the core inference pipeline."""
    print("[Smoke test] Building synthetic data and running inference…")
    data, gt = _make_nd_synthetic(h=40, w=40)
    data_norm = normalise(data)
    model = DemoResNet(in_c=13, nc=5)
    pred = predict_full_image(model, data_norm, patch_size=9)
    print(f"  Predicted map shape: {pred.shape}, unique classes: {np.unique(pred)}")
    seg_rgb = colorise_map(pred, n_classes=5)
    print(f"  Colour map shape: {seg_rgb.shape}")
    print("[Smoke test] PASSED")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if _HAS_STREAMLIT:
        run_streamlit_app()
    else:
        print(
            "Streamlit is not installed. Run:\n"
            "    pip install streamlit\n"
            "    streamlit run streamlit_app.py\n\n"
            "Running headless smoke-test instead…"
        )
        smoke_test()
