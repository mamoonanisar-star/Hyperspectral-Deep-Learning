# Hyperspectral-Deep-Learning
## 🔍 Hyperspectral Image Classification and Segmentation
A comprehensive implementation of deep learning models for hyperspectral image classification and segmentation using PyTorch. The project implements various architectures and techniques in a single streamlined pipeline.

## 📚 Table of Contents
1. **🔄 Data Acquisition and Preprocessing**
   - 📥 Downloading Datasets
   - 🔃 Data Format Conversion and Organization
   - ✂️ Dataset Splitting (80/10/10)
   - 📊 Band Quality Assessment
   - 🧹 Noise Removal

2. **📈 Data Exploration and Visualization**
   - 📉 Spectral Signatures Visualization
   - 🔗 Band Correlation Analysis
   - 📊 Class Distribution Analysis
   - 🎲 3D Datacube Visualizations

3. **🎯 Dimensionality Reduction and Feature Selection**
   - 🧮 PCA, ICA, LDA Analysis
   - 🎯 Band Selection Methods
   - ✨ Reduced Representations Evaluation

4. **🔄 Data Augmentation Strategies**
   - 🌊 Spectral Augmentation
   - 🌍 Spatial Augmentation
   - 🔄 Combined Spectral-Spatial Augmentation

5. **🏗️ Model Architecture Design**
   - 📋 Classification Models
     - 🔥 ResNet18 with Spectral Attention
     - 🤖 Vision Transformer (ViT)
     - 🧊 3D CNN
     - 🔄 Hybrid CNN-Transformer
   - 🎨 Segmentation Models
     - 🌈 U-Net with Spectral Attention
     - 🎲 3D U-Net
     - 🎯 FCN with Spectral Attention

6. **⚙️ Training Pipeline Implementation**
   - 📉 Loss Functions
     - 💫 Cross-entropy with Class Weights
     - 🎯 Dice Loss for Segmentation
   - 🔧 Optimization
     - ⚡ Adam Optimizer
     - 📈 Learning Rate Scheduling
   - 🔒 Regularization
     - 🎭 Spectral Dropout
     - 🏋️ L1/L2 Regularization

7. **📊 Model Training and Monitoring**
   - 🎯 Model Selection
   - 🏃‍♂️ Training and Validation
   - 💾 Checkpoint Management
   - 🛡️ Overfitting Handling

8. **📊 Results and Visualization**
   - 📈 Performance Charts
   - 🎨 Color-graded and Normalized Visualizations

## ✨ Features
### 📚 Supported Datasets
- 🌾 Indian Pines
- 🏛️ Pavia University
- 🌱 Salinas Scene

### 🛠️ Core Functionality
- 🔄 Automatic dataset downloading and processing
- 📊 Band quality assessment and noise removal
- 🔄 Spectral and spatial data augmentation
- 📉 Dimensionality reduction techniques
- ⚡ Advanced training features with regularization
- 📊 Comprehensive visualization and evaluation tools

## 📋 Requirements
```text
torch>=1.8.0
numpy>=1.19.2
pandas>=1.2.0
scipy>=1.6.0
scikit-learn>=0.24.0
matplotlib>=3.3.4
seaborn>=0.11.1
tqdm>=4.59.0
plotly>=4.14.0
```

## 🚀 Usage
1. Install the required dependencies:
```bash
pip install torch numpy pandas scipy scikit-learn matplotlib seaborn tqdm plotly
```

2. Run the main training pipeline:
```python
# Select dataset and models
SELECTED_DATASET = ['indian_pines']  # Options: 'indian_pines', 'pavia_university', 'salinas'
models = ['fcn']  # Options: 'resnet', 'vit', '3dcnn', 'hybrid', 'unet', 'fcn'
# Training will automatically:
# - Download and process the dataset
# - Train the selected models
# - Generate visualizations and metrics
```

## 🔑 Key Components
### 📊 Data Processing
```python
# Load and preprocess dataset
loader = HyperspectralDataLoader(dataset_name)
data, ground_truth = loader.load_dataset()
# Analyze band quality
analyzer = BandQualityAnalyzer(data, dataset_name)
noisy_bands = analyzer.identify_noisy_bands()
```

### 🏃‍♂️ Model Training
```python
# Create model
model = create_model(dataset_name)  # For classification
# or
model = create_fcn_model(dataset_name)  # For segmentation
# Train
trainer = Trainer(config, task_config)
trainer.train()
```

### 📈 Visualization
```python
# Plot results
plot_confusion_matrix(dataset_name, classification_models, device)
plot_segmentation_maps(dataset_name, segmentation_models, device)
plot_training_curves(metrics, dataset_name, model_type)
```

## 📊 Results
The code includes comprehensive evaluation tools that generate:
- 📈 Classification accuracy metrics
- 🎯 Segmentation IoU/Dice scores
- 📊 Confusion matrices
- 📉 ROC curves
- 🔍 Error analysis
- 📊 Band quality visualization
- 📈 Training progress curves

---

## 🚀 Portfolio Projects

Five standalone projects built on top of the pipeline. Each can be run
independently from the repo root.

---

### Project 1 – 🌾 ND Crop Type Classifier (`nd_crop_classifier.py`)

Classifies North Dakota crops (soybean, wheat, barley, corn, sunflower)
using 13-band Sentinel-2-style multispectral data. Three model types are
compared: ResNet-CNN, ViT, and 3D-CNN.

**Outputs** saved to `outputs/nd_crop/`:
- Spectral signature plot per crop type
- Confusion matrix (%) for every model
- Training-curve plots (accuracy & loss)
- OA / AA / Kappa comparison table

```bash
python nd_crop_classifier.py
```

To use **real Sentinel-2 / USDA CroplandCROS data**, replace the
`NDCropDataGenerator.generate()` call in `run_pipeline()` with your own
loader returning `(H, W, bands)` and `(H, W)` ground-truth arrays.

---

### Project 2 – 🌿 Crop Stress / Disease Detection (`crop_stress_detection.py`)

Detects drought stress, fungal disease, and nutrient deficiency using:
- Vegetation indices: **NDVI**, **NDRE**, **MCARI**
- A fine-tuned **U-Net** segmentation model
- A per-patch **health score** (0–100 %)

```bash
python crop_stress_detection.py
```

**Outputs** saved to `outputs/crop_stress/`:
- `vi_maps.png`             – NDVI / NDRE / MCARI / health-score maps
- `health_score_dist.png`   – per-class health-score histograms
- `segmentation_map.png`    – predicted vs ground-truth stress map
- `training_curves.png`     – U-Net loss & accuracy curves

---

### Project 3 – 📊 Interactive Streamlit Demo (`streamlit_app.py`)

A web app that lets anyone try the pipeline without writing code:

```bash
pip install streamlit
streamlit run streamlit_app.py
```

Features:
- Upload a custom `.mat` hyperspectral file **or** pick a built-in sample
  (Indian Pines, Pavia University, Salinas, ND Crops)
- Choose a model: ResNet-CNN, ViT, 3D-CNN, U-Net
- View: false-colour image → predicted class map → colour legend
- Class-distribution bar chart

**Deploy for free:**

| Platform | Command |
|----------|---------|
| [Streamlit Cloud](https://streamlit.io/cloud) | Push repo → connect → share link |
| [Hugging Face Spaces](https://huggingface.co/spaces) | Upload files, select Streamlit SDK |

---

### Project 4 – 🔬 Benchmark Comparison Report (`benchmark_report.py`)

Runs all 6 model types on up to 3 datasets and produces a
publication-quality comparison:

```bash
# Quick synthetic run (no downloads needed)
python benchmark_report.py

# Full run on real datasets (requires downloads)
python benchmark_report.py --real-data

# Only specific models / datasets
python benchmark_report.py --models resnet vit fcn --datasets indian_pines salinas
```

**Outputs** saved to `outputs/benchmark/`:
- `cm_<dataset>_<model>.png`       – confusion matrices
- `roc_<dataset>_<model>.png`      – ROC curves
- `curves_<dataset>_<model>.png`   – training curves
- `benchmark_oa.png / aa.png / kappa.png` – summary heatmaps
- `benchmark_results.csv`          – full numerical table

---

### Project 5 – 🧠 Transfer Learning (`transfer_learning.py`)

Compares three training strategies on the ND crop dataset:

| Strategy | Description |
|----------|-------------|
| **Full Training** | Train ResNet-CNN from scratch on ND crops |
| **Transfer Learning** | Pre-train on Indian Pines / Salinas, fine-tune on ND crops |
| **Few-Shot (k=10)** | Pre-trained backbone + only 10 labelled samples per class |

```bash
python transfer_learning.py

# With real source datasets:
python transfer_learning.py --real-data

# Change few-shot k:
python transfer_learning.py --k-shot 5
```

**Outputs** saved to `outputs/transfer_learning/`:
- `strategy_comparison.png` – val loss & accuracy for all strategies
- `oa_bar_chart.png`         – OA bar chart with values

---

## 📦 Installation

```bash
pip install -r requirements.txt
```
