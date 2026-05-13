# config.py — Central configuration for all modules
import os

# ── Paths ──
DATASET_DIR   = "dataset"
MODEL_DIR     = "models"
OUTPUT_DIR    = "outputs"
LABELS_CSV    = os.path.join(DATASET_DIR, "labels.csv")
PROTOTXT      = os.path.join(MODEL_DIR, "VGG_ILSVRC_16_layers_deploy.prototxt")
CAFFEMODEL    = os.path.join(MODEL_DIR, "VGG_ILSVRC_16_layers.caffemodel")
PCA_PATH      = os.path.join(MODEL_DIR, "pca_model.pkl")
GBR_PATH      = os.path.join(MODEL_DIR, "gbr_model.pkl")

# ── Dataset ──
CLASSES = ["good", "moderate", "unhealthy", "very_unhealthy", "hazardous"]

# SDI range per class (Smoke Density Index 0–1)
CLASS_SDI_RANGE = {
    "good":          (0.00, 0.17),
    "moderate":      (0.17, 0.33),
    "unhealthy":     (0.33, 0.50),
    "very_unhealthy":(0.50, 0.67),
    "hazardous":     (0.67, 1.00),
}

# AQI display properties
AQI_INFO = {
    "good":          {"pm25": "0–50",   "color_bgr": (0, 200, 0),    "color_hex": "#00C800"},
    "moderate":      {"pm25": "51–100", "color_bgr": (0, 200, 255),  "color_hex": "#FFD000"},
    "unhealthy":     {"pm25": "101–150","color_bgr": (0, 140, 255),  "color_hex": "#FF8C00"},
    "very_unhealthy":{"pm25": "151–200","color_bgr": (0, 60, 255),   "color_hex": "#FF3C00"},
    "hazardous":     {"pm25": "200+",   "color_bgr": (0, 0, 200),    "color_hex": "#C80000"},
}

# ── Preprocessing ──
IMG_SIZE        = (224, 224)
VGG_MEAN_BGR    = (103.939, 116.779, 123.68)
CLAHE_CLIP      = 2.0
CLAHE_GRID      = (8, 8)
GAUSS_KERNEL    = (5, 5)
GAUSS_SIGMA     = 1.5
CANNY_LOW       = 50
CANNY_HIGH      = 150
FOURIER_RADIUS  = 0.30          # 30% of image dimension
LAPLACIAN_THRESH= 500           # threshold for Fourier filter trigger

# ── Feature Extraction ──
VGG_LAYER       = "fc6"         # Layer to extract features from
FEATURE_DIM     = 4096          # VGG16 fc6 output dimension

# ── PCA + Regressor ──
PCA_COMPONENTS  = 128
# config.py — replace GBR_PARAMS with this

GBR_PARAMS = {
    "n_estimators":     100,     # was 200 — fewer trees
    "max_depth":        2,       # was 4 — shallower trees (most important fix)
    "learning_rate":    0.05,
    "subsample":        0.7,     # was 0.8 — more randomness
    "min_samples_leaf": 5,       # NEW — prevents memorising rare samples
    "max_features":     0.5,     # NEW — use only 50% of features per split
    "random_state":     42,
}


# ── Train/Val/Test Split ──
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# ── Inference ──
CONFIDENCE_THRESHOLD = 0.80     # Minimum regressor confidence to display
SDI_TO_PM25_SCALE    = 300      # PM2.5 (µg/m³) = SDI × 300
