import torch
import os

# ============================================
# DEVICE CONFIGURATION (Auto GPU/CPU)
# ============================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()  # Mixed precision only on GPU
print(f"Using device: {DEVICE}")
print(f"Mixed precision training: {USE_AMP}")

# ============================================
# MODEL SELECTION
# ============================================
# Text Encoder Options: "banglabert", "xlm-roberta"
TEXT_ENCODER = "banglabert"

# Image Encoder Options: "swin", "vit"
IMAGE_ENCODER = "swin"

# Fusion Strategy Options: "summation", "concatenation", "coattention"
FUSION_TYPE = "concatenation"

# ============================================
# LOSS WEIGHTS
# Paper tested combinations: (0.2, 0.8), (0.5, 0.5), (0.8, 0.2), (1.0, 1.0)
# ============================================
ALPHA = 0.5  # Weight for category loss
BETA = 0.5   # Weight for target group loss

# ============================================
# MODEL ARCHITECTURE
# ============================================
HIDDEN_SIZE = 768
DROPOUT = 0.4
NUM_ATTENTION_HEADS = 8  # For self-attention and co-attention

# ============================================
# HIERARCHICAL CLASSIFICATION STRUCTURE
# ============================================
NUM_BINARY_CLASSES = 2      # Hate vs Non-Hate
NUM_CATEGORY_CLASSES = 5    # Abusive, Political, Gender, Personal, Religious
NUM_TARGET_CLASSES = 4      # Community, Individual, Organization, Society

# ============================================
# TRAINING HYPERPARAMETERS
# ============================================
BATCH_SIZE = 16              # Reduce to 8 or 4 if GPU memory is limited
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 10
WARMUP_STEPS = 0
GRADIENT_ACCUMULATION_STEPS = 1  # Increase if batch size is too small

# ============================================
# EARLY STOPPING
# ============================================
PATIENCE = 3                 # Stop if no improvement for 3 epochs
MIN_DELTA = 0.001           # Minimum improvement to consider

# ============================================
# DATASET PATHS
# ============================================
TRAIN_CSV = "train_data.csv"
VAL_CSV = "val_data.csv"
TEST_CSV = "test_data.csv"
IMAGE_DIR = "Memes"
MAX_TEXT_LENGTH = 128

# ============================================
# MODEL CHECKPOINTING
# ============================================
OUTPUT_DIR = f"output/{TEXT_ENCODER}_{IMAGE_ENCODER}_{FUSION_TYPE}_a{ALPHA}_b{BETA}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAVE_BEST_MODEL = True
SAVE_LAST_MODEL = True

# ============================================
# RANDOM SEED FOR REPRODUCIBILITY
# ============================================
RANDOM_SEED = 42

# ============================================
# LOGGING
# ============================================
LOG_INTERVAL = 10        
SAVE_RESULTS = True
SAVE_CONFUSION_MATRIX = True

# ============================================
# MODEL NAME MAPPINGS
# ============================================
MODEL_NAMES = {
    "banglabert": "csebuetnlp/banglabert",
    "xlm-roberta": "xlm-roberta-base",
    "vit": "google/vit-base-patch16-224",
    "swin": "microsoft/swin-base-patch4-window7-224-in22k"
}

# ============================================
# LABEL MAPPINGS
# ============================================
BINARY_LABELS = {0: "Non-Hate", 1: "Hate"}
CATEGORY_LABELS = {
    0: "Abusive",
    1: "Political",
    2: "Gender",
    3: "Personal",
    4: "Religious"
}
TARGET_LABELS = {
    0: "Community",
    1: "Individual",
    2: "Organization",
    3: "Society"
}
