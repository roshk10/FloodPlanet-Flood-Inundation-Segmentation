from pathlib import Path


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if Path("D:/Basemodel/Datasets").exists():
    DATASET_ROOT = Path("D:/Basemodel/Datasets")
elif Path("D:/Basemodel/FloodPlanet").exists():
    DATASET_ROOT = Path("D:/Basemodel/FloodPlanet")
else:
    DATASET_ROOT = PROJECT_ROOT.parents[1] / "Datasets"

EVENT_SPLIT_FILE = (
    PROJECT_ROOT
    / "configs"
    / "event_split.json"
)

CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "checkpoints"
)

METRICS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

PREDICTIONS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "predictions"
)


# ================================================================
# DATA
# ================================================================

IN_CHANNELS = 4
NUM_CLASSES = 2

PATCH_SIZE = 512
STRIDE = 512

IGNORE_INDEX = -1


# ================================================================
# TRAINING
# ================================================================

BATCH_SIZE = 4
NUM_WORKERS = 2

MAX_EPOCHS = 20

SEED = 2026


# ================================================================
# OPTIMIZER
# ================================================================

OPTIMIZER = "AdamW"

LEARNING_RATE = 6e-4
BACKBONE_LEARNING_RATE = 6e-5

WEIGHT_DECAY = 2.5e-4


# ================================================================
# SCHEDULER
# ================================================================

SCHEDULER = "CosineAnnealingWarmRestarts"

SCHEDULER_T0 = 15
SCHEDULER_T_MULT = 2


# ================================================================
# MODEL
# ================================================================

PRETRAINED = False

BACKBONE_PATH = (
    PROJECT_ROOT
    / "src"
    / "models"
    / "pretrain_weights"
    / "rest_lite.pth"
)

EMBED_DIM = 64
DECODE_CHANNELS = 64


# ================================================================
# LOSS
# ================================================================

LOSS_NAME = "UnetMambaLoss"

SOFT_CE_SMOOTH = 0.05
DICE_SMOOTH = 0.05

AUXILIARY_WEIGHT = 0.4


# ================================================================
# AUGMENTATION
# ================================================================

AUGMENT_FLIP = True
AUGMENT_ROTATION = True


# ================================================================
# FIXED EVENT-LEVEL PROTOCOL
# ================================================================

PROTOCOL = "fixed_event_level_split"

TRAIN_EVENTS = [
    "Colombia",
    "Ghana",
    "Nigeria",
    "Paraguay",
    "Somalia",
    "Spain",
    "US-Alabama",
    "US-Arkansas",
    "US-Carolina",
    "US-Kansas",
    "US-Nebraska",
    "US-Oklahoma",
    "US-Texas",
    "Uzbekistan",
]

VALIDATION_EVENTS = [
    "Bolivia",
    "US-Dakota",
]

TEST_EVENTS = [
    "Bangladesh",
    "Cambodia",
    "Nepal",
]

MONITOR_METRIC = "val_mIoU"

TTA = False


# ================================================================
# CHECKPOINTING
# ================================================================

SAVE_LAST = True
SAVE_BEST = True
SAVE_EVERY_EPOCH = True
