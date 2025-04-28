import os
import torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True       # auto-tune fastest conv kernels
torch.backends.cudnn.deterministic = False  # allow faster ops if non-deterministic

# Paths
this_dir = os.path.dirname(os.path.abspath(__file__))
BASE_FACES = os.path.join(this_dir, 'data/140k-real-vs-fake')
BASE_TPDNE = os.path.join(this_dir, 'data/tpdne')
BASE_DALLE = os.path.join(this_dir, 'data/dalle')

# Ensure data directories exist
for path in (BASE_FACES, BASE_TPDNE, BASE_DALLE):
    os.makedirs(path, exist_ok=True)

# Stage-specific hyperparameters
# Pre-training (low-res, big data)
PRETRAIN_EPOCHS      = 50            # total epochs for domain adaptation
PRETRAIN_BATCH_SIZE  = 200           # batch size for pretraining
LR_PRETRAIN          = 1e-4          # lower base LR for slower start
PRETRAIN_RESOLUTION  = 224           # resolution for pretraining
WARMUP_EPOCHS        = 5             # shorter warm-up to reach full LR sooner
PRETRAIN_ETA_MIN     = 1e-6          # final LR after cosine decay

# Fine-tuning (high-res, small data)
FINETUNE_EPOCHS      = 20            # epochs for high-res fine-tuning
FINETUNE_BATCH_SIZE  = 12            # batch size for fine-tuning
LR_FINETUNE          = 5e-5          # lower LR for stable fine-tuning
FINETUNE_RESOLUTION  = 1024          # resolution for fine-tuning
FINETUNE_ETA_MIN     = 1e-7          # final LR after cosine decay

# EMA
EMA_DECAY            = 0.9999        # decay for exponential moving average

# MixUp
ENABLE_MIXUP         = True         # enable MixUp augmentation ONLY FOR FINETUNING
MIXUP_ALPHA          = 0.2           # alpha for beta distribution

# RandAugment
RAND_AUGMENT_N       = 2             # number of augmentations to apply
RAND_AUGMENT_M       = 9             # magnitude of augmentations

# Checkpoint paths
PRETRAIN_CHECKPOINT  = os.path.join(this_dir, 'pretrain_checkpoint.pth')
FINETUNE_MODEL_CHECKPOINT  = os.path.join(this_dir, 'finetune_model_epoch5.pth')
FINETUNE_EMA_CHECKPOINT = os.path.join(this_dir, 'finetune_ema_epoch5.pth')