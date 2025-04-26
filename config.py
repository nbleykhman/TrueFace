import os
import torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True       # auto-tune fastest conv kernels
torch.backends.cudnn.deterministic = False  # allow non-deterministic but faster ops

# Paths
this_dir = os.path.dirname(os.path.abspath(__file__))
BASE_FACES = os.path.join(this_dir, 'data/140k-real-vs-fake')
BASE_TPDNE = os.path.join(this_dir, 'data/tpdne')
BASE_DALLE = os.path.join(this_dir, 'data/dalle')

# Stage-specific hyperparameters

# Pre-training (low-res, big data)
PRETRAIN_EPOCHS = 2
PRETRAIN_BATCH_SIZE = 200
LR_PRETRAIN = 1e-3

# Fine-tuning (high-res, small data)
FINETUNE_EPOCHS = 5
FINETUNE_BATCH_SIZE = 12
LR_FINETUNE = 5e-5

CHECKPOINT = os.path.join(this_dir, 'checkpoint.pth')