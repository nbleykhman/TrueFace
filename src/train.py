import os
import io
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from config import *
from dataset import FaceDataset
from model import get_model
from sklearn.metrics import roc_auc_score

# Top-level helper for JPEG corruption

def random_jpeg(img, quality=(30,95), p=0.5):
    if random.random() > p:
        return img
    buf = io.BytesIO()
    q = random.randint(quality[0], quality[1])
    img.save(buf, format='JPEG', quality=q)
    buf.seek(0)
    return Image.open(buf).convert('RGB')

# Module-level noise transform to avoid pickling issues
class AddNoise:
    def __init__(self, std):
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std

# Mixup data augmentation
def mixup_data(x, y, alpha):
    lam = np.random.beta(alpha,alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x2, y2 = x[idx], y[idx]
    return lam*x + (1-lam)*x2, y, y2, lam

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['pretrain','finetune'], default='pretrain')
    args = parser.parse_args()

    # Setup stage-specific parameters
    if args.stage == 'pretrain':
        root, epochs, batch_size, lr, eta_min, res = (
            BASE_FACES, PRETRAIN_EPOCHS,
            PRETRAIN_BATCH_SIZE, LR_PRETRAIN, PRETRAIN_ETA_MIN,
            PRETRAIN_RESOLUTION
        )
        results_file = 'pretrain_results.txt'
    else:
        root, epochs, batch_size, lr, eta_min, res = (
            BASE_TPDNE, FINETUNE_EPOCHS,
            FINETUNE_BATCH_SIZE, LR_FINETUNE, FINETUNE_ETA_MIN,
            FINETUNE_RESOLUTION
        )
        results_file = 'finetune_results.txt'

    # Prepare results file
    with open(results_file, 'w') as rf:
        rf.write('epoch,train_loss,val_loss,val_auc\n')

    # Transforms
    light_train_tf = transforms.Compose([
        transforms.Resize(res),
        transforms.CenterCrop(res),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    full_train_tf = transforms.Compose([
        transforms.RandomResizedCrop(res,
                                    scale=(0.6, 1.0),
                                    ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=RAND_AUGMENT_N, magnitude=RAND_AUGMENT_M),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.1)
        ], p=0.5),
        random_jpeg,
        transforms.GaussianBlur(
            kernel_size=3 if args.stage=='pretrain' else 7,
            sigma=(0.1,2.0) if args.stage=='pretrain' else (0.1,3.0)
        ),
        transforms.ToTensor(),
        AddNoise(0.02 if args.stage=='pretrain' else 0.01),
        transforms.RandomErasing(
            p=0.5 if args.stage=='pretrain' else 0.4,
            scale=(0.02,0.2), ratio=(0.3,3.3)
        ),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(res),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Dataset & DataLoader
    if args.stage == 'pretrain':
        ds_train = FaceDataset(os.path.join(root,'train'), transform=light_train_tf)
        ds_val   = FaceDataset(os.path.join(root,'valid'), transform=val_tf)
        sampler  = None
    else:
        ds_list = [
            FaceDataset(os.path.join(p,'train'), transform=light_train_tf)
            for p in (BASE_TPDNE, BASE_FACES, BASE_DALLE)
        ]
        ds_train = ConcatDataset(ds_list)
        weights  = sum(([1/len(ds)]*len(ds) for ds in ds_list), [])
        sampler  = WeightedRandomSampler(weights, len(ds_train), replacement=True)
        ds_val   = ConcatDataset([
            FaceDataset(os.path.join(p,'valid'), transform=val_tf)
            for p in (BASE_TPDNE, BASE_FACES, BASE_DALLE)
        ])
    val_loader = DataLoader(
        ds_val, batch_size=batch_size,
        shuffle=False, num_workers=12, pin_memory=True
    )

    # Model, optimizer, loss
    model = get_model().to(device)
    ema   = get_model().to(device)
    if args.stage == 'finetune':
        if os.path.exists(PRETRAIN_CHECKPOINT):
            print(f"Loading pretrain weights from {PRETRAIN_CHECKPOINT}")
            model.load_state_dict(torch.load(PRETRAIN_CHECKPOINT, map_location=device))
        else:
            print(f"Warning: Pretrain checkpoint not found at {PRETRAIN_CHECKPOINT}. Training from scratch.")
    ema.load_state_dict(model.state_dict())
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    # Schedulers
    sched_w = LambdaLR(optimizer, lr_lambda=lambda e: min((e+1)/WARMUP_EPOCHS, 1.0))
    sched_c = CosineAnnealingLR(optimizer, T_max=max(1, epochs-WARMUP_EPOCHS), eta_min=eta_min)

    # Training loop with early stopping
    best_auc, no_improve = 0.0, 0
    patience = 4 if args.stage == "pretrain" else 2
    for epoch in range(epochs):
        if epoch < WARMUP_EPOCHS and args.stage == 'pretrain':
            active_tf = light_train_tf
        else: 
            active_tf = full_train_tf
        ds_train.transform = active_tf

        if isinstance(ds_train, ConcatDataset):
            for sub_ds in ds_train.datasets:
                sub_ds.transform = active_tf
        
        train_loader = DataLoader(
            ds_train, batch_size=batch_size,
            sampler=sampler, shuffle=(sampler is None),
            num_workers=12, pin_memory=True
        )
        model.train()
        train_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"{args.stage} Train {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Apply MixUp if enabled
            if args.stage == 'finetune' and ENABLE_MIXUP:
                x, y1, y2, lam = mixup_data(imgs, labels, MIXUP_ALPHA)
                out = model(x)
                loss = lam * criterion(out, y1) + (1 - lam) * criterion(out, y2)
            else:
                out = model(imgs)
                loss = criterion(out, labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item() * imgs.size(0)

            # EMA update
            for p, e in zip(model.parameters(), ema.parameters()):
                e.data.mul_(EMA_DECAY).add_(p.data*(1-EMA_DECAY))

        train_loss /= len(ds_train)
        for m_src, m_ema in zip(model.modules(), ema.modules()):
            if isinstance(m_src, nn.BatchNorm2d):
                m_ema.running_mean.data.copy_(m_src.running_mean.data)
                m_ema.running_var.data.copy_(m_src.running_var.data)
        ema.eval()
        val_loss = 0.0
        probs, labs = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"{args.stage} Val   {epoch+1}/{epochs}"):
                imgs, labels = imgs.to(device), labels.to(device)
                # base
                logits = ema(imgs)
                if True:
                    logits += ema(torch.flip(imgs, [-1]))
                    logits /= 2
                val_loss += criterion(logits, labels).detach().item()*imgs.size(0)
                p = torch.softmax(logits,1)[:,1].cpu().numpy()
                probs.extend(p)
                labs.extend(labels.cpu().numpy())
        val_loss /= len(ds_val)
        val_auc = roc_auc_score(labs, probs)

        # LR step
        (sched_w if epoch < WARMUP_EPOCHS else sched_c).step()

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            no_improve = 0
            # save both raw and EMA checkpoints
            model_path = f"{args.stage}_model_epoch{epoch+1}.pth"
            ema_path   = f"{args.stage}_ema_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)
            torch.save(ema.state_dict(),   ema_path)
            print(f"→ Saved raw → {model_path}, EMA → {ema_path} (AUC={val_auc:.4f})")
        else:
            no_improve += 1

        print(f"{args.stage} Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, AUC={val_auc:.4f}")

        # Append results
        with open(results_file, 'a') as rf:
            rf.write(f"{epoch+1},{train_loss:.4f},{val_loss:.4f},{val_auc:.4f}\n")

        if no_improve >= patience:
            print(f"No improvement for {patience} epochs—stopping early.")
            break

if __name__=='__main__':
    print("Using device:", device)
    train()
