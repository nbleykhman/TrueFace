import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms

from config import (
    BASE_FACES, BASE_TPDNE, BASE_DALLE,
    device,
    PRETRAIN_EPOCHS, PRETRAIN_BATCH_SIZE, LR_PRETRAIN,
    FINETUNE_EPOCHS, FINETUNE_BATCH_SIZE, LR_FINETUNE,
    CHECKPOINT,
)
from dataset import FaceDataset
from model import get_model
from sklearn.metrics import roc_auc_score

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--stage',
        choices=['pretrain', 'finetune'],
        default='pretrain',
        help="'pretrain' = big low-res @256px; 'finetune' = joint TPDNE+140K high-res @1024px"
    )
    args = parser.parse_args()

    # Stage-specific settings
    if args.stage == 'pretrain':
        root, epochs, batch_size, lr = BASE_FACES, PRETRAIN_EPOCHS, PRETRAIN_BATCH_SIZE, LR_PRETRAIN
        use_warmup = False
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        val_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        ds_train = FaceDataset(os.path.join(root, 'train'), transform=train_tf)
        ds_val   = FaceDataset(os.path.join(root, 'valid'), transform=val_tf)
        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=12, pin_memory=True)
        val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

    else:
        root, epochs, batch_size, lr = BASE_TPDNE, FINETUNE_EPOCHS, FINETUNE_BATCH_SIZE, LR_FINETUNE
        use_warmup = True
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(1024, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomApply([transforms.GaussianBlur(7)], p=0.7),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        val_tf = transforms.Compose([
            transforms.Resize(1024),
            transforms.CenterCrop(1024),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        # build and balance joint dataset
        ds_tp   = FaceDataset(os.path.join(BASE_TPDNE, 'train'), transform=train_tf)
        ds_140k = FaceDataset(os.path.join(BASE_FACES,  'train'), transform=train_tf)
        ds_dalle = FaceDataset(os.path.join(BASE_DALLE,  'train'), transform=train_tf)
        train_ds = ConcatDataset([ds_tp, ds_140k, ds_dalle])

        ds_tp_v   = FaceDataset(os.path.join(BASE_TPDNE, 'valid'), transform=val_tf)
        ds_140k_v = FaceDataset(os.path.join(BASE_FACES,  'valid'), transform=val_tf)
        ds_dalle_v = FaceDataset(os.path.join(BASE_DALLE,  'valid'), transform=val_tf)
        val_ds    = ConcatDataset([ds_tp_v, ds_140k_v, ds_dalle_v])

        # balanced sampling of two domains
        n1, n2, n3 = len(ds_tp), len(ds_140k), len(ds_dalle)
        weights = [1/n1]*n1 + [1/n2]*n2 + [1/n3]*n3
        sampler = WeightedRandomSampler(weights, num_samples=2 * min(n1, n2), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,  num_workers=12, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

    print(f"[Stage={args.stage}] Epochs={epochs}, BS={batch_size}, LR={lr}")

    # model, loss, optimizer w/ stronger weight decay
    model     = get_model().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    # schedulers: optional 1-epoch warmup, then ReduceLROnPlateau
    warmup_epochs    = 1
    warmup_sched     = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: min((e+1)/warmup_epochs, 1.0))
    plateau_sched    = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    best_val_auc = 0.0
    for epoch in range(epochs):
        # — train —
        model.train()
        train_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)
    
        # — validate —
        model.eval()
        val_loss, all_probs, all_labels = 0.0, [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                imgs, labels = imgs.to(device), labels.to(device)
                out  = model(imgs)
                val_loss += criterion(out, labels).detach().item() * imgs.size(0)
                probs    = torch.softmax(out, dim=1)[:,1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_auc  = roc_auc_score(all_labels, all_probs)

        # step LR
        if use_warmup and epoch < warmup_epochs:
            warmup_sched.step()
        else:
            plateau_sched.step(val_loss)

        # checkpoint on improvement
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"→ Saved best model (epoch {epoch+1}, Val AUC={val_auc:.4f})")

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}")

if __name__ == '__main__':
    print("Using device:", device)
    train()
