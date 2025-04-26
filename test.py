import os
import re
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

from config import BASE_FACES, BASE_TPDNE, BASE_DALLE, device, FINETUNE_BATCH_SIZE, CHECKPOINT
from dataset import FaceDataset
from model import get_model

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def evaluate(loader, name):
    """Run inference on loader, print & return AUC plus formatted text."""
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out  = model(imgs)
            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    preds = [1 if p > 0.5 else 0 for p in all_probs]
    auc  = roc_auc_score(all_labels, all_probs)
    acc  = accuracy_score(all_labels, preds)
    cm   = confusion_matrix(all_labels, preds)
    cr   = classification_report(all_labels, preds, target_names=['Real','Fake'])

    header = f"\n=== {name} ==="
    stats  = f"AUC: {auc:.4f}    Accuracy: {acc:.4f}"
    cm_txt = "Confusion Matrix:\n" + str(cm)
    cr_txt = "Classification Report:\n" + cr

    report = "\n".join([header, stats, cm_txt, cr_txt])
    print(report, end="\n\n")
    return auc, report

if __name__ == "__main__":
    # Load fine-tuned model
    model = get_model().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    # High-res test transform
    test_tf = transforms.Compose([
        transforms.Resize(1024),
        transforms.CenterCrop(1024),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # 1) TPDNE only
    ds_tpdne     = FaceDataset(os.path.join(BASE_TPDNE, 'test'), transform=test_tf)
    loader_tpdne = DataLoader(ds_tpdne,
                              batch_size=FINETUNE_BATCH_SIZE,
                              shuffle=False, num_workers=8, pin_memory=True)
    auc_tpdne, rpt_tpdne = evaluate(loader_tpdne, "TPDNE Test")

    # 2) 140K only
    ds_140k     = FaceDataset(os.path.join(BASE_FACES, 'test'), transform=test_tf)
    loader_140k = DataLoader(ds_140k,
                             batch_size=FINETUNE_BATCH_SIZE,
                             shuffle=False, num_workers=8, pin_memory=True)
    auc_140k, rpt_140k = evaluate(loader_140k, "140K Test")

    # 4) DALL-E only
    ds_dalle     = FaceDataset(os.path.join(BASE_DALLE, 'test'), transform=test_tf)
    loader_dalle = DataLoader(ds_dalle,
                              batch_size=FINETUNE_BATCH_SIZE,
                              shuffle=False, num_workers=8, pin_memory=True)
    auc_dalle, rpt_dalle = evaluate(loader_dalle, "DALL-E Test")

    # 4) Combined
    combined_ds = ConcatDataset([ds_tpdne, ds_140k, ds_dalle])
    loader_comb = DataLoader(combined_ds,
                             batch_size=FINETUNE_BATCH_SIZE,
                             shuffle=False, num_workers=8, pin_memory=True)
    auc_comb, rpt_comb = evaluate(loader_comb, "Combined Test")

    # Prepare full report text
    full_report = "\n".join([rpt_tpdne, rpt_140k, rpt_comb])

    out_path = "test_results.txt"
    # Read previous Combined AUC if file exists
    prev_auc = -1.0
    if os.path.exists(out_path):
        txt = open(out_path, 'r').read()
        m = re.search(r"Combined Test[^\n]*AUC:\s*([0-9.]+)", txt)
        if m:
            prev_auc = float(m.group(1))

    # Only overwrite if combined AUC improved
    if auc_comb > prev_auc:
        with open(out_path, 'w') as f:
            f.write(full_report)
        print(f"✅ Combined AUC improved ({prev_auc:.4f} → {auc_comb:.4f}); results saved to {out_path}")
    else:
        print(f"⚠️ Combined AUC did not improve ({prev_auc:.4f} ≥ {auc_comb:.4f}); no file update")