import os
import re
import argparse
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

from config import (
    BASE_FACES,
    BASE_TPDNE,
    BASE_DALLE,
    device,
    FINETUNE_BATCH_SIZE,
    FINETUNE_RESOLUTION,
    FINETUNE_EMA_CHECKPOINT
)
from dataset import FaceDataset
from model import get_model
from train import IMAGENET_MEAN, IMAGENET_STD

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT     = os.path.dirname(SCRIPT_DIR)
EVAL_RESULTS_DIR = os.path.join(PROJECT_ROOT, "EMApipeline", "test_results")
THRESHOLD_DIR   = os.path.join(PROJECT_ROOT, "EMApipeline", "evaluation_results")
os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)


def test(loader, threshold, name):
    """Run inference on loader with p>threshold ⇒ fake; return (auc, report_str)."""
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labs.numpy())

    preds = [1 if p > threshold else 0 for p in all_probs]
    auc  = roc_auc_score(all_labels, all_probs)
    acc  = accuracy_score(all_labels, preds)
    cm   = confusion_matrix(all_labels, preds)
    cr   = classification_report(all_labels, preds, target_names=['Real','Fake'])

    header = f"\n=== {name} (thr={threshold:.3f}) ==="
    stats  = f"AUC: {auc:.4f}    Accuracy: {acc:.4f}"
    cm_txt = "Confusion Matrix:\n" + str(cm)
    cr_txt = "Classification Report:\n" + cr
    report = "\n".join([header, stats, cm_txt, cr_txt, ""])
    print(report)
    return auc, report


def make_loader(base, split, size):
    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = FaceDataset(os.path.join(base, split), transform=tf)
    return DataLoader(ds,
                      batch_size=FINETUNE_BATCH_SIZE,
                      shuffle=False,
                      num_workers=8,
                      pin_memory=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size", type=int, default=FINETUNE_RESOLUTION,
        help="Image size for resize/crop"
    )
    parser.add_argument(
        "--tests", nargs="+",
        choices=["tpdne", "140k", "dalle", "combined"],
        default=["tpdne", "140k", "dalle", "combined"],
        help="Which tests to run: tpdne, 140k, dalle, combined"
    )
    args = parser.parse_args()

    # Load EMA model
    model = get_model().to(device)
    ckpt  = torch.load(FINETUNE_EMA_CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded EMA checkpoint from {FINETUNE_EMA_CHECKPOINT}")

    # Read the combined threshold
    thr_file  = os.path.join(THRESHOLD_DIR, "combined_threshold.txt")
    threshold = float(open(thr_file).read().strip())
    print(f"Using combined threshold = {threshold:.4f}\n")

    # Build possible loaders
    size = args.size
    all_loaders = {
        "tpdne":   ("TPDNE Test",  make_loader(BASE_TPDNE,  "test", size)),
        "140k":    ("140K Test",   make_loader(BASE_FACES,  "test", size)),
        "dalle":   ("DALL-E Test", make_loader(BASE_DALLE,  "test", size)),
        "combined":("Combined Test", DataLoader(
            ConcatDataset([
                FaceDataset(os.path.join(b, "test"), transform=transforms.Compose([
                    transforms.Resize(size),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]))
                for b in (BASE_TPDNE, BASE_FACES, BASE_DALLE)
            ]),
            batch_size=FINETUNE_BATCH_SIZE,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        ))
    }

    results = {}
    for key in args.tests:
        name, loader = all_loaders[key]
        auc, rpt = test(loader, threshold, name)
        results[name] = (auc, rpt)

    out_path = os.path.join(EVAL_RESULTS_DIR, "test_results.txt")
    with open(out_path, "w") as f:
        for name in results:
            f.write(results[name][1] + "\n")
    print(f"\n✅ Written selected test reports to {out_path}")

    if "combined" in args.tests:
        new_auc = results["Combined Test"][0]
        prev_auc = -1.0
        if os.path.exists(out_path):
            txt = open(out_path).read()
            m   = re.search(r"Combined Test[^\n]*AUC:\s*([0-9.]+)", txt)
            if m:
                prev_auc = float(m.group(1))
        if new_auc > prev_auc:
            print(f"✅ Combined AUC improved ({prev_auc:.4f} → {new_auc:.4f})")
        else:
            print(f"⚠️ Combined AUC did not improve ({prev_auc:.4f} ≥ {new_auc:.4f})")

