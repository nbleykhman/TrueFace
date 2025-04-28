import os
import argparse
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

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

SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT      = os.path.dirname(SCRIPT_DIR)
EVAL_RESULTS_DIR  = os.path.join(PROJECT_ROOT, "EMApipeline", "evaluation_results")
os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

def find_best_threshold(model, loader):
    """Run model on loader and pick threshold that maximizes Youden's J."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device)
            out  = model(imgs)
            p = F.softmax(out, 1)[:, 1].cpu().numpy()
            probs.extend(p)
            labels.extend(labs.numpy())
    fpr, tpr, ths = roc_curve(labels, probs)
    j_scores = tpr - fpr
    best_idx = j_scores.argmax()
    return ths[best_idx]

def evaluate(model, loader, threshold, name):
    """Evaluate model on loader using a fixed threshold; return report str."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device)
            out  = model(imgs)
            p = F.softmax(out, 1)[:, 1].cpu().numpy()
            probs.extend(p)
            labels.extend(labs.numpy())

    preds = [1 if p > threshold else 0 for p in probs]
    auc  = roc_auc_score(labels, probs)
    acc  = accuracy_score(labels, preds)
    cm   = confusion_matrix(labels, preds)
    cr   = classification_report(labels, preds, target_names=['Real','Fake'])

    header = f"\n=== {name} (thr={threshold:.3f}) ==="
    stats  = f"AUC: {auc:.4f}    Accuracy: {acc:.4f}"
    cm_txt = "Confusion Matrix:\n" + str(cm)
    cr_txt = "Classification Report:\n" + cr
    report = "\n".join([header, stats, cm_txt, cr_txt, ""])
    print(report)
    return report

def make_loader(base_path, split, size, batch_size):
    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = FaceDataset(os.path.join(base_path, split), transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=8, pin_memory=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size", type=int, default=FINETUNE_RESOLUTION,
        help="Image size for resize/crop"
    )
    parser.add_argument(
        "--output", type=str, default="evaluation_results.txt",
        help="Filename (inside EMApipeline/evaluation_results) to write reports"
    )
    args = parser.parse_args()

    output_path = os.path.join(EVAL_RESULTS_DIR, args.output)

    # Load EMA-smoothed model
    model = get_model().to(device)
    ckpt  = torch.load(FINETUNE_EMA_CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt)
    print(f"Loaded EMA checkpoint from {FINETUNE_EMA_CHECKPOINT}\n")

    reports = []
    datasets = [
        ("TPDNE", BASE_TPDNE),
        ("140K",  BASE_FACES),
        ("DALL-E", BASE_DALLE),
    ]

    # Per-domain threshold search & test evaluation
    for name, base in datasets:
        val_loader  = make_loader(base, "valid", args.size, FINETUNE_BATCH_SIZE)
        test_loader = make_loader(base, "test",  args.size, FINETUNE_BATCH_SIZE)

        best_thr = find_best_threshold(model, val_loader)
        rpt = evaluate(model, test_loader, best_thr, f"{name} Test")
        reports.append(rpt)

    # Combined validation & test
    combined_val_ds  = ConcatDataset([
        FaceDataset(os.path.join(b, "valid"), transform=transforms.Compose([
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])) for _, b in datasets
    ])
    combined_test_ds = ConcatDataset([
        FaceDataset(os.path.join(b, "test"), transform=transforms.Compose([
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])) for _, b in datasets
    ])
    comb_val_loader  = DataLoader(combined_val_ds,  batch_size=FINETUNE_BATCH_SIZE,
                                  shuffle=False, num_workers=8, pin_memory=True)
    comb_test_loader = DataLoader(combined_test_ds, batch_size=FINETUNE_BATCH_SIZE,
                                  shuffle=False, num_workers=8, pin_memory=True)

    best_comb_thr = find_best_threshold(model, comb_val_loader)
    rpt = evaluate(model, comb_test_loader, best_comb_thr, "Combined Test")
    reports.append(rpt)

    # Save the combined optimal threshold
    thr_path = os.path.join(EVAL_RESULTS_DIR, "combined_threshold.txt")
    with open(thr_path, "w") as f:
        f.write(f"{best_comb_thr:.6f}\n")
    print(f"\nCombined optimal threshold saved to {thr_path}")

    # Write all reports to file
    with open(output_path, "w") as f:
        for r in reports:
            f.write(r + "\n")
    print(f"\n✅ All reports written to {output_path}")