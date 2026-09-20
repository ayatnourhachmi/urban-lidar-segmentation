"""
Entry point: train a PointNet / PointNet++ segmentation model on urban
LiDAR point clouds, evaluate it, then quantize it and report the
before/after memory footprint + accuracy comparison.

Usage:
    python main.py --model pointnet2 --epochs 10 --num_points 4096
    python main.py --model pointnet  --epochs 10 --num_points 2048 --data_root /path/to/real/dataset

If --data_root is omitted, trains on a procedurally generated synthetic
urban point cloud dataset (see dataset.py) so the full pipeline is
runnable and testable without downloading a real dataset first. Swap in
--data_root pointing at a preprocessed SemanticKITTI / Toronto-3D
directory (see dataset.py docstring) for real results.
"""

import argparse
import copy

import torch
from torch.utils.data import DataLoader

from dataset import SyntheticUrbanLiDAR, RealLiDARDataset, NUM_CLASSES, CLASS_NAMES
from models import build_model
from transformations import train_transform, eval_transform
from quantization import benchmark_fp32_vs_int8, print_benchmark_report


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["pointnet", "pointnet2"], default="pointnet2")
    p.add_argument("--data_root", type=str, default=None,
                    help="Path to preprocessed real dataset (points/, labels/ .npy pairs). "
                         "If omitted, uses synthetic data.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_points", type=int, default=4096)
    p.add_argument("--voxel_size", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="checkpoint.pt")
    return p.parse_args()


def build_dataloaders(args):
    tr_tf = train_transform(voxel_size=args.voxel_size, num_points=args.num_points)
    ev_tf = eval_transform(voxel_size=args.voxel_size, num_points=args.num_points)

    if args.data_root:
        full = RealLiDARDataset(args.data_root, transform=None)
        n_val = max(1, len(full) // 5)
        n_train = len(full) - n_val
        train_set, val_set = torch.utils.data.random_split(full, [n_train, n_val])
        train_set.dataset.transform = tr_tf
        val_set.dataset.transform = ev_tf
    else:
        train_set = SyntheticUrbanLiDAR(num_samples=160, points_per_sample=8000, transform=tr_tf, seed=1)
        val_set = SyntheticUrbanLiDAR(num_samples=40, points_per_sample=8000, transform=ev_tf, seed=2)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader


@torch.no_grad()
def confusion_matrix(preds, labels, num_classes):
    """preds, labels: flat LongTensors."""
    mask = (labels >= 0) & (labels < num_classes)
    preds, labels = preds[mask], labels[mask]
    idx = num_classes * labels + preds
    cm = torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm


def miou_from_cm(cm, eps=1e-8):
    """cm: (C, C) with rows = true, cols = pred. Returns mean IoU + per-class IoU."""
    cm = cm.float()
    tp = cm.diag()
    support = cm.sum(1)  # true counts
    pred = cm.sum(0)
    iou = tp / (support + pred - tp + eps)
    # ignore classes with no true pixels in this eval pass
    valid = support > 0
    mean = iou[valid].mean().item() if valid.any() else 0.0
    return mean, iou.tolist()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_correct, total_points = 0.0, 0, 0
    for points, labels in loader:
        points, labels = points.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(points)  # (B, N, C)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.numel()
        total_correct += (logits.argmax(-1) == labels).sum().item()
        total_points += labels.numel()
    return total_loss / total_points, total_correct / total_points


@torch.no_grad()
def validate(model, loader, device, num_classes=NUM_CLASSES):
    model.eval()
    total_loss, total_correct, total_points = 0.0, 0, 0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for points, labels in loader:
        points, labels = points.to(device), labels.to(device)
        logits = model(points)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        )
        preds = logits.argmax(-1)
        total_loss += loss.item() * labels.numel()
        total_correct += (preds == labels).sum().item()
        total_points += labels.numel()
        cm += confusion_matrix(preds.reshape(-1).cpu(), labels.reshape(-1).cpu(), num_classes)
    acc = total_correct / total_points
    miou, per_class = miou_from_cm(cm)
    return total_loss / total_points, acc, miou, per_class


def main():
    args = get_args()
    device = args.device
    print(f"Device: {device} | Model: {args.model} | num_points: {args.num_points}")

    train_loader, val_loader = build_dataloaders(args)
    model = build_model(args.model, num_classes=NUM_CLASSES, extra_channels=1).to(device)  # +1 for intensity
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_miou, per_class_iou = validate(model, val_loader, device)
        print(
            f"Epoch {epoch:3d} | train_loss {train_loss:.4f} acc {train_acc:.4f} "
            f"| val_loss {val_loss:.4f} acc {val_acc:.4f} mIoU {val_miou:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), args.out)
    print(f"\nSaved best checkpoint (val acc {best_acc:.4f}) to {args.out}")

    print("Per-class IoU:")
    for name, v in zip(CLASS_NAMES, per_class_iou):
        print(f"  {name:12s} {v:.4f}")

    # --- Quantization + memory/accuracy benchmark ---
    sample_points, _ = next(iter(val_loader))
    results = benchmark_fp32_vs_int8(model, val_loader, sample_points)
    print_benchmark_report(results)


if __name__ == "__main__":
    main()
