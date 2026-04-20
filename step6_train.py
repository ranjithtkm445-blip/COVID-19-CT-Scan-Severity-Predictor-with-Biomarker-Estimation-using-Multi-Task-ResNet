# File    : step6_train.py
# Purpose : Train the multi-task CNN model using CT scan images.
#           Uses combined MSE + CrossEntropy loss.
#           Saves best model based on validation loss.
#           Runs for 50 epochs on CPU.

import os
import torch
import torch.optim as optim

from step3_dataset import get_dataloaders
from step5_model   import MultiTaskCT, MultiTaskLoss

SAVE_DIR   = r"D:\COVID\saved_model"
MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pth")
EPOCHS     = 50
BATCH_SIZE = 8
LR         = 0.001
VAL_RATIO  = 0.2


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    total_bm   = 0
    total_sev  = 0

    for images, biomarkers, severities in loader:
        images     = images.to(device)
        biomarkers = biomarkers.to(device)
        severities = severities.to(device)

        optimizer.zero_grad()
        pred_bm, pred_sev         = model(images)
        loss, l_bm, l_sev         = loss_fn(pred_bm, biomarkers, pred_sev, severities)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_bm   += l_bm
        total_sev  += l_sev

    n = len(loader)
    return total_loss / n, total_bm / n, total_sev / n


def val_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_bm   = 0
    total_sev  = 0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for images, biomarkers, severities in loader:
            images     = images.to(device)
            biomarkers = biomarkers.to(device)
            severities = severities.to(device)

            pred_bm, pred_sev     = model(images)
            loss, l_bm, l_sev     = loss_fn(pred_bm, biomarkers, pred_sev, severities)

            total_loss += loss.item()
            total_bm   += l_bm
            total_sev  += l_sev

            preds    = torch.argmax(pred_sev, dim=1)
            correct += (preds == severities).sum().item()
            total   += severities.size(0)

    n        = len(loader)
    accuracy = correct / total * 100
    return total_loss / n, total_bm / n, total_sev / n, accuracy


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cpu")

    print("=" * 60)
    print("  Multi-Task CT Training  -  step6_train.py")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  LR         : {LR}")
    print("=" * 60)

    train_loader, val_loader, bm_means, bm_stds = get_dataloaders(
        val_ratio=VAL_RATIO,
        batch_size=BATCH_SIZE
    )

    model   = MultiTaskCT().to(device)
    loss_fn = MultiTaskLoss().to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=LR, weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    best_epoch    = 0

    print(f"\n  {'Epoch':<7} {'Train Loss':<12} {'Val Loss':<12} {'BM Loss':<10} {'Sev Loss':<10} {'Val Acc'}")
    print(f"  {'-'*6} {'-'*11} {'-'*11} {'-'*9} {'-'*9} {'-'*8}")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_bm, train_sev = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        val_loss, val_bm, val_sev, val_acc = val_one_epoch(
            model, val_loader, loss_fn, device
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "loss_state" : loss_fn.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "bm_means"   : bm_means,
                "bm_stds"    : bm_stds,
                "val_loss"   : best_val_loss,
                "val_acc"    : val_acc,
            }, MODEL_PATH)
            flag = " <-- best"
        else:
            flag = ""

        print(f"  {epoch:<7} {train_loss:<12.4f} {val_loss:<12.4f} {val_bm:<10.4f} {val_sev:<10.4f} {val_acc:>6.1f}%{flag}")

    print("\n" + "=" * 60)
    print(f"  Training complete!")
    print(f"  Best epoch    : {best_epoch}")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Model saved  -> {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()