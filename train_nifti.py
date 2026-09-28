import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nifti_dataset import NiftiSliceDataset
from unet import UNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_CSV = r"D:\DL\Pytorch-UNet\cor_train.csv"
VAL_CSV = r"D:\DL\Pytorch-UNet\cor_val.csv"

BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 1e-4
TARGET_SIZE = 512

CHECKPOINT_DIR = r"D:\DL\Pytorch-UNet\checkpoints"
BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "baseline_unet_cor_best.pth"
)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)[:, 1]
        targets = (targets == 1).float()

        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )

        return 1.0 - dice.mean()


def dice_score(logits, targets):
    pred = torch.argmax(logits, dim=1)

    pred = (pred == 1).float()
    target = (targets == 1).float()

    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)

    dice = (
        2.0 * intersection + 1e-6
    ) / (
        pred.sum(dim=1) + target.sum(dim=1) + 1e-6
    )

    return dice.mean().item()


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading training dataset...")
    train_dataset = NiftiSliceDataset(
        TRAIN_CSV,
        target_size=TARGET_SIZE
    )

    print("\nLoading validation dataset...")
    val_dataset = NiftiSliceDataset(
        VAL_CSV,
        target_size=TARGET_SIZE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print("\nCreating baseline U-Net...")

    model = UNet(
        n_channels=1,
        n_classes=2,
        bilinear=False
    ).to(DEVICE)

    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()

    best_dice = 0.0

    print("\nStarting training:")
    print("Epochs       :", EPOCHS)
    print("Batch size   :", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Input size   :", TARGET_SIZE)
    print("Input channel: 1")
    print("Output class : 2")
    print()

    for epoch in range(EPOCHS):
        model.train()

        train_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}"
        )

        for images, masks, _, _ in progress:
            images = images.to(
                DEVICE,
                dtype=torch.float32
            )

            masks = masks.to(
                DEVICE,
                dtype=torch.long
            )

            optimizer.zero_grad()

            logits = model(images)

            loss_ce = ce_loss(logits, masks)
            loss_dice = dice_loss(logits, masks)

            loss = loss_ce + loss_dice

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss /= len(train_loader)

        model.eval()

        val_loss = 0.0
        val_dice = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, masks, _, _ in tqdm(
                val_loader,
                desc="Validation"
            ):
                images = images.to(
                    DEVICE,
                    dtype=torch.float32
                )

                masks = masks.to(
                    DEVICE,
                    dtype=torch.long
                )

                logits = model(images)

                loss_ce = ce_loss(logits, masks)
                loss_dice = dice_loss(logits, masks)

                loss = loss_ce + loss_dice

                val_loss += loss.item()
                val_dice += dice_score(
                    logits,
                    masks
                )

                val_batches += 1

        val_loss /= val_batches
        val_dice /= val_batches

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )
        print(
            f"Train Loss: {train_loss:.6f}"
        )
        print(
            f"Val Loss  : {val_loss:.6f}"
        )
        print(
            f"Val Dice  : {val_dice:.6f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                },
                BEST_MODEL_PATH
            )

            print(
                f"Saved best model: {BEST_MODEL_PATH}"
            )

    print()
    print("Training finished.")
    print(f"Best validation Dice: {best_dice:.6f}")


if __name__ == "__main__":
    main()
