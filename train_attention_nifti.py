import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nifti_dataset import NiftiSliceDataset
from unet.attention_unet import AttentionUNet


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TRAIN_CSV = r"D:\DL\Pytorch-UNet\cor_train.csv"
VAL_CSV = r"D:\DL\Pytorch-UNet\cor_val.csv"

BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 1e-4
TARGET_SIZE = 512

CHECKPOINT_DIR = r"D:\DL\Pytorch-UNet\checkpoints"
BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "attention_unet_cor_best.pth"
)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)[:, 1]

        targets = targets.float()

        intersection = (probs * targets).sum(
            dim=(1, 2)
        )

        denominator = (
            probs.sum(dim=(1, 2))
            + targets.sum(dim=(1, 2))
        )

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
        )

        return 1.0 - dice.mean()


def dice_score(logits, targets):
    preds = torch.argmax(
        logits,
        dim=1
    )

    preds = preds.bool()
    targets = targets.bool()

    intersection = (
        preds & targets
    ).sum(dim=(1, 2)).float()

    denominator = (
        preds.sum(dim=(1, 2)).float()
        + targets.sum(dim=(1, 2)).float()
    )

    dice = torch.where(
        denominator > 0,
        (2.0 * intersection + 1e-6)
        / (denominator + 1e-6),
        torch.ones_like(denominator)
    )

    return dice.mean().item()


def train_one_epoch(
    model,
    loader,
    optimizer,
    ce_loss,
    dice_loss,
    epoch
):
    model.train()

    running_loss = 0.0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch} Train"
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

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)

        loss_ce = ce_loss(
            logits,
            masks
        )

        loss_dice = dice_loss(
            logits,
            masks
        )

        loss = loss_ce + loss_dice

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return running_loss / len(loader)


@torch.no_grad()
def validate(
    model,
    loader,
    ce_loss,
    dice_loss
):
    model.eval()

    running_loss = 0.0
    running_dice = 0.0

    progress = tqdm(
        loader,
        desc="Validation"
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

        logits = model(images)

        loss_ce = ce_loss(
            logits,
            masks
        )

        loss_dice = dice_loss(
            logits,
            masks
        )

        loss = loss_ce + loss_dice

        dice = dice_score(
            logits,
            masks
        )

        running_loss += loss.item()
        running_dice += dice

        progress.set_postfix(
            dice=f"{dice:.4f}"
        )

    avg_loss = (
        running_loss / len(loader)
    )

    avg_dice = (
        running_dice / len(loader)
    )

    return avg_loss, avg_dice


def main():

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()
    print("Loading datasets...")

    train_dataset = NiftiSliceDataset(
        TRAIN_CSV,
        target_size=TARGET_SIZE
    )

    val_dataset = NiftiSliceDataset(
        VAL_CSV,
        target_size=TARGET_SIZE
    )

    print(
        "Train slices:",
        len(train_dataset)
    )

    print(
        "Val slices:",
        len(val_dataset)
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

    model = AttentionUNet(
        n_channels=1,
        n_classes=2
    ).to(DEVICE)

    print()
    print(
        "Trainable parameters:",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    ce_loss = nn.CrossEntropyLoss()

    dice_loss = DiceLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_dice = -1.0

    print()
    print("Starting Attention U-Net training:")
    print(
        f"Epochs:          {EPOCHS}"
    )
    print(
        f"Batch size:      {BATCH_SIZE}"
    )
    print(
        f"Learning rate:   {LEARNING_RATE}"
    )
    print(
        f"Target size:     {TARGET_SIZE}"
    )
    print(
        f"Checkpoint:      {BEST_MODEL_PATH}"
    )
    print()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            ce_loss,
            dice_loss,
            epoch
        )

        val_loss, val_dice = validate(
            model,
            val_loader,
            ce_loss,
            dice_loss
        )

        print()
        print(
            f"Train Loss: {train_loss:.6f}"
        )

        print(
            f"Val Loss  : {val_loss:.6f}"
        )

        print(
            f"Val Dice  : {val_dice:.6f}"
        )

        if val_dice > best_val_dice:

            best_val_dice = val_dice

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_loss": val_loss,
                },
                BEST_MODEL_PATH
            )

            print(
                "Saved best model:",
                BEST_MODEL_PATH
            )

        print()
        print("-" * 70)

    print()
    print("Training finished.")

    print(
        f"Best validation Dice: "
        f"{best_val_dice:.6f}"
    )

    print(
        "Best model:",
        BEST_MODEL_PATH
    )


if __name__ == "__main__":
    main()
