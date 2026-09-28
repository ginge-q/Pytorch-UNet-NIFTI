import os
import csv
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from tqdm import tqdm

from unet import UNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_CSV = r"D:\DL\Pytorch-UNet\cor_test.csv"

MODEL_PATH = r"D:\DL\Pytorch-UNet\checkpoints\baseline_unet_cor_best.pth"

OUTPUT_DIR = r"D:\DL\Pytorch-UNet\predictions\baseline_cor"

TARGET_SIZE = 512


def normalize_slice(image_slice):
    image_slice = image_slice.astype(np.float32)

    mean = image_slice.mean()
    std = image_slice.std()

    if std > 1e-8:
        image_slice = (image_slice - mean) / std
    else:
        image_slice = image_slice - mean

    return image_slice


def preprocess_slice(image_slice):
    image_slice = normalize_slice(image_slice)

    tensor = torch.from_numpy(image_slice).unsqueeze(0).unsqueeze(0)

    tensor = F.interpolate(
        tensor,
        size=(TARGET_SIZE, TARGET_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    return tensor


def restore_mask(mask_512, original_height, original_width):
    tensor = torch.from_numpy(
        mask_512.astype(np.float32)
    ).unsqueeze(0).unsqueeze(0)

    tensor = F.interpolate(
        tensor,
        size=(original_height, original_width),
        mode="nearest",
    )

    return tensor.squeeze().numpy().astype(np.uint8)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = UNet(
        n_channels=1,
        n_classes=2,
        bilinear=False,
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Loaded model:", MODEL_PATH)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Checkpoint val Dice:", checkpoint.get("val_dice"))

    with open(
        TEST_CSV,
        "r",
        encoding="utf-8-sig"
    ) as f:
        rows = list(csv.DictReader(f))

    print("Test patients:", len(rows))
    print("Output directory:", OUTPUT_DIR)

    for row in tqdm(rows, desc="Predicting patients"):
        case = row["case"]
        image_path = row["image_path"]

        image_itk = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image_itk)

        z_count, original_height, original_width = image.shape

        prediction = np.zeros(
            (z_count, original_height, original_width),
            dtype=np.uint8,
        )

        for z in range(z_count):
            input_tensor = preprocess_slice(
                image[z]
            ).to(
                DEVICE,
                dtype=torch.float32,
            )

            with torch.no_grad():
                logits = model(input_tensor)
                pred = torch.argmax(
                    logits,
                    dim=1
                )

            pred_512 = pred[0].cpu().numpy()

            pred_original = restore_mask(
                pred_512,
                original_height,
                original_width,
            )

            prediction[z] = pred_original

        prediction_itk = sitk.GetImageFromArray(
            prediction
        )

        prediction_itk.CopyInformation(
            image_itk
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{case}_pred.nii.gz"
        )

        sitk.WriteImage(
            prediction_itk,
            output_path,
        )

    print()
    print("Prediction finished.")
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
