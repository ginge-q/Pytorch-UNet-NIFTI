import csv
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from unet.attention_unet import AttentionUNet


# ============================================================
# 配置
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_CSV = r"D:\DL\Pytorch-UNet\cor_test.csv"

CHECKPOINT_PATH = (
    r"D:\DL\Pytorch-UNet\checkpoints\attention_unet_cor_best.pth"
)

OUTPUT_DIR = Path(
    r"D:\DL\Pytorch-UNet\predictions\attention_cor"
)

TARGET_SIZE = 512


# ============================================================
# 读取 Test CSV
# ============================================================

def load_test_cases(csv_path):
    cases = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            cases.append({
                "case": row["case"],
                "image": row["image_path"],
                "mask": row["mask_path"],
            })

    return cases


# ============================================================
# 单个 slice 预处理
# ============================================================

def preprocess_slice(image_slice):

    image_slice = image_slice.astype(np.float32)

    # 与训练时保持完全一致：
    # 每个 slice 单独做 z-score
    mean = image_slice.mean()
    std = image_slice.std()

    if std > 1e-8:
        image_slice = (image_slice - mean) / std
    else:
        image_slice = image_slice - mean

    image_tensor = torch.from_numpy(image_slice).unsqueeze(0)

    # resize 到 512 × 512
    if image_tensor.shape[-2:] != (TARGET_SIZE, TARGET_SIZE):

        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(TARGET_SIZE, TARGET_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return image_tensor


# ============================================================
# 主程序
# ============================================================

def main():

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------------
    # 创建 Attention U-Net
    # --------------------------------------------------------

    model = AttentionUNet(
        n_channels=1,
        n_classes=2,
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # 加载最佳 checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Loaded model:", CHECKPOINT_PATH)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Checkpoint val Dice:", checkpoint.get("val_dice"))

    # --------------------------------------------------------
    # 输出目录
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Test cases
    # --------------------------------------------------------

    cases = load_test_cases(TEST_CSV)

    print("Test patients:", len(cases))
    print("Output directory:", OUTPUT_DIR)

    # --------------------------------------------------------
    # 逐患者预测
    # --------------------------------------------------------

    with torch.no_grad():

        for case_info in tqdm(
            cases,
            desc="Predicting patients",
        ):

            case_name = case_info["case"]
            image_path = case_info["image"]

            # 读取原始 NIfTI
            image_itk = sitk.ReadImage(image_path)

            image_array = sitk.GetArrayFromImage(
                image_itk
            )

            # image_array:
            # [Z, H, W]

            predicted_slices = []

            # ------------------------------------------------
            # 逐 slice
            # ------------------------------------------------

            for z in range(image_array.shape[0]):

                image_slice = image_array[z]

                image_tensor = preprocess_slice(
                    image_slice
                )

                # [1, 1, H, W]
                image_tensor = image_tensor.unsqueeze(0)

                image_tensor = image_tensor.to(DEVICE)

                # [1, 2, 512, 512]
                logits = model(image_tensor)

                # [512, 512]
                prediction = torch.argmax(
                    logits,
                    dim=1,
                )[0]

                prediction = (
                    prediction
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                # ------------------------------------------------
                # resize 回原始尺寸
                # ------------------------------------------------

                if prediction.shape != image_slice.shape:

                    prediction_tensor = torch.from_numpy(
                        prediction
                    ).float()

                    prediction_tensor = F.interpolate(
                        prediction_tensor.unsqueeze(0).unsqueeze(0),
                        size=image_slice.shape,
                        mode="nearest",
                    )

                    prediction = (
                        prediction_tensor[0, 0]
                        .numpy()
                        .astype(np.uint8)
                    )

                predicted_slices.append(
                    prediction
                )

            # ------------------------------------------------
            # 重新组成 3D volume
            # ------------------------------------------------

            prediction_volume = np.stack(
                predicted_slices,
                axis=0,
            )

            # ------------------------------------------------
            # 转回 SimpleITK
            # ------------------------------------------------

            prediction_itk = sitk.GetImageFromArray(
                prediction_volume
            )

            # 保留原始图像空间信息
            prediction_itk.CopyInformation(
                image_itk
            )

            # ------------------------------------------------
            # 保存 NIfTI
            # ------------------------------------------------

            output_path = (
                OUTPUT_DIR
                / f"{case_name}_pred.nii.gz"
            )

            sitk.WriteImage(
                prediction_itk,
                str(output_path),
            )

    print()
    print("Prediction finished.")
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
