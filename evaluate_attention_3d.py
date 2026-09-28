import csv
from pathlib import Path

import SimpleITK as sitk
import numpy as np


# ============================================================
# 配置
# ============================================================

TEST_CSV = r"D:\DL\Pytorch-UNet\cor_test.csv"

PRED_DIR = Path(
    r"D:\DL\Pytorch-UNet\predictions\attention_cor"
)

OUTPUT_CSV = Path(
    r"D:\DL\Pytorch-UNet\attention_test_metrics.csv"
)


# ============================================================
# 计算指标
# ============================================================

def calculate_metrics(pred, gt):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    # Dice
    denominator = 2 * tp + fp + fn

    if denominator == 0:
        dice = 1.0
    else:
        dice = 2 * tp / denominator

    # IoU
    union = tp + fp + fn

    if union == 0:
        iou = 1.0
    else:
        iou = tp / union

    # Precision
    precision_denominator = tp + fp

    if precision_denominator == 0:
        precision = 1.0 if (tp + fn) == 0 else 0.0
    else:
        precision = tp / precision_denominator

    # Recall
    recall_denominator = tp + fn

    if recall_denominator == 0:
        recall = 1.0
    else:
        recall = tp / recall_denominator

    return dice, iou, precision, recall


# ============================================================
# 读取 Test CSV
# ============================================================

def load_test_cases(csv_path):

    cases = []

    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            cases.append({
                "case": row["case"],
                "mask": row["mask_path"],
            })

    return cases


# ============================================================
# 主程序
# ============================================================

def main():

    cases = load_test_cases(TEST_CSV)

    print("Attention U-Net 3D Test Evaluation")
    print("=" * 60)
    print("Number of patients :", len(cases))
    print("Prediction directory:", PRED_DIR)
    print()

    results = []

    for case_info in cases:

        case_name = case_info["case"]
        mask_path = case_info["mask"]

        pred_path = (
            PRED_DIR
            / f"{case_name}_pred.nii.gz"
        )

        if not pred_path.exists():

            print(
                f"[WARNING] Prediction not found: "
                f"{case_name}"
            )

            continue

        # ----------------------------------------------------
        # 读取 GT
        # ----------------------------------------------------

        gt_itk = sitk.ReadImage(mask_path)

        gt = sitk.GetArrayFromImage(
            gt_itk
        )

        # ----------------------------------------------------
        # 读取 prediction
        # ----------------------------------------------------

        pred_itk = sitk.ReadImage(
            str(pred_path)
        )

        pred = sitk.GetArrayFromImage(
            pred_itk
        )

        # ----------------------------------------------------
        # 检查 shape
        # ----------------------------------------------------

        if gt.shape != pred.shape:

            raise ValueError(
                f"Shape mismatch for {case_name}: "
                f"GT={gt.shape}, "
                f"Pred={pred.shape}"
            )

        # ----------------------------------------------------
        # 二值化
        # ----------------------------------------------------

        gt = gt > 0
        pred = pred > 0

        # ----------------------------------------------------
        # 计算指标
        # ----------------------------------------------------

        dice, iou, precision, recall = (
            calculate_metrics(pred, gt)
        )

        results.append({
            "case": case_name,
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
        })

        print(
            f"{case_name:30s} "
            f"Dice={dice:.4f}  "
            f"IoU={iou:.4f}  "
            f"Precision={precision:.4f}  "
            f"Recall={recall:.4f}"
        )

    # ========================================================
    # 汇总
    # ========================================================

    if len(results) == 0:

        raise RuntimeError(
            "No valid prediction results found."
        )

    dice_values = np.array(
        [x["dice"] for x in results]
    )

    iou_values = np.array(
        [x["iou"] for x in results]
    )

    precision_values = np.array(
        [x["precision"] for x in results]
    )

    recall_values = np.array(
        [x["recall"] for x in results]
    )

    print()
    print("=" * 60)
    print("Patient-level 3D Test Results")
    print("=" * 60)

    print(
        f"Number of patients : {len(results)}"
    )

    print(
        f"\nMean Dice          : "
        f"{dice_values.mean():.6f}"
    )

    print(
        f"Std Dice           : "
        f"{dice_values.std():.6f}"
    )

    print(
        f"Median Dice        : "
        f"{np.median(dice_values):.6f}"
    )

    print(
        f"\nMean IoU           : "
        f"{iou_values.mean():.6f}"
    )

    print(
        f"Std IoU            : "
        f"{iou_values.std():.6f}"
    )

    print(
        f"\nMean Precision     : "
        f"{precision_values.mean():.6f}"
    )

    print(
        f"Std Precision      : "
        f"{precision_values.std():.6f}"
    )

    print(
        f"\nMean Recall        : "
        f"{recall_values.mean():.6f}"
    )

    print(
        f"Std Recall         : "
        f"{recall_values.std():.6f}"
    )

    # ========================================================
    # 保存 CSV
    # ========================================================

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "case",
            "dice",
            "iou",
            "precision",
            "recall",
        ])

        for x in results:

            writer.writerow([
                x["case"],
                f"{x['dice']:.6f}",
                f"{x['iou']:.6f}",
                f"{x['precision']:.6f}",
                f"{x['recall']:.6f}",
            ])

    print()
    print("Saved detailed results to:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
