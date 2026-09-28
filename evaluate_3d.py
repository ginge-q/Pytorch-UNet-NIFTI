import os
import csv
import numpy as np
import SimpleITK as sitk

TEST_CSV = r"D:\DL\Pytorch-UNet\cor_test.csv"
PRED_DIR = r"D:\DL\Pytorch-UNet\predictions\baseline_cor"
OUTPUT_CSV = r"D:\DL\Pytorch-UNet\baseline_test_metrics.csv"


def calculate_metrics(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()

    dice_den = 2 * tp + fp + fn
    iou_den = tp + fp + fn
    precision_den = tp + fp
    recall_den = tp + fn

    dice = (2 * tp / dice_den) if dice_den > 0 else 1.0
    iou = (tp / iou_den) if iou_den > 0 else 1.0
    precision = (tp / precision_den) if precision_den > 0 else 0.0
    recall = (tp / recall_den) if recall_den > 0 else 0.0

    return dice, iou, precision, recall, tp, fp, fn


def main():
    with open(TEST_CSV, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    print("Test patients:", len(rows))
    print("Prediction directory:", PRED_DIR)
    print()

    results = []

    for row in rows:
        case = row["case"]
        mask_path = row["mask_path"]
        pred_path = os.path.join(PRED_DIR, f"{case}_pred.nii.gz")

        if not os.path.exists(pred_path):
            raise FileNotFoundError(
                f"Prediction not found for {case}: {pred_path}"
            )

        gt_itk = sitk.ReadImage(mask_path)
        pred_itk = sitk.ReadImage(pred_path)

        gt = sitk.GetArrayFromImage(gt_itk)
        pred = sitk.GetArrayFromImage(pred_itk)

        if gt.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch for {case}: "
                f"GT={gt.shape}, Pred={pred.shape}"
            )

        if gt_itk.GetSize() != pred_itk.GetSize():
            raise ValueError(
                f"Size mismatch for {case}: "
                f"GT={gt_itk.GetSize()}, Pred={pred_itk.GetSize()}"
            )

        dice, iou, precision, recall, tp, fp, fn = calculate_metrics(
            gt > 0,
            pred > 0
        )

        gt_voxels = int((gt > 0).sum())
        pred_voxels = int((pred > 0).sum())

        results.append({
            "case": case,
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "gt_voxels": gt_voxels,
            "pred_voxels": pred_voxels,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        })

    dice_values = np.array([r["dice"] for r in results])
    iou_values = np.array([r["iou"] for r in results])
    precision_values = np.array([r["precision"] for r in results])
    recall_values = np.array([r["recall"] for r in results])

    print("=" * 70)
    print("Patient-level 3D Test Results")
    print("=" * 70)

    print(f"Number of patients : {len(results)}")

    print()
    print(f"Mean Dice          : {dice_values.mean():.6f}")
    print(f"Std Dice           : {dice_values.std():.6f}")
    print(f"Median Dice        : {np.median(dice_values):.6f}")

    print()
    print(f"Mean IoU           : {iou_values.mean():.6f}")
    print(f"Std IoU            : {iou_values.std():.6f}")

    print()
    print(f"Mean Precision     : {precision_values.mean():.6f}")
    print(f"Std Precision      : {precision_values.std():.6f}")

    print()
    print(f"Mean Recall        : {recall_values.mean():.6f}")
    print(f"Std Recall         : {recall_values.std():.6f}")

    print()
    print("=" * 70)
    print("Per-patient results")
    print("=" * 70)

    for r in results:
        print(
            f"{r['case']:20s} "
            f"Dice={r['dice']:.4f} "
            f"IoU={r['iou']:.4f} "
            f"Precision={r['precision']:.4f} "
            f"Recall={r['recall']:.4f}"
        )

    fieldnames = [
        "case",
        "dice",
        "iou",
        "precision",
        "recall",
        "gt_voxels",
        "pred_voxels",
        "tp",
        "fp",
        "fn",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print()
    print("Saved metrics:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
