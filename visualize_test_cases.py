import os
import csv
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

TEST_CSV = r"D:\DL\Pytorch-UNet\cor_test.csv"
METRICS_CSV = r"D:\DL\Pytorch-UNet\baseline_test_metrics.csv"
PRED_DIR = r"D:\DL\Pytorch-UNet\predictions\baseline_cor"
OUTPUT_DIR = r"D:\DL\Pytorch-UNet\visualizations\baseline_cases"


def load_metrics():
    with open(METRICS_CSV, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        row["dice"] = float(row["dice"])

    return rows


def load_test_info():
    with open(TEST_CSV, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    return {row["case"]: row for row in rows}


def choose_slice(gt):
    positive_counts = (gt > 0).sum(axis=(1, 2))

    positive_indices = np.where(positive_counts > 0)[0]

    if len(positive_indices) == 0:
        return gt.shape[0] // 2

    best_index = positive_indices[
        np.argmax(positive_counts[positive_indices])
    ]

    return int(best_index)


def normalize_for_display(image):
    image = image.astype(np.float32)

    low = np.percentile(image, 1)
    high = np.percentile(image, 99)

    if high > low:
        image = np.clip(image, low, high)
        image = (image - low) / (high - low)
    else:
        image = np.zeros_like(image)

    return image


def create_case_figure(case, dice, image, gt, pred, slice_index, output_path):
    image_slice = normalize_for_display(image[slice_index])
    gt_slice = gt[slice_index] > 0
    pred_slice = pred[slice_index] > 0

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image_slice, cmap="gray")
    axes[0].set_title(
        f"MRI\n{case}\nSlice {slice_index}"
    )
    axes[0].axis("off")

    axes[1].imshow(image_slice, cmap="gray")
    axes[1].imshow(
        gt_slice,
        cmap="Reds",
        alpha=0.45
    )
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(image_slice, cmap="gray")
    axes[2].imshow(
        pred_slice,
        cmap="Blues",
        alpha=0.45
    )
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(image_slice, cmap="gray")

    tp = gt_slice & pred_slice
    fp = (~gt_slice) & pred_slice
    fn = gt_slice & (~pred_slice)

    overlay = np.zeros(
        (gt_slice.shape[0], gt_slice.shape[1], 4),
        dtype=np.float32
    )

    overlay[tp] = [0.0, 1.0, 0.0, 0.60]
    overlay[fp] = [1.0, 0.0, 0.0, 0.60]
    overlay[fn] = [0.0, 0.0, 1.0, 0.60]

    axes[3].imshow(overlay)
    axes[3].set_title(
        "Overlay\n"
        "Green=TP  Red=FP  Blue=FN"
    )
    axes[3].axis("off")

    fig.suptitle(
        f"{case} | Test Dice = {dice:.4f}",
        fontsize=14
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metrics = load_metrics()
    test_info = load_test_info()

    metrics_sorted = sorted(
        metrics,
        key=lambda x: x["dice"]
    )

    n = len(metrics_sorted)

    selected = []

    # Lowest 2
    selected.extend(metrics_sorted[:2])

    # Middle 2
    middle_start = max(0, n // 2 - 1)
    selected.extend(
        metrics_sorted[middle_start:middle_start + 2]
    )

    # Highest 2
    selected.extend(metrics_sorted[-2:])

    # Remove duplicates while preserving order
    unique_selected = []
    seen = set()

    for row in selected:
        if row["case"] not in seen:
            unique_selected.append(row)
            seen.add(row["case"])

    print("=" * 70)
    print("Selected cases for visualization")
    print("=" * 70)

    for row in unique_selected:
        print(
            f"{row['case']:25s} "
            f"Dice = {row['dice']:.4f}"
        )

    print()

    for row in unique_selected:
        case = row["case"]
        dice = row["dice"]

        info = test_info[case]

        image_path = info["image_path"]
        mask_path = info["mask_path"]
        pred_path = os.path.join(
            PRED_DIR,
            f"{case}_pred.nii.gz"
        )

        image_itk = sitk.ReadImage(image_path)
        gt_itk = sitk.ReadImage(mask_path)
        pred_itk = sitk.ReadImage(pred_path)

        image = sitk.GetArrayFromImage(image_itk)
        gt = sitk.GetArrayFromImage(gt_itk)
        pred = sitk.GetArrayFromImage(pred_itk)

        if not (
            image.shape == gt.shape == pred.shape
        ):
            raise ValueError(
                f"Shape mismatch: {case} "
                f"image={image.shape}, "
                f"gt={gt.shape}, "
                f"pred={pred.shape}"
            )

        slice_index = choose_slice(gt)

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{case}_dice_{dice:.4f}_slice_{slice_index}.png"
        )

        create_case_figure(
            case,
            dice,
            image,
            gt,
            pred,
            slice_index,
            output_path
        )

        print(
            f"Saved: {output_path}"
        )

    print()
    print("=" * 70)
    print("Visualization finished.")
    print("Output directory:")
    print(OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
