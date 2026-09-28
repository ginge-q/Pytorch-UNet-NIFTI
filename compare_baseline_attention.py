import csv
from pathlib import Path
import numpy as np

BASELINE_CSV = Path(r"D:\DL\Pytorch-UNet\baseline_test_metrics.csv")
ATTENTION_CSV = Path(r"D:\DL\Pytorch-UNet\attention_test_metrics.csv")

OUTPUT_CSV = Path(r"D:\DL\Pytorch-UNet\baseline_vs_attention.csv")


def load_csv(path):
    results = {}

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            results[row["case"]] = {
                "dice": float(row["dice"]),
                "iou": float(row["iou"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
            }

    return results


def main():

    baseline = load_csv(BASELINE_CSV)
    attention = load_csv(ATTENTION_CSV)

    common_cases = sorted(
        set(baseline.keys()) & set(attention.keys())
    )

    print("=" * 70)
    print("Baseline vs Attention U-Net")
    print("=" * 70)
    print("Common patients:", len(common_cases))
    print()

    results = []

    for case in common_cases:

        b = baseline[case]
        a = attention[case]

        dice_diff = a["dice"] - b["dice"]
        iou_diff = a["iou"] - b["iou"]
        precision_diff = a["precision"] - b["precision"]
        recall_diff = a["recall"] - b["recall"]

        results.append({
            "case": case,
            "baseline_dice": b["dice"],
            "attention_dice": a["dice"],
            "dice_diff": dice_diff,
            "baseline_iou": b["iou"],
            "attention_iou": a["iou"],
            "iou_diff": iou_diff,
            "baseline_precision": b["precision"],
            "attention_precision": a["precision"],
            "precision_diff": precision_diff,
            "baseline_recall": b["recall"],
            "attention_recall": a["recall"],
            "recall_diff": recall_diff,
        })

    # 按 Dice 提升幅度排序
    results.sort(
        key=lambda x: x["dice_diff"],
        reverse=True
    )

    print(
        f"{'Case':30s}"
        f"{'Baseline':>12s}"
        f"{'Attention':>12s}"
        f"{'Delta':>12s}"
    )

    print("-" * 70)

    for x in results:

        print(
            f"{x['case']:30s}"
            f"{x['baseline_dice']:12.4f}"
            f"{x['attention_dice']:12.4f}"
            f"{x['dice_diff']:12.4f}"
        )

    dice_diff = np.array(
        [x["dice_diff"] for x in results]
    )

    print()
    print("=" * 70)
    print("Paired Dice Comparison")
    print("=" * 70)

    print(f"Mean Delta Dice   : {dice_diff.mean():.6f}")
    print(f"Median Delta Dice : {np.median(dice_diff):.6f}")
    print(f"Std Delta Dice    : {dice_diff.std():.6f}")

    print(
        f"Improved cases    : "
        f"{np.sum(dice_diff > 0)}"
    )

    print(
        f"Decreased cases   : "
        f"{np.sum(dice_diff < 0)}"
    )

    print(
        f"Unchanged cases   : "
        f"{np.sum(dice_diff == 0)}"
    )

    print(
        f"Max improvement   : "
        f"{dice_diff.max():.6f}"
    )

    print(
        f"Max decrease      : "
        f"{dice_diff.min():.6f}"
    )

    # 保存
    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "case",
            "baseline_dice",
            "attention_dice",
            "dice_diff",
            "baseline_iou",
            "attention_iou",
            "iou_diff",
            "baseline_precision",
            "attention_precision",
            "precision_diff",
            "baseline_recall",
            "attention_recall",
            "recall_diff",
        ])

        for x in results:

            writer.writerow([
                x["case"],
                f"{x['baseline_dice']:.6f}",
                f"{x['attention_dice']:.6f}",
                f"{x['dice_diff']:.6f}",
                f"{x['baseline_iou']:.6f}",
                f"{x['attention_iou']:.6f}",
                f"{x['iou_diff']:.6f}",
                f"{x['baseline_precision']:.6f}",
                f"{x['attention_precision']:.6f}",
                f"{x['precision_diff']:.6f}",
                f"{x['baseline_recall']:.6f}",
                f"{x['attention_recall']:.6f}",
                f"{x['recall_diff']:.6f}",
            ])

    print()
    print("Saved:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
