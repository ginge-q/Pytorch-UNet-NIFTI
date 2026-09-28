import csv
from pathlib import Path

import numpy as np


CSV_PATH = Path(
    r"D:\DL\Pytorch-UNet\baseline_vs_attention.csv"
)


def main():

    rows = []

    with open(
        CSV_PATH,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            rows.append({
                "case": row["case"],
                "baseline": float(row["baseline_dice"]),
                "attention": float(row["attention_dice"]),
                "diff": float(row["dice_diff"]),
            })


    # ========================================================
    # 按提升幅度排序
    # ========================================================

    improved = sorted(
        rows,
        key=lambda x: x["diff"],
        reverse=True
    )

    decreased = sorted(
        rows,
        key=lambda x: x["diff"]
    )


    print("=" * 75)
    print("Top 5 Improved Cases")
    print("=" * 75)

    for i, x in enumerate(improved[:5], 1):

        print(
            f"{i}. {x['case']:30s} "
            f"Baseline={x['baseline']:.4f}  "
            f"Attention={x['attention']:.4f}  "
            f"Delta={x['diff']:+.4f}"
        )


    print()
    print("=" * 75)
    print("Top 5 Decreased Cases")
    print("=" * 75)

    for i, x in enumerate(decreased[:5], 1):

        print(
            f"{i}. {x['case']:30s} "
            f"Baseline={x['baseline']:.4f}  "
            f"Attention={x['attention']:.4f}  "
            f"Delta={x['diff']:+.4f}"
        )


    print()
    print("=" * 75)
    print("All Cases Sorted by Delta Dice")
    print("=" * 75)

    for x in improved:

        print(
            f"{x['case']:30s} "
            f"{x['diff']:+.4f}"
        )


if __name__ == "__main__":
    main()
