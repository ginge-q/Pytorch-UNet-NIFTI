import os
import csv
import random


INPUT_CSV = r"D:\DL\Pytorch-UNet\cor_cases.csv"

TRAIN_CSV = r"D:\DL\Pytorch-UNet\cor_train.csv"
VAL_CSV = r"D:\DL\Pytorch-UNet\cor_val.csv"
TEST_CSV = r"D:\DL\Pytorch-UNet\cor_test.csv"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


def read_valid_cases(csv_path):
    cases = []

    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["status"].strip().lower() == "valid":
                cases.append(row)

    return cases


def save_csv(path, cases):
    fieldnames = [
        "case",
        "image_path",
        "mask_path",
        "direction_max_diff",
        "status",
        "reason"
    ]

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(cases)


def main():
    print("=" * 70)
    print("Patient-level COR dataset split")
    print("=" * 70)

    cases = read_valid_cases(INPUT_CSV)

    print("Valid cases:", len(cases))
    print()

    if len(cases) == 0:
        raise RuntimeError("No valid cases found.")

    # Shuffle patients, not slices
    random.seed(RANDOM_SEED)
    random.shuffle(cases)

    total = len(cases)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_cases = cases[:train_count]
    val_cases = cases[train_count:train_count + val_count]
    test_cases = cases[train_count + val_count:]

    # Save
    save_csv(TRAIN_CSV, train_cases)
    save_csv(VAL_CSV, val_cases)
    save_csv(TEST_CSV, test_cases)

    # Check patient overlap
    train_ids = {x["case"] for x in train_cases}
    val_ids = {x["case"] for x in val_cases}
    test_ids = {x["case"] for x in test_cases}

    overlap_train_val = train_ids & val_ids
    overlap_train_test = train_ids & test_ids
    overlap_val_test = val_ids & test_ids

    print("=" * 70)
    print("SPLIT RESULT")
    print("=" * 70)

    print("Total :", total)
    print("Train :", len(train_cases))
    print("Val   :", len(val_cases))
    print("Test  :", len(test_cases))
    print()

    print("Train + Val + Test:",
          len(train_cases) + len(val_cases) + len(test_cases))

    print()
    print("Train/Val overlap :", len(overlap_train_val))
    print("Train/Test overlap:", len(overlap_train_test))
    print("Val/Test overlap  :", len(overlap_val_test))

    print()

    if (
        len(overlap_train_val) == 0
        and len(overlap_train_test) == 0
        and len(overlap_val_test) == 0
        and len(train_cases) + len(val_cases) + len(test_cases) == total
    ):
        print("PASS: Patient-level split is correct.")
    else:
        print("ERROR: Dataset split has a problem.")

    print()
    print("Files:")
    print(TRAIN_CSV)
    print(VAL_CSV)
    print(TEST_CSV)

    print("=" * 70)


if __name__ == "__main__":
    main()