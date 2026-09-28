import os
import csv
import numpy as np
import SimpleITK as sitk


DATA_ROOT = r"D:\2PlacentaMRcor"
OUTPUT_CSV = r"D:\DL\Pytorch-UNet\cor_cases.csv"


def find_files(case_dir):
    nii_files = [
        f for f in os.listdir(case_dir)
        if f.lower().endswith(".nii.gz")
    ]

    image_files = [
        f for f in nii_files
        if "mask" not in f.lower()
    ]

    mask_files = [
        f for f in nii_files
        if "mask" in f.lower()
    ]

    return image_files, mask_files


def check_case(case_dir):
    case_name = os.path.basename(case_dir)

    image_files, mask_files = find_files(case_dir)

    reasons = []

    # Check image file
    if len(image_files) == 0:
        reasons.append("missing_image")
        image_path = ""
    elif len(image_files) > 1:
        reasons.append("multiple_images")
        image_path = ""
    else:
        image_path = os.path.join(case_dir, image_files[0])

    # Check mask file
    if len(mask_files) == 0:
        reasons.append("missing_mask")
        mask_path = ""
    elif len(mask_files) > 1:
        reasons.append("multiple_masks")
        mask_path = ""
    else:
        mask_path = os.path.join(case_dir, mask_files[0])

    # Cannot continue without exactly one image and one mask
    if reasons:
        return {
            "case": case_name,
            "image_path": image_path,
            "mask_path": mask_path,
            "direction_max_diff": "",
            "status": "invalid",
            "reason": ";".join(reasons)
        }

    # Read image
    try:
        image = sitk.ReadImage(image_path)
    except Exception as e:
        return {
            "case": case_name,
            "image_path": image_path,
            "mask_path": mask_path,
            "direction_max_diff": "",
            "status": "invalid",
            "reason": "image_read_error"
        }

    # Read mask
    try:
        mask = sitk.ReadImage(mask_path)
    except Exception as e:
        return {
            "case": case_name,
            "image_path": image_path,
            "mask_path": mask_path,
            "direction_max_diff": "",
            "status": "invalid",
            "reason": "mask_read_error"
        }

    # Size must be exactly the same
    if image.GetSize() != mask.GetSize():
        reasons.append("size_mismatch")

    # Spacing should be the same within numerical tolerance
    image_spacing = np.array(image.GetSpacing(), dtype=np.float64)
    mask_spacing = np.array(mask.GetSpacing(), dtype=np.float64)

    if not np.allclose(
        image_spacing,
        mask_spacing,
        rtol=0.0,
        atol=1e-6
    ):
        reasons.append("spacing_mismatch")

    # Origin should be the same within numerical tolerance
    image_origin = np.array(image.GetOrigin(), dtype=np.float64)
    mask_origin = np.array(mask.GetOrigin(), dtype=np.float64)

    if not np.allclose(
        image_origin,
        mask_origin,
        rtol=0.0,
        atol=1e-5
    ):
        reasons.append("origin_mismatch")

    # Check mask values
    try:
        mask_array = sitk.GetArrayFromImage(mask)
        unique_values = np.unique(mask_array)

        # Empty mask
        if not np.any(mask_array > 0):
            reasons.append("empty_mask")

        # Mask should contain only 0 and 1
        if not np.all(np.isin(unique_values, [0, 1])):
            reasons.append("non_binary_mask")

    except Exception as e:
        reasons.append("mask_array_error")

    # Direction is recorded but NOT used as a hard exclusion criterion
    image_direction = np.array(
        image.GetDirection(),
        dtype=np.float64
    ).reshape(3, 3)

    mask_direction = np.array(
        mask.GetDirection(),
        dtype=np.float64
    ).reshape(3, 3)

    direction_max_diff = float(
        np.max(np.abs(image_direction - mask_direction))
    )

    if len(reasons) == 0:
        status = "valid"
        reason = ""
    else:
        status = "invalid"
        reason = ";".join(reasons)

    return {
        "case": case_name,
        "image_path": image_path,
        "mask_path": mask_path,
        "direction_max_diff": direction_max_diff,
        "status": status,
        "reason": reason
    }


def main():
    case_dirs = []

    for name in sorted(os.listdir(DATA_ROOT)):
        path = os.path.join(DATA_ROOT, name)

        if os.path.isdir(path):
            case_dirs.append(path)

    print("=" * 70)
    print("Final COR NIfTI QC")
    print("=" * 70)
    print("Data root:", DATA_ROOT)
    print("Total cases:", len(case_dirs))
    print()

    results = []

    for i, case_dir in enumerate(case_dirs, 1):
        result = check_case(case_dir)
        results.append(result)

        if result["status"] == "valid":
            print(
                "[VALID] {:>3}/{} {}".format(
                    i,
                    len(case_dirs),
                    result["case"]
                )
            )
        else:
            print(
                "[INVALID] {:>3}/{} {} -> {}".format(
                    i,
                    len(case_dirs),
                    result["case"],
                    result["reason"]
                )
            )

    # Save ALL cases and their QC status
    fieldnames = [
        "case",
        "image_path",
        "mask_path",
        "direction_max_diff",
        "status",
        "reason"
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    valid_cases = [
        r for r in results
        if r["status"] == "valid"
    ]

    invalid_cases = [
        r for r in results
        if r["status"] == "invalid"
    ]

    print()
    print("=" * 70)
    print("QC SUMMARY")
    print("=" * 70)
    print("Total cases :", len(results))
    print("Valid cases :", len(valid_cases))
    print("Invalid cases:", len(invalid_cases))
    print()

    if invalid_cases:
        print("Invalid cases:")
        for r in invalid_cases:
            print(
                "  {} -> {}".format(
                    r["case"],
                    r["reason"]
                )
            )

    print()
    print("CSV saved to:")
    print(OUTPUT_CSV)
    print("=" * 70)


if __name__ == "__main__":
    main()