import os
from collections import Counter
import SimpleITK as sitk
import numpy as np


DATA_ROOTS = {
    "cor": r"D:\2PlacentaMRcor",
    "sag": r"D:\2PlacentaMRsag",
    "tra": r"D:\2PlacentaMRtra",
}

NUM_DIRECTION_ERRORS = 10
NUM_VALID_CASES = 3

SPECIAL_CASES = [
    "147lixiugui",
    "36liyan",
    "210wangjing",
    "gaohuiting",
]


def direction_key(direction):
    return tuple(round(float(x), 4) for x in direction)


def same(a, b):
    return np.allclose(a, b)


def inspect_case(direction, case_name):

    root = DATA_ROOTS[direction]
    case_dir = os.path.join(root, case_name)

    image_path = os.path.join(
        case_dir,
        f"{case_name}_SE_{direction}.nii.gz"
    )

    mask_path = os.path.join(
        case_dir,
        f"{case_name}_SE_{direction}_placenta_mask.nii.gz"
    )

    if not os.path.isfile(image_path):
        return {
            "status": "missing_image",
            "case": case_name,
        }

    if not os.path.isfile(mask_path):
        return {
            "status": "missing_mask",
            "case": case_name,
        }

    try:
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)

        image_direction = image.GetDirection()
        mask_direction = mask.GetDirection()

        return {
            "status": "ok",
            "case": case_name,

            "size_same":
                image.GetSize() == mask.GetSize(),

            "spacing_same":
                same(
                    image.GetSpacing(),
                    mask.GetSpacing()
                ),

            "origin_same":
                same(
                    image.GetOrigin(),
                    mask.GetOrigin()
                ),

            "direction_same":
                same(
                    image_direction,
                    mask_direction
                ),

            "image_direction":
                direction_key(image_direction),

            "mask_direction":
                direction_key(mask_direction),
        }

    except Exception as e:

        return {
            "status": "error",
            "case": case_name,
            "error": str(e),
        }


def collect_cases(direction):

    root = DATA_ROOTS[direction]

    valid = []
    errors = []

    case_dirs = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])

    for case_name in case_dirs:

        result = inspect_case(direction, case_name)

        if result["status"] != "ok":
            continue

        if result["direction_same"]:
            valid.append(result)
        else:
            errors.append(result)

    return valid, errors


def print_direction_patterns(results, title):

    print(f"\n{title}")

    image_patterns = Counter(
        r["image_direction"]
        for r in results
    )

    mask_patterns = Counter(
        r["mask_direction"]
        for r in results
    )

    print("Image Direction patterns:")

    for pattern, count in image_patterns.items():
        print(f"  {count} cases: {pattern}")

    print("Mask Direction patterns:")

    for pattern, count in mask_patterns.items():
        print(f"  {count} cases: {pattern}")


def main():

    print("=" * 80)
    print("NIFTI DIRECTION SUMMARY")
    print("=" * 80)

    for direction in DATA_ROOTS:

        valid, errors = collect_cases(direction)

        print("\n" + "#" * 80)
        print(f"# {direction.upper()}")
        print("#" * 80)

        print(f"Direction SAME cases:  {len(valid)}")
        print(f"Direction ERROR cases: {len(errors)}")

        # Check what other geometry differences exist
        all_results = valid + errors

        size_errors = sum(
            not r["size_same"]
            for r in all_results
        )

        spacing_errors = sum(
            not r["spacing_same"]
            for r in all_results
        )

        origin_errors = sum(
            not r["origin_same"]
            for r in all_results
        )

        print("\nOther geometry differences:")
        print(f"  Size:    {size_errors}")
        print(f"  Spacing: {spacing_errors}")
        print(f"  Origin:   {origin_errors}")

        # Direction patterns among abnormal cases
        selected_errors = errors[:NUM_DIRECTION_ERRORS]

        print_direction_patterns(
            selected_errors,
            f"First {len(selected_errors)} Direction ERROR cases"
        )

        # Direction patterns among normal cases
        selected_valid = valid[:NUM_VALID_CASES]

        print_direction_patterns(
            selected_valid,
            f"First {len(selected_valid)} Direction SAME cases"
        )

        # Special cases
        print("\nSpecial cases:")

        for case in SPECIAL_CASES:

            result = inspect_case(
                direction,
                case
            )

            if result["status"] != "ok":
                print(
                    f"  {case}: "
                    f"{result['status']}"
                )
                continue

            print(
                f"  {case}: "
                f"Direction "
                f"{'SAME' if result['direction_same'] else 'DIFFERENT'}"
            )

            print(
                f"      Size "
                f"{'SAME' if result['size_same'] else 'DIFFERENT'}, "
                f"Spacing "
                f"{'SAME' if result['spacing_same'] else 'DIFFERENT'}, "
                f"Origin "
                f"{'SAME' if result['origin_same'] else 'DIFFERENT'}"
            )

        # Compare image direction and mask direction
        # in the selected abnormal cases
        print("\nDetailed abnormal-case comparison:")

        for result in selected_errors:

            print(
                f"  {result['case']}:"
            )

            print(
                f"    Image: {result['image_direction']}"
            )

            print(
                f"    Mask:  {result['mask_direction']}"
            )


if __name__ == "__main__":
    main()