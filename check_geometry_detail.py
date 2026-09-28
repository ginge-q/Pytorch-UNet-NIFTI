import os
import SimpleITK as sitk
import numpy as np


DATA_ROOTS = {
    "cor": r"D:\2PlacentaMRcor",
    "sag": r"D:\2PlacentaMRsag",
    "tra": r"D:\2PlacentaMRtra",
}


def check_direction(direction, root):

    print("\n" + "=" * 80)
    print(f"Checking {direction.upper()}")
    print(f"Root: {root}")
    print("=" * 80)

    case_dirs = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]

    size_error = 0
    spacing_error = 0
    origin_error = 0
    direction_error = 0
    multiple_error = 0

    valid = 0
    missing_image = 0
    missing_mask = 0

    for case_name in sorted(case_dirs):

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
            print(f"[MISSING IMAGE] {case_name}")
            missing_image += 1
            continue

        if not os.path.isfile(mask_path):
            print(f"[MISSING MASK] {case_name}")
            missing_mask += 1
            continue

        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)

        errors = []

        # 1. Size
        if image.GetSize() != mask.GetSize():
            errors.append("Size")

        # 2. Spacing
        if not np.allclose(
            image.GetSpacing(),
            mask.GetSpacing()
        ):
            errors.append("Spacing")

        # 3. Origin
        if not np.allclose(
            image.GetOrigin(),
            mask.GetOrigin()
        ):
            errors.append("Origin")

        # 4. Direction
        if not np.allclose(
            image.GetDirection(),
            mask.GetDirection()
        ):
            errors.append("Direction")

        if len(errors) == 0:
            valid += 1

        elif len(errors) == 1:

            if errors[0] == "Size":
                size_error += 1

            elif errors[0] == "Spacing":
                spacing_error += 1

            elif errors[0] == "Origin":
                origin_error += 1

            elif errors[0] == "Direction":
                direction_error += 1

            print(f"[{errors[0]} ERROR] {case_name}")

        else:

            multiple_error += 1

            print(
                f"[MULTIPLE ERRORS] {case_name}: "
                + ", ".join(errors)
            )

    print("\nSummary")
    print("-" * 80)

    print(f"Total cases:       {len(case_dirs)}")
    print(f"Valid:             {valid}")
    print(f"Missing image:     {missing_image}")
    print(f"Missing mask:      {missing_mask}")

    print()
    print(f"Size errors:       {size_error}")
    print(f"Spacing errors:    {spacing_error}")
    print(f"Origin errors:     {origin_error}")
    print(f"Direction errors:  {direction_error}")
    print(f"Multiple errors:   {multiple_error}")


if __name__ == "__main__":

    for direction, root in DATA_ROOTS.items():
        check_direction(direction, root)