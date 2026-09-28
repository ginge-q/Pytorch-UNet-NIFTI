import os
import SimpleITK as sitk
import numpy as np


DATA_ROOTS = {
    "cor": r"D:\2PlacentaMRcor",
    "sag": r"D:\2PlacentaMRsag",
    "tra": r"D:\2PlacentaMRtra",
}


def check_direction(direction, root):
    print("\n" + "=" * 70)
    print(f"Checking {direction.upper()}")
    print(f"Root: {root}")
    print("=" * 70)

    if not os.path.isdir(root):
        print(f"[ERROR] Directory does not exist: {root}")
        return

    case_dirs = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]

    print(f"Number of case directories: {len(case_dirs)}")

    valid = 0
    missing_image = 0
    missing_mask = 0
    geometry_error = 0
    empty_mask = 0
    non_binary_mask = 0

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

        try:
            image = sitk.ReadImage(image_path)
            mask = sitk.ReadImage(mask_path)

            image_array = sitk.GetArrayFromImage(image)
            mask_array = sitk.GetArrayFromImage(mask)

            geometry_ok = True

            if image.GetSize() != mask.GetSize():
                geometry_ok = False

            if not np.allclose(
                image.GetSpacing(),
                mask.GetSpacing()
            ):
                geometry_ok = False

            if not np.allclose(
                image.GetOrigin(),
                mask.GetOrigin()
            ):
                geometry_ok = False

            if not np.allclose(
                image.GetDirection(),
                mask.GetDirection()
            ):
                geometry_ok = False

            unique_values = np.unique(mask_array)

            if mask_array.sum() == 0:
                empty_mask += 1

            if not np.all(np.isin(unique_values, [0, 1])):
                non_binary_mask += 1

            if not geometry_ok:
                geometry_error += 1

                print(f"\n[GEOMETRY ERROR] {case_name}")

                print("  Image:")
                print("    Size:", image.GetSize())
                print("    Spacing:", image.GetSpacing())
                print("    Origin:", image.GetOrigin())
                print("    Direction:", image.GetDirection())

                print("  Mask:")
                print("    Size:", mask.GetSize())
                print("    Spacing:", mask.GetSpacing())
                print("    Origin:", mask.GetOrigin())
                print("    Direction:", mask.GetDirection())

            else:
                valid += 1

        except Exception as e:
            print(f"[READ ERROR] {case_name}: {e}")

    print("\nSummary")
    print("-" * 70)
    print(f"Total cases:       {len(case_dirs)}")
    print(f"Valid cases:       {valid}")
    print(f"Missing image:     {missing_image}")
    print(f"Missing mask:      {missing_mask}")
    print(f"Geometry errors:   {geometry_error}")
    print(f"Empty masks:       {empty_mask}")
    print(f"Non-binary masks:  {non_binary_mask}")


if __name__ == "__main__":

    for direction, root in DATA_ROOTS.items():
        check_direction(direction, root)