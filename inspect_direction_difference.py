import os
import numpy as np
import SimpleITK as sitk


ROOTS = {
    "COR": r"D:\2PlacentaMRcor",
    "SAG": r"D:\2PlacentaMRsag",
    "TRA": r"D:\2PlacentaMRtra",
}


def max_direction_difference(image, mask):
    d1 = np.array(image.GetDirection(), dtype=float)
    d2 = np.array(mask.GetDirection(), dtype=float)
    return np.max(np.abs(d1 - d2))


def get_cases(root):
    cases = []

    for name in os.listdir(root):
        case_dir = os.path.join(root, name)

        if not os.path.isdir(case_dir):
            continue

        image_files = [
            f for f in os.listdir(case_dir)
            if f.endswith(".nii.gz")
            and "mask" not in f.lower()
        ]

        mask_files = [
            f for f in os.listdir(case_dir)
            if f.endswith(".nii.gz")
            and "mask" in f.lower()
        ]

        if len(image_files) == 1 and len(mask_files) == 1:
            cases.append((
                name,
                os.path.join(case_dir, image_files[0]),
                os.path.join(case_dir, mask_files[0])
            ))

    return cases


for direction, root in ROOTS.items():

    print()
    print("=" * 60)
    print(direction)
    print("=" * 60)

    results = []

    for case, image_path, mask_path in get_cases(root):

        try:
            image = sitk.ReadImage(image_path)
            mask = sitk.ReadImage(mask_path)

            diff = max_direction_difference(image, mask)
            results.append((case, diff))

        except Exception as e:
            print("READ ERROR:", case, e)

    if not results:
        print("No valid cases found.")
        continue

    diffs = np.array([x[1] for x in results])

    print("Total cases:", len(results))
    print("Maximum difference:", np.max(diffs))
    print("Minimum difference:", np.min(diffs))
    print("Mean difference:", np.mean(diffs))

    print()
    print("Difference distribution:")

    print("<= 1e-12 :", np.sum(diffs <= 1e-12))
    print("<= 1e-10 :", np.sum(diffs <= 1e-10))
    print("<= 1e-8  :", np.sum(diffs <= 1e-8))
    print("<= 1e-6  :", np.sum(diffs <= 1e-6))
    print("<= 1e-4  :", np.sum(diffs <= 1e-4))
    print(">  1e-4  :", np.sum(diffs > 1e-4))

    print()
    print("Top 10 largest differences:")

    results.sort(key=lambda x: x[1], reverse=True)

    for case, diff in results[:10]:
        print(f"{case}: {diff:.12e}")