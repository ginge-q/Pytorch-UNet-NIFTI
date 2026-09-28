import os
import numpy as np
import SimpleITK as sitk


ROOTS = {
    "COR": r"D:\2PlacentaMRcor",
    "SAG": r"D:\2PlacentaMRsag",
    "TRA": r"D:\2PlacentaMRtra",
}


def find_files(case_dir):
    image_files = [
        f for f in os.listdir(case_dir)
        if f.endswith(".nii.gz") and "mask" not in f.lower()
    ]

    mask_files = [
        f for f in os.listdir(case_dir)
        if f.endswith(".nii.gz") and "mask" in f.lower()
    ]

    if len(image_files) != 1 or len(mask_files) != 1:
        return None, None

    return (
        os.path.join(case_dir, image_files[0]),
        os.path.join(case_dir, mask_files[0])
    )


targets = {
    "COR": [
        "yangqiuhua_20211216",
        "liyiping",
    ],
    "TRA": [
        "yangjie_20220301",
        "lianggelin",
        "zhangchunhua_20220922",
        "zhangnana_20211222",
        "zhangxue_20230725",
        "hanjingjing_20210728",
    ],
}


for direction, cases in targets.items():

    root = ROOTS[direction]

    print()
    print("=" * 70)
    print(direction)
    print("=" * 70)

    for case in cases:

        case_dir = os.path.join(root, case)

        image_path, mask_path = find_files(case_dir)

        print()
        print("-" * 70)
        print("CASE:", case)

        if image_path is None:
            print("ERROR: image or mask file not found")
            continue

        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)

        image_direction = np.array(
            image.GetDirection(),
            dtype=float
        )

        mask_direction = np.array(
            mask.GetDirection(),
            dtype=float
        )

        difference = np.abs(
            image_direction - mask_direction
        )

        print("Maximum direction difference:")
        print(np.max(difference))

        print()
        print("Image Direction:")
        print(image_direction.reshape(3, 3))

        print()
        print("Mask Direction:")
        print(mask_direction.reshape(3, 3))

        print()
        print("Absolute Difference:")
        print(difference.reshape(3, 3))

        print()
        print("Size:")
        print("Image:", image.GetSize())
        print("Mask :", mask.GetSize())

        print()
        print("Spacing:")
        print("Image:", image.GetSpacing())
        print("Mask :", mask.GetSpacing())

        print()
        print("Origin:")
        print("Image:", image.GetOrigin())
        print("Mask :", mask.GetOrigin())