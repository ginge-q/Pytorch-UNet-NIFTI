import os
import numpy as np
import SimpleITK as sitk


ROOTS = {
    "COR": r"D:\2PlacentaMRcor",
    "TRA": r"D:\2PlacentaMRtra",
}


TARGETS = {
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


for direction, cases in TARGETS.items():

    print()
    print("=" * 70)
    print(direction)
    print("=" * 70)

    root = ROOTS[direction]

    for case in cases:

        case_dir = os.path.join(root, case)

        image_path, mask_path = find_files(case_dir)

        print()
        print("-" * 70)
        print("CASE:", case)

        if image_path is None:
            print("ERROR: image or mask not found")
            continue

        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)

        size = image.GetSize()

        points = [
            (0, 0, 0),
            (
                size[0] // 2,
                size[1] // 2,
                size[2] // 2
            ),
            (
                size[0] - 1,
                size[1] - 1,
                size[2] - 1
            )
        ]

        print("Size:", size)

        for index in points:

            p_image = np.array(
                image.TransformIndexToPhysicalPoint(index)
            )

            p_mask = np.array(
                mask.TransformIndexToPhysicalPoint(index)
            )

            difference = np.abs(p_image - p_mask)

            print()
            print("Index:", index)
            print("Image physical point:", p_image)
            print("Mask  physical point:", p_mask)
            print("Absolute difference:", difference)
            print("Maximum difference:", np.max(difference))