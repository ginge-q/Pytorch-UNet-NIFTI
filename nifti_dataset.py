import csv
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class NiftiSliceDataset(Dataset):
    """
    将 3D NIfTI 图像按 z 方向切成 2D slice。

    CSV 格式：
        case,image_path,mask_path

    每个 __getitem__ 返回：
        image       : Tensor [1, H, W], float32
        mask        : Tensor [H, W], int64
        case_name   : str
        slice_index : int
    """

    def __init__(self, csv_path, target_size=512):
        self.csv_path = Path(csv_path)
        self.target_size = target_size

        self.cases = []

        with open(self.csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                self.cases.append({
                    "case": row["case"],
                    "image": row["image_path"],
                    "mask": row["mask_path"],
                })

        # 建立所有 slice 的索引
        self.samples = []

        for case_info in self.cases:
            image_path = case_info["image"]
            mask_path = case_info["mask"]

            image_itk = sitk.ReadImage(image_path)
            mask_itk = sitk.ReadImage(mask_path)

            image_array = sitk.GetArrayFromImage(image_itk)
            mask_array = sitk.GetArrayFromImage(mask_itk)

            if image_array.shape != mask_array.shape:
                raise ValueError(
                    f"Image/mask shape mismatch:\n"
                    f"Case: {case_info['case']}\n"
                    f"Image: {image_array.shape}\n"
                    f"Mask : {mask_array.shape}"
                )

            num_slices = image_array.shape[0]

            for z in range(num_slices):
                self.samples.append({
                    "case": case_info["case"],
                    "image": image_path,
                    "mask": mask_path,
                    "slice_index": z,
                })

        print(f"Patients: {len(self.cases)}")
        print(f"Slices: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        case_name = sample["case"]
        image_path = sample["image"]
        mask_path = sample["mask"]
        z = sample["slice_index"]

        # ---------------------------------------------------------
        # 读取 NIfTI
        # ---------------------------------------------------------
        image_itk = sitk.ReadImage(image_path)
        mask_itk = sitk.ReadImage(mask_path)

        image = sitk.GetArrayFromImage(image_itk)[z]
        mask = sitk.GetArrayFromImage(mask_itk)[z]

        # ---------------------------------------------------------
        # 图像转 float32
        # ---------------------------------------------------------
        image = image.astype(np.float32)

        # ---------------------------------------------------------
        # 每个 slice 做 z-score normalization
        # ---------------------------------------------------------
        mean = image.mean()
        std = image.std()

        if std > 1e-8:
            image = (image - mean) / std
        else:
            image = image - mean

        # ---------------------------------------------------------
        # mask 二值化
        # ---------------------------------------------------------
        mask = (mask > 0).astype(np.int64)

        # ---------------------------------------------------------
        # 转成 PyTorch Tensor
        # image: [1, H, W]
        # mask : [H, W]
        # ---------------------------------------------------------
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask)

        # ---------------------------------------------------------
        # resize 到固定 512 × 512
        # image 使用 bilinear
        # mask 使用 nearest
        # ---------------------------------------------------------
        if image.shape[-2:] != (self.target_size, self.target_size):

            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(self.target_size, self.target_size),
                mode="nearest",
            ).squeeze(0).squeeze(0).long()

        # ---------------------------------------------------------
        # 最终返回
        #
        # 注意：
        # 这里必须返回真正的数据，而不是 dictionary。
        # ---------------------------------------------------------
        return image, mask, case_name, z


if __name__ == "__main__":
    # 单独运行本文件时进行测试
    csv_path = r"D:\DL\Pytorch-UNet\cor_train.csv"

    dataset = NiftiSliceDataset(
        csv_path=csv_path,
        target_size=512,
    )

    sample = dataset[0]

    print("\nDataset test:")
    print("Number of returned items:", len(sample))
    print("Types:", [type(x).__name__ for x in sample])
    print("Image shape:", sample[0].shape)
    print("Mask shape:", sample[1].shape)
    print("Image dtype:", sample[0].dtype)
    print("Mask dtype:", sample[1].dtype)
    print("Case:", sample[2])
    print("Slice:", sample[3])
    print("Mask values:", torch.unique(sample[1]).tolist())