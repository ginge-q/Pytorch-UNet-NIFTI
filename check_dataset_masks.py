from nifti_dataset import NiftiSliceDataset
import torch

dataset = NiftiSliceDataset(
    r"D:\DL\Pytorch-UNet\cor_train.csv",
    target_size=512,
)

positive_slices = 0
empty_slices = 0

for i in range(len(dataset)):
    mask = dataset[i]["mask"]

    if torch.any(mask > 0):
        positive_slices += 1
    else:
        empty_slices += 1

print("Total slices   :", len(dataset))
print("Positive slices:", positive_slices)
print("Empty slices   :", empty_slices)

sample_indices = []

for i in range(len(dataset)):
    if torch.any(dataset[i]["mask"] > 0):
        sample_indices.append(i)
        if len(sample_indices) >= 5:
            break

print("\nFirst 5 positive slices:")

for i in sample_indices:
    sample = dataset[i]
    print(
        "index =", i,
        "case =", sample["case"],
        "slice =", sample["slice_index"],
        "values =", torch.unique(sample["mask"]).tolist()
    )
