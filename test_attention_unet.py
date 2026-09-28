import torch

from unet.attention_unet import AttentionUNet


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

model = AttentionUNet(
    n_channels=1,
    n_classes=2
).to(device)

print(
    "Trainable parameters:",
    f"{count_parameters(model):,}"
)

x = torch.randn(
    1,
    1,
    512,
    512,
    device=device
)

print("Input shape :", tuple(x.shape))

model.eval()

with torch.no_grad():
    y = model(x)

print("Output shape:", tuple(y.shape))

print(
    "Output finite:",
    bool(torch.isfinite(y).all())
)

assert y.shape == (
    1,
    2,
    512,
    512
)

assert torch.isfinite(y).all()

print()
print("=" * 60)
print("Attention U-Net structure test PASSED.")
print("=" * 60)
