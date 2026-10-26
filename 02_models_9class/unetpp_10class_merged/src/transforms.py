import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """CHW normalized float tensor -> HWC uint8 RGB numpy array."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img_tensor.detach().cpu() * std + mean
    return (img.clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def get_train_transform(height: int, width: int):
    return A.Compose([
        A.RandomScale(scale_limit=(-0.2, 0.3), p=0.5),
        A.PadIfNeeded(min_height=height, min_width=width,
                      border_mode=0, fill=0, fill_mask=0),
        A.RandomCrop(height=height, width=width, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.GaussNoise(p=0.15),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transform(height: int, width: int):
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_train_transform_cached():
    """For CachedUAVScenesDataset: images are already resized to the target
    size at cache-build time, so no Resize/RandomScale/Crop here — just the
    cheap per-epoch augmentations plus normalize."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transform_cached():
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
