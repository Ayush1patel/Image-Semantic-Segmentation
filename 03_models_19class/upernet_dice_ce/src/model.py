import segmentation_models_pytorch as smp

from class_mapping import NUM_CLASSES


def build_model(encoder_name: str = "resnet34", encoder_weights: str = "imagenet",
                 num_classes: int = NUM_CLASSES, in_channels: int = 3):
    """UPerNet (Xiao et al. 2018 - https://arxiv.org/abs/1807.10221) with a
    pretrained-ImageNet ResNet encoder via segmentation_models_pytorch.

    This is the decoder/backbone pairing used in the UAVScenes paper's own
    image-segmentation benchmark (Table 3: ResNet-34/50/101 + UperNet via
    timm). Here we use smp's UPerNet implementation instead of timm so the
    rest of the pipeline (dataset, loss, optimizer, metrics, progress viz)
    stays identical to the sibling UNet/UNet++ projects in this workspace -
    only the decoder architecture changes, everything else (input size,
    optimizer, batch size, loss) is matched to `unet_dice_ce`
    for a fair architecture-only comparison.
    """
    model = smp.UPerNet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # raw logits; loss functions handle softmax/sigmoid
    )
    return model
