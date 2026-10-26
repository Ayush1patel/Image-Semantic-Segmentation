import segmentation_models_pytorch as smp

from class_mapping import NUM_CLASSES


def build_model(encoder_name: str = "resnet34", encoder_weights: str = "imagenet",
                 num_classes: int = NUM_CLASSES, in_channels: int = 3):
    """UNet++ with a pretrained-ImageNet encoder via segmentation_models_pytorch."""
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # raw logits; loss functions handle softmax/sigmoid
    )
    return model
