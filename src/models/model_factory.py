import segmentation_models_pytorch as smp

def build_model(config):
    architecture = config["model"]["architecture"]
    encoder = config["model"]["encoder"]
    encoder_weights = config["model"]["encoder_weights"]
    activation = config["model"]["activation"]
    num_classes = len(config["data"]["classes"])

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=num_classes,
        activation=activation,
    )

    return model
