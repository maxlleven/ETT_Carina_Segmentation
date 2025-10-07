import os
import yaml
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from src.models.model_factory import build_model
from src.utils.dataset import MedicalDataset, get_training_augmentation, get_validation_augmentation, get_preprocessing
from src.utils.losses import DiceBCELoss
from src.utils.trainer import train_one_epoch, validate_one_epoch
from src.utils.metrics import iou_score, dice_loss

def main(config_path="./config/carina_config.yaml"):
    # --- Load config ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
    os.makedirs(config["train"]["checkpoint_dir"], exist_ok=True)

    # --- Build model ---
    model = build_model(config).to(device)
    loss_fn = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    metrics = [iou_score, dice_loss]

    # --- Dataset setup ---
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        config["model"]["encoder"], config["model"]["encoder_weights"]
    )

    base = config["data"]["base_dir"]
    fold = config["data"]["fold"]

    train_dataset = MedicalDataset(
        images_dir=f"{base}/Fold{fold}/train",
        masks_dir=f"{base}/Fold{fold}/trainannot",
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=config["data"]["classes"],
    )

    val_dataset = MedicalDataset(
        images_dir=f"{base}/Fold{fold}/val",
        masks_dir=f"{base}/Fold{fold}/valannot",
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=config["data"]["classes"],
    )

    train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # --- Training loop ---
    best_score = float("inf")
    num_epochs = config["train"]["epochs"]

    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch + 1}/{num_epochs}")

        train_logs = train_one_epoch(model, train_loader, loss_fn, optimizer, device, metrics)
        val_logs = validate_one_epoch(model, val_loader, loss_fn, device, metrics)

        train_loss = train_logs.get("loss", None)
        val_loss = val_logs.get("dice_loss", None)
        val_iou = 1 - val_loss if val_loss is not None else None

        print(f"Train Loss: {train_loss:.4f} | Val Dice Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")

        # Save best model
        if val_loss < best_score:
            best_score = val_loss
            torch.save(model.state_dict(), os.path.join(config["train"]["checkpoint_dir"], "best_model.pth"))
            print(f" Model saved at epoch {epoch + 1} with Dice Loss = {val_loss:.4f}")

    print("\n Training completed!")


if __name__ == "__main__":
    # Choose config here: Carina or ETT
    # main("./config/ett_config.yaml")
    main("./config/carina_config.yaml")
