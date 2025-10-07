import os
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from src.models.model_factory import build_model
from src.utils.dataset import MedicalDataset, get_validation_augmentation, get_preprocessing
from src.utils.metrics import iou_score, dice_loss


def visualize_prediction(image, mask_true, mask_pred):
    """Display image, ground truth mask, and prediction side by side"""
    # image shape is (C, H, W)
    img = np.transpose(image, (1, 2, 0))
    img = img - img.min()         # normalize range [0, 1]
    img = img / (img.max() + 1e-8)

    mask_true = mask_true.squeeze()
    mask_pred = mask_pred.squeeze()

    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    if img.shape[-1] == 1:
        plt.imshow(img.squeeze(), cmap="gray")
    else:
        plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(mask_true, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(mask_pred, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def test_model(config_path="./config/carina_config.yaml", visualize=True):
    # --- Load config ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")

    # --- Build model ---
    model = build_model(config).to(device)
    model_path = os.path.join(config["train"]["checkpoint_dir"], "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Load test data ---
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        config["model"]["encoder"], config["model"]["encoder_weights"]
    )

    base = config["data"]["base_dir"]
    fold = config["data"]["fold"]
    test_dataset = MedicalDataset(
        images_dir=f"{base}/Fold{fold}/test",
        masks_dir=f"{base}/Fold{fold}/testannot",
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=config["data"]["classes"],
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- Evaluation metrics ---
    dice_scores = []
    iou_scores = []

    print(f"\nEvaluating on test set ({len(test_dataset)} images)...")

    with torch.no_grad():
        for i, (image, mask) in enumerate(test_loader):
            image, mask = image.to(device), mask.to(device)
            pred = torch.sigmoid(model(image))
            pred_bin = (pred > 0.5).float()

            dice = 1 - dice_loss(pred, mask).item()
            iou = iou_score(pred, mask).item()
            dice_scores.append(dice)
            iou_scores.append(iou)

            if visualize and i < 3:  # show a few examples
                visualize_prediction(
                    image[0].cpu().numpy(), mask[0].cpu().numpy(), pred_bin[0].cpu().numpy()
                )

    # --- Print results ---
    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)
    print(f"\n Test Results:")
    print(f"  • Mean Dice Coefficient: {mean_dice:.4f}")
    print(f"  • Mean IoU:              {mean_iou:.4f}")
    print(f"  • Total images:          {len(test_dataset)}")


if __name__ == "__main__":
    test_model("./config/carina_config.yaml", visualize=True)
