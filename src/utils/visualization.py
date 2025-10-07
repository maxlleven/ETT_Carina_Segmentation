import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(image, mask_true, mask_pred, name=None, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image.transpose(1,2,0))
    plt.title("Input Image"); plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_true.squeeze(), cmap="gray")
    plt.title("Ground Truth"); plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(mask_pred.squeeze(), cmap="gray")
    plt.title("Prediction"); plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.show()
