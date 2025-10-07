import os
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import imgaug.augmenters as iaa


class MedicalDataset(BaseDataset):
    """
    Dataset class for Carina / ETT segmentation.
    Handles augmentation, preprocessing, and cropping.
    """

    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

        # For medical datasets where masks use .dcm1.png or .png
        self.masks_fps = [
            os.path.join(masks_dir, image_id.replace(".dcm.jpg", ".dcm1.png"))
            if os.path.exists(os.path.join(masks_dir, image_id.replace(".dcm.jpg", ".dcm1.png")))
            else os.path.join(masks_dir, image_id.replace(".jpg", ".png"))
            for image_id in self.ids
        ]

        # Class definitions (binary segmentation by default)
        self.class_values = [255]
        self.classes = classes or ["tube"]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        #  Add back the fixed cropping/normalization sequence
        self.seq = iaa.Sequential([
            iaa.MaxPooling(3, keep_size=False),
            iaa.CropToFixedSize(width=384, height=712, position="center-bottom"),
            iaa.CropToFixedSize(width=384, height=384, position="center")
        ])

    def __getitem__(self, i):
        # --- Load image and mask ---
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # --- Convert mask to binary ---
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype("float32")

        # --- Apply imgaug fixed preprocessing ---
        image = self.seq.augment_image(image)
        mask = self.seq.augment_image(mask)

        # --- Apply Albumentations augmentations ---
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # --- Apply preprocessing (normalization + to_tensor) ---
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.ids)


# ---- Helper functions ----
def get_training_augmentation():
    train_transform = [
        albu.PadIfNeeded(384, 384),
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=5, shift_limit=0.1, p=1, border_mode=0),
        albu.CLAHE(clip_limit=[1, 3], p=0.5),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [albu.PadIfNeeded(384, 384)]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
