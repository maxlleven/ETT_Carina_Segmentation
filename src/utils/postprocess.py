import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def measure_localization(pred_masks, true_masks, names, output_excel):
    df = pd.DataFrame(columns=["Name", "X_True", "Y_True", "X_Pred", "Y_Pred", "Distance_cm"])
    pixel_to_cm = 1 / 72.0  # example scale
    for name, pred_mask, gt_mask in zip(names, pred_masks, true_masks):
        pred_cnts, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        gt_cnts, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(pred_cnts) == 0 or len(gt_cnts) == 0:
            continue
        pred_center = np.mean(pred_cnts[0][:, 0, :], axis=0)
        gt_center = np.mean(gt_cnts[0][:, 0, :], axis=0)
        dist_cm = np.linalg.norm(pred_center - gt_center) * pixel_to_cm
        df.loc[len(df)] = [name, gt_center[0], gt_center[1], pred_center[0], pred_center[1], dist_cm]
    df.to_excel(output_excel, index=False)
    return df
