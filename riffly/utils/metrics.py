"""Custom metrics to be used in evaluation."""

import numpy as np


def pixel_iou(y_true, y_pred, average="macro"):
    # Flatten the 2D arrays to 1D
    y_true_flat = y_true
    y_pred_flat = y_pred

    # True Positives, False Positives, False Negatives
    intersection = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    union = np.sum((y_true_flat == 1) | (y_pred_flat == 1))

    # IoU calculation
    return intersection / union if union != 0 else 0
