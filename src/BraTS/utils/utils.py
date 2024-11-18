import numpy as np
import torch
import os
from monai.losses import DiceLoss, FocalLoss, TverskyLoss, DiceCELoss, DiceFocalLoss
from tqdm import tqdm
import pandas as pd
import csv
from monai.metrics import DiceMetric


# Define utility functions
def dice_metric(predicted, target):
    dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True, reduction='mean')
    value = 1 - dice_value(predicted, target).item()
    return value

def iou_metric(predicted, target):
    dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True, jaccard=True, reduction='mean')
    value = 1 - dice_value(predicted, target).item()
    return value

def calculate_weights(val1, val2):
    count = np.array([val1, val2])
    summ = count.sum()
    weights = count / summ
    weights = 1 / weights
    summ = weights.sum()
    weights = weights / summ
    return torch.tensor(weights, dtype=torch.float32)

def count_foreground_background(label):
    label_np = label.cpu().detach().numpy()
    foreground_pixels = np.sum(label_np != 0)
    background_pixels = np.sum(label_np == 0)
    return foreground_pixels, background_pixels

def binarize_output(logits, threshold=0.5):
    return (logits >= threshold).float()

def merge_epoch_csv(model_dir, max_epochs):
    all_data = pd.DataFrame()
    for epoch in range(1, max_epochs + 1):
        csv_file = os.path.join(model_dir, f'dice_metrics_epoch_{epoch}.csv')
        if os.path.exists(csv_file):
            epoch_data = pd.read_csv(csv_file)
            all_data = pd.concat([all_data, epoch_data])
    merged_csv_file = os.path.join(model_dir, 'merged_epoch_metrics.csv')
    all_data.to_csv(merged_csv_file, index=False)
    print(f"All epoch CSV files merged and saved to: {merged_csv_file}")

def calculate_gradient(image):
    gradient_x = np.gradient(image, axis=0)
    gradient_y = np.gradient(image, axis=1)
    gradient_z = np.gradient(image, axis=2)
    return gradient_x, gradient_y, gradient_z

def calculate_gradient_magnitude(gradient_x, gradient_y, gradient_z):
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2 + gradient_z**2)
    return magnitude

def calculate_mean_smoothness(gradient_magnitude):
    return np.mean(gradient_magnitude)

def compute_smoothness(image_tensor):
    gradient_x, gradient_y, gradient_z = calculate_gradient(image_tensor)
    gradient_magnitude = calculate_gradient_magnitude(gradient_x, gradient_y, gradient_z)
    return calculate_mean_smoothness(gradient_magnitude)