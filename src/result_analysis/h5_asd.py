

import os
import glob
import numpy as np
import nibabel as nib
import torch
import pandas as pd
from monai.metrics import DiceMetric, compute_average_surface_distance
from scipy.spatial.distance import directed_hausdorff

# -------------------- Config --------------------

IN_DIR = r'D:\MY_paper\train_model\picai_dataset\SegResgNet_AFL\main\pred_mask'   
OUT_EXCEL = r"D:\MY_paper\train_model\picai_dataset\SegResgNet_AFL\main\SegResgNet_AFL_main_metrics.xlsx"

# -------------------- Helpers --------------------
def load_nifti_binmask(path):
    """Load NIfTI and return a binary (0/1) ndarray and the nib object."""
    nii = nib.load(path)
    arr = nii.get_fdata()
    arr = (arr > 0).astype(np.uint8)
    return arr, nii

def dice_coefficient(pred, gt):
    """Dice via MONAI on single-case binary masks."""
    pred_tensor = torch.tensor(pred).unsqueeze(0).unsqueeze(0).float()
    gt_tensor   = torch.tensor(gt).unsqueeze(0).unsqueeze(0).float()
    dice = DiceMetric(include_background=False, reduction="mean")
    dice(y_pred=pred_tensor, y=gt_tensor)
    result = dice.aggregate().item()
    dice.reset()
    return float(result)

def hausdorff_distance_mm(pred, gt, spacing):
    """Symmetric (two-sided) Hausdorff distance in mm using point sets."""
    # Points in index coords -> scale by spacing to mm coords
    pred_pts = np.argwhere(pred)
    gt_pts   = np.argwhere(gt)

    if pred_pts.size == 0 and gt_pts.size == 0:
        return 0.0  # both empty → distance 0
    if pred_pts.size == 0 or gt_pts.size == 0:
        return float('inf')  # one empty → undefined → inf

    pred_pts_mm = pred_pts * np.array(spacing)
    gt_pts_mm   = gt_pts   * np.array(spacing)

    hd1 = directed_hausdorff(pred_pts_mm, gt_pts_mm)[0]
    hd2 = directed_hausdorff(gt_pts_mm, pred_pts_mm)[0]
    return float(max(hd1, hd2))

def average_surface_distance_mm(pred, gt, spacing=None):
    """Average Surface Distance (symmetric) in mm using MONAI."""
    pred_tensor = torch.tensor(pred).unsqueeze(0).unsqueeze(0).float()
    gt_tensor   = torch.tensor(gt).unsqueeze(0).unsqueeze(0).float()
    img_dim = pred_tensor.ndim - 2  # spatial dims
    if spacing is not None:
        spacing = [float(s) for s in spacing[:img_dim]]
    try:
        asd = compute_average_surface_distance(
            y_pred=pred_tensor.numpy(),
            y=gt_tensor.numpy(),
            include_background=False,
            symmetric=True,
            distance_metric='euclidean',
            spacing=spacing
        )
        # MONAI returns ndarray shape [B, C], here [1, 1]
        if hasattr(asd, "__len__"):
            return float(np.array(asd).reshape(-1)[0])
        return float(asd)
    except Exception as e:
        print(f"[Warning] ASD failed: {e}")
        return np.nan

def extract_patient_id(basename):
    """
    From '10699_1000715_0000-seg-resized.nii.gz' -> '10699'
    """
    core = basename.replace(".nii.gz", "")
    return core.split("_")[0]

# -------------------- Main --------------------
def main():
    seg_files = sorted(glob.glob(os.path.join(IN_DIR, "*-seg-resized.nii.gz")))
    if not seg_files:
        print(f"No '*-seg-resized.nii.gz' found in: {IN_DIR}")
        return

    rows = []
    for seg_path in seg_files:
        base = os.path.basename(seg_path)
        case_core = base.replace("-seg-resized.nii.gz", "")  # e.g., '10699_1000715_0000'
        pred_path = os.path.join(IN_DIR, f"{case_core}-pred.nii.gz")

        if not os.path.exists(pred_path):
            print(f"[Skip] Missing pred for {case_core} -> {pred_path}")
            continue

        # Load masks & spacing
        gt, gt_nii = load_nifti_binmask(seg_path)
        pred, _ = load_nifti_binmask(pred_path)

        # Get spacing in mm from the ground-truth (resized) file
        spacing = gt_nii.header.get_zooms()[:3]  # (sx, sy, sz)

        # Metrics
        dsc = dice_coefficient(pred, gt)
        hd  = hausdorff_distance_mm(pred, gt, spacing)
        asd = average_surface_distance_mm(pred, gt, spacing)

        patient_id = extract_patient_id(base)

        rows.append({
            "patient_id": patient_id,
            "case_name": case_core,
            "dice": dsc,
            "hd_mm": hd,
            "asd_mm": asd,
            "seg_path": seg_path,
            "pred_path": pred_path,
            "spacing_x": spacing[0],
            "spacing_y": spacing[1],
            "spacing_z": spacing[2],
        })

        print(f"[OK] {case_core} | Dice={dsc:.4f} | HD={hd:.3f} mm | ASD={asd:.3f} mm")

    if not rows:
        print("No valid pairs processed.")
        return

    df = pd.DataFrame(rows)
    # Optional: sort by patient_id then case_name
    df = df.sort_values(["patient_id", "case_name"]).reset_index(drop=True)

    # Save to Excel
    os.makedirs(os.path.dirname(OUT_EXCEL), exist_ok=True)
    with pd.ExcelWriter(OUT_EXCEL, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="metrics")

    print(f"\nSaved metrics to: {OUT_EXCEL}")
    print(df[["patient_id", "case_name", "dice", "hd_mm", "asd_mm"]].head())

if __name__ == "__main__":
    main()