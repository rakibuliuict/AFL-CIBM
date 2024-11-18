import SimpleITK as sitk
import os
import pandas as pd
import sys
import argparse

from lession_wise_seg import eval_metrics as sg

def calculate_metrics_for_all_patients(ground_truth_folder, prediction_folder, output_excel_file):
    results = []

    for gt_filename in os.listdir(ground_truth_folder):
    
        patient_id = gt_filename.split('.')[0]
        gt_path = os.path.join(ground_truth_folder, gt_filename)

    
        pred_filename = f"{patient_id}-pred.nii.gz"
        pred_path = os.path.join(prediction_folder, pred_filename)

        if not os.path.exists(pred_path):
            print(f"Prediction file for {gt_filename} not found.")
            continue

        ground_truth = sitk.ReadImage(gt_path)
        prediction = sitk.ReadImage(pred_path)

        ground_truth_array = (sitk.GetArrayFromImage(ground_truth) > 0).astype(bool)
        prediction_array = (sitk.GetArrayFromImage(prediction) > 0).astype(bool)

        # Calculate lesion-wise metrics
        tp, fp, fn, _ = sg.compute_lesion_f1_score(ground_truth_array, prediction_array)  # Ignore F1 Score

        print(f"Patient: {patient_id}, TP: {tp}, FP: {fp}, FN: {fn}")

        results.append({
            'Patient ID': patient_id,
            'True Positives': tp,
            'False Positives': fp,
            'False Negatives': fn
        })

    df = pd.DataFrame(results)
    df.to_excel(output_excel_file, index=False)
    print(f"Metrics saved to {output_excel_file}")

def main():
    parser = argparse.ArgumentParser(description="Calculate lesion-wise metrics from prediction and ground truth files.")
    parser.add_argument("ground_truth_folder", type=str, help="Path to the folder containing ground truth .nii.gz files")
    parser.add_argument("prediction_folder", type=str, help="Path to the folder containing predicted .nii.gz files")
    parser.add_argument("output_folder", type=str, help="Path to the folder where the output Excel file will be saved")

    args = parser.parse_args()

    ground_truth_folder = args.ground_truth_folder
    prediction_folder = args.prediction_folder
    output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)
    

    output_excel_file = os.path.join(output_folder, "lesion_wise_TP_FP_FN.xlsx")

    calculate_metrics_for_all_patients(ground_truth_folder, prediction_folder, output_excel_file)

if __name__ == "__main__":
    main()
