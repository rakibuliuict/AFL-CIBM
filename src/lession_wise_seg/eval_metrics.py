# test_eval_metrics.py

import numpy as np
import cc3d

# Define compute_lesion_f1_score directly here for testing purposes
def compute_lesion_f1_score(ground_truth, prediction, empty_value=1.0, connectivity=26):
    ground_truth = np.asarray(ground_truth).astype(bool)
    prediction = np.asarray(prediction).astype(bool)
    tp, fp, fn = 0, 0, 0

    # Intersection and connected components
    intersection = np.logical_and(ground_truth, prediction)
    labeled_ground_truth, N = cc3d.connected_components(ground_truth, connectivity=connectivity, return_N=True)

    # Count TP and FN in ground truth
    if N > 0:
        for _, binary_cluster_image in cc3d.each(labeled_ground_truth, binary=True, in_place=True):
            if np.logical_and(binary_cluster_image, intersection).any():
                tp += 1
            else:
                fn += 1

    # Count FP in prediction
    labeled_prediction, N = cc3d.connected_components(prediction, connectivity=connectivity, return_N=True)
    if N > 0:
        for _, binary_cluster_image in cc3d.each(labeled_prediction, binary=True, in_place=True):
            if not np.logical_and(binary_cluster_image, ground_truth).any():
                fp += 1

    # Calculate F1 score
    if tp + fp + fn == 0:
        _, N = cc3d.connected_components(ground_truth, connectivity=connectivity, return_N=True)
        f1_score = empty_value if N == 0 else 0
    else:
        f1_score = tp / (tp + (fp + fn) / 2)

    # Return all four values
    return tp, fp, fn, f1_score