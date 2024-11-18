import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import argparse
import os

def load_model_paths(input_txt_file):

    file_paths = {}
    with open(input_txt_file, 'r') as file:
        for line in file:
            model_name, file_path = line.strip().split(',')
            file_paths[model_name] = file_path.strip()
    return file_paths

def perform_wilcoxon_test(file_paths, hypothesized_median, output_file):

    results = []

    for model_name, file_path in file_paths.items():
        df = pd.read_excel(file_path)

        score_differences = df['Test Dice'] - hypothesized_median
        statistic, p_value = wilcoxon(score_differences)

        results.append({
            'Model': model_name,
            'Wilcoxon Test Statistic': statistic,
            'P-value': f"{p_value:.2e}",
        })

    results_df = pd.DataFrame(results)
    results_df.to_excel(output_file, index=False)
    print(f"\nResults have been saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Perform Wilcoxon signed-rank test for model performance.")
    parser.add_argument("input_txt_file", type=str, help="Path to the text file containing model names and file paths.")
    parser.add_argument("output_file", type=str, help="Path to save the Wilcoxon test results Excel file.")
    parser.add_argument("--median", type=float, default=0.7, help="Hypothesized median for the Wilcoxon test (default: 0.7).")

    args = parser.parse_args()

    file_paths = load_model_paths(args.input_txt_file)
    perform_wilcoxon_test(file_paths, args.median, args.output_file)

if __name__ == "__main__":
    main()
