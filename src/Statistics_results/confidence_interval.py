import pandas as pd
from scipy import stats
import sys

def read_file_paths(file_path):
    file_paths = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                model_name, path = line.split(maxsplit=1)
                file_paths[model_name] = path
    return file_paths


def calculate_confidence_interval(scores, confidence=0.95):
    std_dev = scores.std(ddof=1)  
    n = len(scores)
    margin_of_error = std_dev * stats.t.ppf((1 + confidence) / 2, n - 1) / (n ** 0.5)  # Margin of error
    lower_bound = scores.mean() - margin_of_error
    upper_bound = scores.mean() + margin_of_error
    return lower_bound, upper_bound


if len(sys.argv) < 3:
    print("Usage: python script_name.py <path_to_text_file> <output_file_path>")
    sys.exit(1)


text_file_path = sys.argv[1]
output_file_path = sys.argv[2]


file_paths = read_file_paths(text_file_path)


dice_scores = {}

for model_name, file_path in file_paths.items():
    df = pd.read_excel(file_path)
    dice_scores[model_name] = df['Test Dice']  

results = []


for model_name, scores in dice_scores.items():
    confidence_interval = calculate_confidence_interval(scores)
    results.append({
        'Model': model_name,
        'Confidence Interval Lower Bound': confidence_interval[0],
        'Confidence Interval Upper Bound': confidence_interval[1]
    })

results_df = pd.DataFrame(results)
results_df.to_excel(output_file_path, index=False)

print(f'Confidence interval results saved to {output_file_path}')
