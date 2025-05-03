import pandas as pd
from scipy.stats import pointbiserialr
import re
import os

OutputText = ""

def get_latest_ex_folder(folder_path):
    pattern = re.compile(r"^ex_(\d+)$")
    max_id = -1
    latest_folder = None

    for name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, name)
        if os.path.isdir(full_path):
            match = pattern.match(name)
            if match:
                current_id = int(match.group(1))
                if current_id > max_id:
                    max_id = current_id
                    latest_folder = full_path

    return latest_folder

# Run it
ex_folder = get_latest_ex_folder("data-processing/experiments")
print(ex_folder)


# ==== CONFIGURATION ====
csv_path = ex_folder + "/distances_with_groundtruth.csv"

# === LOAD CSV (no header) ===
df = pd.read_csv(csv_path, header=1)

# === Extract last two columns ===
df = df.iloc[:, -2:]  # Get last two columns only
df.columns = ['value', 'target']

# === Print original counts ===
OutputText = "English and French averages:"
OutputText = OutputText + "============================================="
OutputText = OutputText + "Original Target value counts:"
OutputText = OutputText + str(df['target'].value_counts())

# === Ensure target is binary ===
if not set(df['target']).issubset({0, 1}):
    raise ValueError("Target variable must be binary (0 or 1).")

# === Balance the dataset by downsampling class 0 ===
n_positives = df[df['target'] == 1].shape[0]
positives = df[df['target'] == 1]
negatives = df[df['target'] == 0]
negatives_sampled = negatives.sample(n=n_positives, random_state=42)

# Combine and shuffle
df_balanced = pd.concat([positives, negatives_sampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# === Print balanced counts ===
OutputText = OutputText + "Balanced Target value counts:"
OutputText = OutputText + str(df_balanced['target'].value_counts())

# === Calculate Point-Biserial Correlation on balanced data ===
corr, p_value = pointbiserialr(df_balanced['target'], df_balanced['value'])

# === Output ===
OutputText = OutputText + "Point-Biserial Correlation Analysis for average of English and French"
OutputText = OutputText + f"Correlation Coefficient: {corr:.4f}"
OutputText = OutputText + f"P-value: {p_value:.6f}"

if p_value < 0.05:
    OutputText = OutputText + "There is a statistically significant correlation between the numeric value and the target."
else:
    OutputText = OutputText + "No statistically significant correlation found."


print(OutputText)


# ==== CONFIGURATION ====
csv_path = ex_folder + "/distances_with_groundtruth--en.csv"

# === LOAD CSV (no header) ===
df = pd.read_csv(csv_path, header=1)

# === Extract last two columns ===
df = df.iloc[:, -2:]  # Get last two columns only
df.columns = ['value', 'target']

# === Print original counts ===
OutputText = OutputText + "\n"
OutputText = OutputText + "\n"
OutputText = OutputText + "English only correlations:"
OutputText = OutputText + "======================================="
OutputText = OutputText + "Original Target value counts:"
OutputText = OutputText + str(df['target'].value_counts())

# === Ensure target is binary ===
if not set(df['target']).issubset({0, 1}):
    raise ValueError("Target variable must be binary (0 or 1).")

# === Balance the dataset by downsampling class 0 ===
n_positives = df[df['target'] == 1].shape[0]
positives = df[df['target'] == 1]
negatives = df[df['target'] == 0]
negatives_sampled = negatives.sample(n=n_positives, random_state=42)

# Combine and shuffle
df_balanced = pd.concat([positives, negatives_sampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# === Print balanced counts ===
OutputText = OutputText + "Balanced Target value counts:"
OutputText = OutputText + str(df_balanced['target'].value_counts())

# === Calculate Point-Biserial Correlation on balanced data ===
corr, p_value = pointbiserialr(df_balanced['target'], df_balanced['value'])

# === Output ===
OutputText = OutputText + "Point-Biserial Correlation Analysis for English"
OutputText = OutputText + f"Correlation Coefficient: {corr:.4f}"
OutputText = OutputText + f"P-value: {p_value:.6f}"

if p_value < 0.05:
    OutputText = OutputText + "There is a statistically significant correlation between the numeric value and the target."
else:
    OutputText = OutputText + "No statistically significant correlation found."


print(OutputText)
print()


# ==== CONFIGURATION ====
csv_path = ex_folder + "/distances_with_groundtruth--fr.csv"

# === LOAD CSV (no header) ===
df = pd.read_csv(csv_path, header=1)

# === Extract last two columns ===
df = df.iloc[:, -2:]  # Get last two columns only
df.columns = ['value', 'target']

# === Print original counts ===
OutputText = OutputText + "\n"
OutputText = OutputText + "\n"
OutputText = OutputText + "French only correlations:"
OutputText = OutputText + "======================================="
OutputText = OutputText + "Original Target value counts:"
OutputText = OutputText + str(df['target'].value_counts())

# === Ensure target is binary ===
if not set(df['target']).issubset({0, 1}):
    raise ValueError("Target variable must be binary (0 or 1).")

# === Balance the dataset by downsampling class 0 ===
n_positives = df[df['target'] == 1].shape[0]
positives = df[df['target'] == 1]
negatives = df[df['target'] == 0]
negatives_sampled = negatives.sample(n=n_positives, random_state=42)

# Combine and shuffle
df_balanced = pd.concat([positives, negatives_sampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# === Print balanced counts ===
OutputText = OutputText + "Balanced Target value counts:"
OutputText = OutputText + str(df_balanced['target'].value_counts())

# === Calculate Point-Biserial Correlation on balanced data ===
corr, p_value = pointbiserialr(df_balanced['target'], df_balanced['value'])

# === Output ===
OutputText = OutputText + "Point-Biserial Correlation Analysis for French"
OutputText = OutputText + f"Correlation Coefficient: {corr:.4f}"
OutputText = OutputText + f"P-value: {p_value:.6f}"

if p_value < 0.05:
    OutputText = OutputText + "There is a statistically significant correlation between the numeric value and the target."
else:
    OutputText = OutputText + "No statistically significant correlation found."


# Define the output file path
output_file = "final_correlations.txt"

# Write the text to the file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(OutputText)

print(f"Text written to {output_file}")
