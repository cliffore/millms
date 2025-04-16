import os
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import re


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

data_folder = ex_folder + "/data"


# === CONFIGURATION ===
output_file = ex_folder + "/all_distances_optimized.csv"

# === Load and filter tensors ===
concepts = {}
params_map = {}

for fname in os.listdir(data_folder):
    if fname.endswith(".npy"):
        if "weighted_avg" in fname:
            parts = fname.replace(".npy", "").split("--")
            class_name = parts[2]
            concept_name = class_name
            tensor = np.load(os.path.join(data_folder, fname))
            concepts[concept_name] = tensor
            params_map[concept_name] = (class_name, avtype, vs, hr)

# === Pad all tensors to the same shape and flatten ===
concept_names = sorted(concepts.keys())
tensors = [concepts[name] for name in concept_names]
max_len = max(t.shape[0] for t in tensors)
tensor_matrix = np.array([
    np.pad(t, (0, max_len - t.shape[0])) for t in tensors
])

# === Compute pairwise Euclidean distances ===
distance_matrix = pairwise_distances(tensor_matrix, metric='euclidean')

# === Generate output rows (avoid self-comparison & same-class pairs) ===
rows = []
for i, name1 in enumerate(concept_names):
    class1 = params_map[name1][0]
    for j in range(i + 1, len(concept_names)):
        name2 = concept_names[j]
        class2 = params_map[name2][0]
        if class1 != class2:
            dist = distance_matrix[i, j]
            row = [
                ",".join(params_map[name1]),
                ",".join(params_map[name2]),
                dist
            ]
            rows.append(row)

# === Save results to CSV ===
df = pd.DataFrame(rows, columns=["params1", "params2", "euclidean_distance"])
df.to_csv(output_file, index=False)

print(f"Euclidean distance comparisons complete. Output saved to:\n{output_file}")
