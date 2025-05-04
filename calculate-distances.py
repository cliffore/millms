
import os
import re
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.spatial.distance import euclidean
import torch
#from scipy.stats import wasserstein_distance



def resize_vector(vec, target_length):
    if len(vec) == target_length:
        return vec
    elif len(vec) > target_length:
        return vec[:target_length]
    else:
        return np.pad(vec, (0, target_length - len(vec)))
    
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
data_folder = ex_folder + "/data"
print(data_folder)


# Paths (update as needed)
#npy_folder = "/path/to/npy/folder"
ground_truth_file = ex_folder + "/all_mappings.csv"
output_csv = ex_folder + "/distances_with_groundtruth.csv"

# Load all .npy files and map concept name to tensor
concept_tensors = {}
for fname in os.listdir(data_folder):
    if fname.endswith(".npy") and  "weighted_avg" in fname:
        concept = fname.replace(".npy", "")  # e.g., "edas-Author"
        tensor = np.load(os.path.join(data_folder, fname))
        concept_tensors[concept] = tensor

# Load ground truth into a dictionary for fast lookup
# Format: {("edas-Author", "ekaw-Paper_Author"): 1.0}
ground_truth = {}
df_gt = pd.read_csv(ground_truth_file, header=1, names=["file", "entity1", "entity2", "weighting"])

for _, row in df_gt.iterrows():
    src1, src2 = row["file"].split("-")
    c1 = f"{src1}-{row['entity1']}"
    c2 = f"{src2}-{row['entity2']}"
    ground_truth[(c1, c2)] = int(row["weighting"])
    ground_truth[(c2, c1)] = int(row["weighting"])  # make symmetric

# Calculate distances between all unique pairs
results = []

def tensor_to_dense(tensor, vocab):
    dense = torch.zeros(len(vocab))
    id_to_index = {cid: i for i, cid in enumerate(vocab)}
    for row in tensor:
        cid = int(row[0].item())
        weight = row[1].item()
        dense[id_to_index[cid]] = weight
    return dense




cnt = 0
for (concept1, tensor1), (concept2, tensor2) in combinations(concept_tensors.items(), 2):
    cnt = cnt + 1
    if cnt % 100 == 0:
        print("English and French average: " + str(cnt))
    if tensor1.shape != tensor2.shape:
        target_len = max(len(t) for t in concept_tensors.values())
        tensor1 = resize_vector(tensor1, target_len)
        tensor2 = resize_vector(tensor2, target_len)

    
    # Suppose you have two tensors: tensorA and tensorB
    # Build combined vocabulary
    vocab = sorted(set(tensor1[:, 0].tolist()) | set(tensor2[:, 0].tolist()))

    denseA = tensor_to_dense(tensor1, vocab)
    denseB = tensor_to_dense(tensor2, vocab)

    # Euclidean distance
    euclidean = torch.norm(denseA - denseB)
    # Cosine similarity
    cosine = torch.nn.functional.cosine_similarity(denseA.unsqueeze(0), denseB.unsqueeze(0)).item()

    '''
    a_ids = tensor1[:, 0]
    a_weights = tensor1[:, 1]
    b_ids = tensor2[:, 0]
    b_weights = tensor2[:, 1]
    
    emd = wasserstein_distance(
        u_values=a_ids, v_values=b_ids,
        u_weights=a_weights, v_weights=b_weights
    )
    '''

    dist = cosine
    #dist = euclidean(tensor1, tensor2)
    c1 = concept1.split('--')[4]
    c2 = concept2.split('--')[4]
    gt = ground_truth.get((c1, c2), 0)
    results.append([c1, c2, dist, gt])


# Write to CSV
output_df = pd.DataFrame(results, columns=["concept1", "concept2", "distance", "ground_truth"])
output_df.to_csv(output_csv, index=False)

print(f"Saved English and French average distances and averages ground truth to: {output_csv}")




# process english data

# Load all .npy files and map concept name to tensor
concept_tensors_2 = {}
for fname in os.listdir(data_folder):
    if fname.endswith(".npy") and '-fr--' not in fname and  "weighted_avg" not in fname:
        concept = fname.replace(".npy", "")  # e.g., "edas-Author"
        tensor = np.load(os.path.join(data_folder, fname))
        concept_tensors_2[concept] = tensor


cnt = 0
for (concept1, tensor1), (concept2, tensor2) in combinations(concept_tensors_2.items(), 2):
    cnt = cnt + 1
    if cnt % 100 == 0:
        print("English only distances: " + str(cnt))
    if tensor1.shape != tensor2.shape:
        target_len = max(len(t) for t in concept_tensors_2.values())
        tensor1 = resize_vector(tensor1, target_len)
        tensor2 = resize_vector(tensor2, target_len)

    #print(tensor1)
    #print(tensor2)

    # Flatten to shape (N, 2)
    tensor1 = tensor1.reshape(-1, 2)
    tensor2 = tensor2.reshape(-1, 2)

    # Step 1: Build unified vocabulary of concept IDs
    ids1 = set(tensor1[:, 0].tolist())
    ids2 = set(tensor2[:, 0].tolist())
    vocab = sorted(ids1.union(ids2))

    # Step 2: Function to map tensor to dense vector
    def tensor_to_dense(tensor, vocab):
        dense = torch.zeros(len(vocab))
        id_to_index = {cid: i for i, cid in enumerate(vocab)}
        for row in tensor:
            cid = int(row[0].item())
            weight = row[1].item()
            dense[id_to_index[cid]] += weight  # use += in case of duplicate IDs
        return dense

    # Step 3: Build dense vectors
    dense1 = tensor_to_dense(tensor1, vocab)
    dense2 = tensor_to_dense(tensor2, vocab)

    cosine = torch.nn.functional.cosine_similarity(dense1.unsqueeze(0), dense2.unsqueeze(0)).item()

    dist = cosine
    c1 = concept1.split('--')[2]
    c2 = concept2.split('--')[2]
    gt = ground_truth.get((c1, c2), 0)
    results.append([c1, c2, dist, gt])


# Write to CSV
output_df = pd.DataFrame(results, columns=["concept1", "concept2", "distance", "ground_truth"])
output_df.to_csv(output_csv.replace('.csv', '--en.csv'), index=False)

print(f"Saved English distances and ground truth to: {output_csv.replace('.csv', '--en.csv')}")






# process french data

# Load all .npy files and map concept name to tensor
concept_tensors_2 = {}
for fname in os.listdir(data_folder):
    if fname.endswith(".npy") and '-fr--' in fname and  "weighted_avg" not in fname:
        concept = fname.replace(".npy", "")  # e.g., "edas-Author"
        tensor = np.load(os.path.join(data_folder, fname))
        concept_tensors_2[concept] = tensor


cnt = 0
for (concept1, tensor1), (concept2, tensor2) in combinations(concept_tensors_2.items(), 2):
    cnt = cnt + 1
    if cnt % 100 == 0:
        print("French only distances: " + str(cnt))
    if tensor1.shape != tensor2.shape:
        target_len = max(len(t) for t in concept_tensors_2.values())
        tensor1 = resize_vector(tensor1, target_len)
        tensor2 = resize_vector(tensor2, target_len)

    #print(tensor1)
    #print(tensor2)

    # Flatten to shape (N, 2)
    tensor1 = tensor1.reshape(-1, 2)
    tensor2 = tensor2.reshape(-1, 2)

    # Step 1: Build unified vocabulary of concept IDs
    ids1 = set(tensor1[:, 0].tolist())
    ids2 = set(tensor2[:, 0].tolist())
    vocab = sorted(ids1.union(ids2))

    # Step 2: Function to map tensor to dense vector
    def tensor_to_dense(tensor, vocab):
        dense = torch.zeros(len(vocab))
        id_to_index = {cid: i for i, cid in enumerate(vocab)}
        for row in tensor:
            cid = int(row[0].item())
            weight = row[1].item()
            dense[id_to_index[cid]] += weight  # use += in case of duplicate IDs
        return dense

    # Step 3: Build dense vectors
    dense1 = tensor_to_dense(tensor1, vocab)
    dense2 = tensor_to_dense(tensor2, vocab)

    cosine = torch.nn.functional.cosine_similarity(dense1.unsqueeze(0), dense2.unsqueeze(0)).item()

    dist = cosine
    c1 = concept1.split('--')[2]
    c2 = concept2.split('--')[2]
    gt = ground_truth.get((c1, c2), 0)
    results.append([c1, c2, dist, gt])


# Write to CSV
output_df = pd.DataFrame(results, columns=["concept1", "concept2", "distance", "ground_truth"])
output_df.to_csv(output_csv.replace('.csv', '--fr.csv'), index=False)

print(f"Saved French distances and ground truth to: {output_csv.replace('.csv', '--fr.csv')}")



