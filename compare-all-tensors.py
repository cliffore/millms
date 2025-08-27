import os
import numpy as np
import re
import torch
import datetime

thisProcess = "compare-all-tensors.py"

def update_log(file_path: str, message: str):
    """Append a timestamped message to a log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}\n"
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(line)


update_log("experiment-log.log", thisProcess + ": start")


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

#input_folder = "/Users/Shared/Documents/personal/PhD/dev/year2/pramantha/gem1/mi5/ex/iterations/ex15"  
file_list = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]

c = 0

eng = []
fre = []
chi = []

for file_name in file_list:
    if 'DS_Store' not in file_name:
        if '-zh--' in file_name:
            chi.append(file_name)
        elif '-fr--' in file_name:
            fre.append(file_name)
        else:
            eng.append(file_name)

for fe in eng:

    engs = fe.split("--")
    eontoclass = engs[2]

    file1_path = os.path.join(data_folder, fe)
    t1 = np.load(file1_path)
    tensor1 = torch.tensor(t1[0][1:])  # shape (n1, 2)# [1:] = remove the first vector of the tensor because it represents an outlier - the BOS token


    for ff in fre: # english comparison with french

        fres = ff.split("--")
        fontoclass = fres[2]


        if eontoclass == fontoclass:

            print(eontoclass + ' >> ' + fe)
            print(fontoclass + ' >> ' + ff)


            file2_path = os.path.join(data_folder, ff)
            t2 = np.load(file2_path)
            tensor2 = torch.tensor(t2[0][1:]) # [1:] = remove the first vector of the tensor because it represents an outlier - the BOS token

            # Flatten and concatenate
            data = torch.cat((tensor1.view(-1, 2), tensor2.view(-1, 2)), dim=0)

            # Split into concept IDs and weights
            concept_ids = data[:, 0]
            weights = data[:, 1]

            # Use a dictionary to accumulate weights
            from collections import defaultdict

            sums = defaultdict(float)
            counts = defaultdict(int)

            for cid, weight in zip(concept_ids, weights):
                cid = int(cid.item())
                sums[cid] += weight.item()
                counts[cid] += 1

            # Compute average
            avg_weighted = sorted([(cid, sums[cid] / counts[cid]) for cid in sums])

            # Convert to tensor
            avg_tensor = torch.tensor(avg_weighted)

            print(avg_tensor)

            newFileName =  "weighted_avg--fve--" + ff.replace(".npy", "")
            np.save(os.path.join(data_folder, newFileName + ".npy"), avg_tensor)



    for fc in chi: # english comparison with chinese

        cres = fc.split("--")
        contoclass = cres[2]


        if eontoclass == contoclass:

            print(eontoclass + ' >> ' + fe)
            print(contoclass + ' >> ' + fc)


            file2_path = os.path.join(data_folder, fc)
            t2 = np.load(file2_path)
            tensor2 = torch.tensor(t2[0][1:]) # [1:] = remove the first vector of the tensor because it represents an outlier - the BOS token

            # Flatten and concatenate
            data = torch.cat((tensor1.view(-1, 2), tensor2.view(-1, 2)), dim=0)

            # Split into concept IDs and weights
            concept_ids = data[:, 0]
            weights = data[:, 1]

            # Use a dictionary to accumulate weights
            from collections import defaultdict

            sums = defaultdict(float)
            counts = defaultdict(int)

            for cid, weight in zip(concept_ids, weights):
                cid = int(cid.item())
                sums[cid] += weight.item()
                counts[cid] += 1

            # Compute average
            avg_weighted = sorted([(cid, sums[cid] / counts[cid]) for cid in sums])

            # Convert to tensor
            avg_tensor = torch.tensor(avg_weighted)

            print(avg_tensor)

            newFileName =  "weighted_avg--cve--" + fc.replace(".npy", "")
            np.save(os.path.join(data_folder, newFileName + ".npy"), avg_tensor)


update_log("experiment-log.log", thisProcess + ": end")