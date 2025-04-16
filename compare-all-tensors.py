import os
import numpy as np
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

#input_folder = "/Users/Shared/Documents/personal/PhD/dev/year2/pramantha/gem1/mi5/ex/iterations/ex15"  
file_list = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]

c = 0

eng = []
fre = []

for file_name in file_list:
    if '--fr' in file_name:
        fre.append(file_name)
    else:
        eng.append(file_name)


for fe in eng:

    engs = fe.split("--")
    eontoclass = engs[2]


    file1_path = os.path.join(data_folder, fe)
    t1 = np.load(file1_path)
    tensor1 = t1[0][1:]  # shape (n1, 2)# [1:] = remove the first vector of the tensor because it represents an outlier - the BOS token


    for ff in fre: # english comparison with french

        fres = ff.split("--")
        fontoclass = fres[2]


        if eontoclass == fontoclass:

            print(eontoclass + ' >> ' + fe)
            print(fontoclass + ' >> ' + ff)


            file2_path = os.path.join(data_folder, ff)
            t2 = np.load(file2_path)
            tensor2 = t2[0][1:] # [1:] = remove the first vector of the tensor because it represents an outlier - the BOS token


            # Step 2: Build dictionaries {class_id: weight}
            dict1 = {int(row[0]): row[1] for row in tensor1}
            dict2 = {int(row[0]): row[1] for row in tensor2}

            # Step 3: Simple average — only for matching class IDs
            '''common_classes = sorted(set(dict1.keys()) & set(dict2.keys()))

            if not common_classes:
                print("No common classes between the two tensors (after removing first row).")
                simple_avg_vector = np.array([])
            else:
                simple_avg_vector = np.array([
                    (dict1[cls] + dict2[cls]) / 2 for cls in common_classes
                ])
            '''

            # Step 4: Weighted average — use all class IDs
            all_classes = sorted(set(dict1) | set(dict2))
            weighted_avg_vector = []

            for cls in all_classes:
                w1 = dict1.get(cls, 0.0)
                w2 = dict2.get(cls, 0.0)
                if w1 > 0 and w2 > 0:
                    weighted_avg = (cls * w1 + cls * w2) / (w1 + w2)
                elif w1 > 0:
                    weighted_avg = cls
                else:
                    weighted_avg = cls
                weighted_avg_vector.append(weighted_avg)

            weighted_avg_vector = np.array(weighted_avg_vector)

            newFileName =  "weighted_avg--fve--" + fe.replace(".npy", "")
            np.save(os.path.join(data_folder, newFileName + ".npy"), weighted_avg_vector)

            #newFileName = "sa--fve--" + fe
            #np.save(os.path.join(data_folder + "_means", newFileName + ".npy"), simple_avg_vector)



