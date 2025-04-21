import csv
import re
import os

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

script_dir = os.path.dirname(os.path.abspath(__file__))

experiments_dir = os.path.join(script_dir, 'data-processing', 'experiments')

ex_folder = get_latest_ex_folder(experiments_dir)
print(ex_folder)

prompt_files = []

for filename in os.listdir(ex_folder):
    if "ontology_classes" in filename:
        prompt_files.append(ex_folder + "/" + filename)


mapping_csv_file = os.path.join(ex_folder, "all_mappings.csv")

cgt = 0
cp = 0

results = ""

with open(mapping_csv_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)

    for gt in reader:
        #print(gt)  
        cgt += 1

        for f in prompt_files:

            if cgt > 1:
                prompt_csv_file = os.path.join(ex_folder, f)

                mapping1_o = gt[0].split("-")[0]
                mapping2_o = gt[0].split("-")[1]
                mapping1_c = gt[1]
                mapping2_c = gt[2]
                

                with open(prompt_csv_file, newline='', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile)

                    for p in reader:
                        #print(p)
                        #print(gt)

                        prompt_o = p[0].split('-')[0]
                        prompt_c = p[0].split('-')[1]

                        if mapping1_o == prompt_o and mapping1_c == prompt_c:
                            cp += 1
                            res = mapping1_o + "," + mapping1_c + "," + p[1]
                            print(str(cp) + " >> " + res)
                            results = results + res + "\n"

                        if mapping2_o == prompt_o and mapping2_c == prompt_c:
                            cp += 1
                            res = mapping2_o + "," + mapping2_c + "," + p[1]
                            print(str(cp) + " >> " + res)
                            results = results + res + "\n"


outfile = mapping_csv_file.replace('.csv', '_with_prompts.csv')
with open(outfile, 'w') as file:
    file.write(results)
