from googletrans import Translator
import pandas as pd
import asyncio
from googletrans import Translator
import os
import re


async def translate_to_french(word):
    translator = Translator()
    translation = await translator.translate(word, src="en", dest="fr")
    return translation.text


print("English to French Translator")


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

finputs = []

for filename in os.listdir(ex_folder):
    if "ontology_classes" in filename:
        finputs.append(ex_folder + "/" + filename)



for csv_file in finputs:

    df_prompts = pd.read_csv(csv_file, header=None, names=['Concept', 'Prompt'])

    # Ensure order by resetting index
    df_prompts = df_prompts.reset_index()

    # Extract prompts and concepts in their original order
    concepts = df_prompts['Concept'].tolist()
    prompts = df_prompts['Prompt'].tolist()

    output = ""
    cn = 0
    for p in prompts:
        
        print(str(cn) + " ;; " + concepts[cn] + " ;; " + p)
        french_word = asyncio.run(translate_to_french(p))
        print(french_word)
        output = output + concepts[cn] + "," + french_word.replace(","," ") + "\n"
        cn = cn + 1


    outfile = csv_file.replace(".csv", "-fr.csv")
    with open(outfile, 'w') as file:
        file.write(output)

