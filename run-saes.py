from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from huggingface_hub import login, hf_hub_download, notebook_login, HfApi
import numpy as np
import torch
#from sae_lens import SAE  # pip install sae-lens
import torch.nn as nn
from IPython.display import IFrame
import os
import re
import pandas as pd


allEnabled = True

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
output_folder = ex_folder + "/data" 

finputs = []

for filename in os.listdir(ex_folder):
    if "ontology_classes" in filename:
        finputs.append(ex_folder + "/" + filename)



filerepo = "google/gemma-scope-2b-pt-res"
api = HfApi()
files = api.list_repo_files(filerepo)
#print(files)


def file_exists(folder_path, filename):
    # Join the folder path and filename
    full_path = os.path.join(folder_path, filename)
    
    # Check if the file exists
    return os.path.isfile(full_path)



class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, d_sae):
        # Note that we initialise these to zeros because we're loading in pre-trained weights.
        # If you want to train your own SAEs then we recommend using blah
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon



torch.set_grad_enabled(False) # avoid blowing up mem

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    device_map='auto',
)

tokenizer =  AutoTokenizer.from_pretrained("google/gemma-2-2b")

# Prompt example
prompts = [[0,"prompt text"]]

topSAEs = []
SAEout = ""


input = 0
for csv_file in finputs:

    # Read the CSV file into a pandas DataFrame
    df_prompts = pd.read_csv(csv_file, header=None, names=['Concept', 'Prompt'])

    # Extract prompts and concepts
    prompts = df_prompts['Prompt'].tolist()
    concepts = df_prompts['Concept'].tolist()
    print(prompts)
    print(concepts)

    cf = 0
    for file in files:
        cf = cf + 1
        if 'layer_0' in file:
            if 'width_16k/average_l0_13' in file:
                if 'README' not in file:
                    print(str(cf) + " >> " + file)

                    path_to_params = hf_hub_download(repo_id=filerepo, filename=file, force_download=False)

                    p = 0
                    
                    for prompt in prompts:
                        if prompt != "":

                            print(prompt)
                            # Use the tokenizer to convert it to tokens. Note that this implicitly adds a special "Beginning of Sequence" or <bos> token to the start
                            inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to("mps")
                            print()
                            print("input text:")
                            print(inputs)

                            # Pass it in to the model and generate text
                            outputs = model.generate(input_ids=inputs, max_new_tokens=50)
                            print()
                            print("first output:")
                            print(tokenizer.decode(outputs[0]))


                            device = torch.device('mps')

                            params = np.load(path_to_params)
                            pt_params = {k: torch.from_numpy(v) for k, v in params.items()}

                            print()
                            print("shape - residual stream SAE on file " + file + ":")
                            print({k:v.shape for k, v in pt_params.items()})


                            print()
                            print("weights - residual stream SAE on file " + file + ":")
                            print(pt_params["W_enc"].norm(dim=0))

                            sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
                            print()
                            # amazonq-ignore-next-line
                            print(sae.load_state_dict(pt_params))


                            def gather_residual_activations(model, target_layer, inputs):
                                target_act = None
                                def gather_target_act_hook(mod, inputs, outputs):
                                    nonlocal target_act # make sure we can modify the target_act from the outer scope
                                    target_act = outputs[0]
                                    return outputs
                                handle = model.model.layers[target_layer].register_forward_hook(gather_target_act_hook)
                                _ = model.forward(inputs)
                                handle.remove()
                                return target_act


                            target_act = gather_residual_activations(model, 20, inputs)

                            sae.to(device)

                            sae_acts = sae.encode(target_act.to(torch.float32))
                            recon = sae.decode(sae_acts)

                            print()
                            print("checking the model looks sensible (decent chunk of variance):")
                            print(1 - torch.mean((recon[:, 1:] - target_act[:, 1:].to(torch.float32)) **2) / (target_act[:, 1:].to(torch.float32).var()))

                            print()
                            print("checking the model looks sensible (L0 should be around 70):")
                            activs = (sae_acts > 1).sum(-1)
                            print(activs)

                            values, inds = sae_acts.max(-1)

                            print()
                            print("highest activating values on each token:")
                            print(values)

                            print()
                            print("highest activating feature on each token:")
                            print(inds)

                            paired_tensor = torch.stack((activs, values), dim=2)

                            print(concepts[p])
                            print(file)
                            print(paired_tensor)

                            # Convert to NumPy and save
                            filen = csv_file.split("/")[-1].replace("ontology_classes_", "").replace(".csv", "")
                            np_tensor = paired_tensor.cpu().numpy()  # Move to CPU if necessary
                            np.save(output_folder + '/' + filen + '--' + str(p) + '--' + str(concepts[p]) + '--' + file.replace('/','-') + '.npy', np_tensor)

                            p = p + 1
                            print(p)

