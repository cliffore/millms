# Mechanistic Interpretability - Large Language Models (MILLMs)

This repository contains data and application code to prepare, process and output a PhD experiment in mechanistic intepretability.

This repository is tailored to a specific set of inputs and ground-truth mappings (taken from the ontology matching task here: https://oaei.ontologymatching.org/2024/conference/index.html), but it can be amended to include any sources. The raw code is included here and there are no licence restrictions.


---

## 🛠️ Explanation

1: Java executable millm-owlapi-1 is executed via the command line and takes a folder name which is the location of the raw ontology.owl files. The Java process takes each owl ontology, parses it using the OWLAPI library and extracts class relationships and properties, and generates a CSV file representing the structure and semantics of the ontology classes. The parameter 'level' can be supplied as either 's' or 'v' which guides the output to be a summary (s) or verbose (v). The IntelliJ Java maven project code is included in the repository (/millm-owlapi-1).

The output of this process should be a single csv file in the format {owl ontology ref-owl class name},{text representation of the owl class}.

2: Python script make-ref-align.py is executed via the command line (e.g. python make-ref-align.py) and it will look for source mapping files in the input-data/ground-truth folder. This script will output a single file called all_mappings.csv and it will be in the format {source mapping file},{class 1},{class 2},{match weighting}.

3: Python script ....py is executed via the command line and 


---

## 🚀 Usage



### Requirements

1: This experiment has been run successfully on Mac OS running Sequioa 15.3.2

2: Python 3.12.7 and the following libraries:

    transformers
    huggingface_hub
    numpy
    torch
    sae_lens
    torch.nn
    IPython.display
    re
    pandas
    sklearn
    scipy

3: Java 24 and the following libraries:

    OWLAPI (the included Intellij project uses Maven to manage this library)


Any other configuration may work, but cannot be supported in advance.




### Steps

To process this experiment using the included ontologies:


1: Clone this repository, e.g. s

    gh repo clone https://github.com/cliffore/millm.git
    cd millms


2: Execute the Java program, e.g.

    cd millms-owlapi/out/artifacts/millms_owlapi_jar
    java -jar millms-owlapi.jar v


3: Execute the Python script to create the groung truth mappings, e.g. 
    
    python make-ref-align.py


4: Execute the Python script to create a new experiment folder and move the data files into it, e.g.

    python set-experiment.py


5: Execute the python script to convert each prompt in the experiment source csv files into French:

    python translate.py


6: Execute the Python script to enter each prompt through the suite of SAEs, e.g.
    
    python run-saes.py

This script can be executed many times and will take the highest number experiment folder.


7: Execute the python script that takes each tensor from the SAE and creates a set of comparisons, such as weighted average vector, simple vector:

    python compare-all-tensors.py


8: Execute the python script that takes the averages and then calculates the distances between vectors, e.g.

    python calculate-distances.py
    
    
9: Execute the python script that takes the averages and then correlates them and outputs a measure of the relationship, e.g.

    python calculate-correlation.py




### Making changes

1: To update this process to use a different source, change the Java code to parse the source owl files as needed (and build the jar artifact again if needed) and also change the make-ref-align.py script that parses the particular format of the ground truth raw data.
2: To run this code on a non-Apple Silicon machine, change the run-saes.py script and replace "mps" with whichever Pytorch chip needed, e.g. "cude" or just "cpu"

