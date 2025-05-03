#!/bin/bash
set -e  # Exit the script if any command fails


# Read layer from the command line
LAYER=$1

# run the ontology parse
echo "Extract class representations from ontologies..."
cd millms-owlapi/out/artifacts/millms_owlapi_jar
java -jar millms-owlapi.jar v

# run the ground truth reference alignment python script
echo "Extract the ground trth reference alignments..."
cd ../../../..
python make-ref-align.py

# set up the folders
echo "Set up the folders for the experiment..."
python set-experiment.py

# translate the english to french
echo "Translate from English to French..."
python translate.py

# put the prompts through the SAE
echo "Push the prompts through the SAE..."
python run-saes.py --layer=$INPUT_FILE

# calculate all the vector distances
echo "Calculate the vector distances between concept arrays..."
python calculate-distances.py

# calculate the correlations
echo "Calculate the correlations and output to file..."
python calculate-correlation.py


echo "All scripts completed successfully."

