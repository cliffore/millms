
import os
import xml.etree.ElementTree as ET
import csv


# Folder containing RDF files
input_folder = 'input-data/ground-truth'
output_file = 'data-processing/all_mappings.csv'

# RDF & Alignment namespaces
ns = {
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'align': 'http://knowledgeweb.semanticweb.org/heterogeneity/alignment'
}

rows = []

# Iterate through all RDF files in the folder
for filename in os.listdir(input_folder):
    if filename.endswith('.rdf') or filename.endswith('.xml'):
        filepath = os.path.join(input_folder, filename)
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()

            maps = root.findall('.//align:map', ns)

            for map_elem in maps:
                cell = map_elem.find('align:Cell', ns)
                if cell is not None:
                    entity1_elem = cell.find('align:entity1', ns)
                    entity2_elem = cell.find('align:entity2', ns)
                    measure_elem = cell.find('align:measure', ns)
                    if entity1_elem is not None and entity2_elem is not None:
                        entity1 = entity1_elem.attrib.get(f'{{{ns["rdf"]}}}resource')
                        entity2 = entity2_elem.attrib.get(f'{{{ns["rdf"]}}}resource')
                        weighting = measure_elem.text
                        print(weighting)
                        if entity1 and entity2:
                            rows.append([filename.replace('.rdf', ''), entity1.split('#')[1], entity2.split('#')[1], str(weighting)])
                            print(rows) 
        except Exception as e:
            print(f"Failed to parse {filename}: {e}")

# Write all mappings to one CSV
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['file', 'entity1', 'entity2', 'weighting'])
    writer.writerows(rows)

print(f"Finished! Extracted {len(rows)} mappings from {input_folder} into {output_file}")