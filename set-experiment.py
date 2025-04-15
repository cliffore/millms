import os
import shutil
import re

# 🔧 Set these paths
source_folder = "data-processing"  # where CSVs are found
destination_root = "data-processing/experiments"  # where ex_* folders are created

# Find all ex_<number> folders in the destination root
existing_ids = []
pattern = re.compile(r"ex_(\d+)$")

for name in os.listdir(destination_root):
    match = pattern.match(name)
    if match:
        existing_ids.append(int(match.group(1)))

# Determine the next folder name
next_id = max(existing_ids, default=0) + 1
new_folder_name = f"ex_{next_id}"
new_folder_path = os.path.join(destination_root, new_folder_name)

# Create the new folder
os.makedirs(new_folder_path, exist_ok=True)

# Move CSV files
moved_files = 0
for file_name in os.listdir(source_folder):
    if file_name.endswith(".csv"):
        src_path = os.path.join(source_folder, file_name)
        dest_path = os.path.join(new_folder_path, file_name)
        shutil.move(src_path, dest_path)
        moved_files += 1

print(f"Moved {moved_files} CSV file(s) to: {new_folder_path}")