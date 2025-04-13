import os
import shutil
from collections import defaultdict

# Set up directories
data_dir = "dataset"
clean_dir = "filtered_images"

# Create output directory if needed
os.makedirs(clean_dir, exist_ok=True)

# Track how many images we have for each roll number
student_counts = defaultdict(int)

# Go through all files in the data directory
for filename in os.listdir(data_dir):
    file_path = os.path.join(data_dir, filename)
    
    # Skip directories
    if not os.path.isfile(file_path):
        continue
        
    # Keep only files with "i2" identifier
    if "i2" in filename:
        # Extract the roll number (starts with "i2" and continues until non-alphanumeric)
        roll_id = ""
        start_idx = filename.find("i2")
        
        for char in filename[start_idx:]:
            if char.isalnum():
                roll_id += char
            else:
                break
                
        # Update count and create new filename
        student_counts[roll_id] += 1
        new_name = f"{roll_id}_{student_counts[roll_id]}.jpg"
        
        # Move and rename file
        shutil.move(file_path, os.path.join(clean_dir, new_name))
        print(f"Moved: {filename} → {new_name}")
    else:
        # Remove files without "i2" marker
        os.remove(file_path)
        print(f"Removed: {filename}")

print(f"Cleaned dataset: {sum(student_counts.values())} images kept")