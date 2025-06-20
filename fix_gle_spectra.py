import os
import pandas as pd
import glob
import re
from datetime import datetime

# Directory containing the GLE Spectra files
spectra_dir = "GLE Spectra"

# Find all GLE*_SEP_Spectra.csv files (excluding the ones with _original suffix)
gle_files = [f for f in glob.glob(os.path.join(spectra_dir, "GLE*_SEP_Spectra.csv")) 
             if "_original" not in f]

for file_path in gle_files:
    print(f"Processing: {file_path}")
    
    # Get the corresponding original file path
    original_file = file_path.replace(".csv", "_original.csv")
    
    # Read the original file to get the date/time format
    with open(original_file, 'r') as f:
        original_data = [line.strip().split(',') for line in f.readlines()]
    
    # Extract the header from the original file
    original_header = original_data[0]
    
    # Extract the date information from the original file
    date_pattern = re.compile(r'(\d+/\d+/\d+)')
    base_date = None
    for row in original_data[1:]:
        if row and len(row) > 0:
            match = date_pattern.search(row[0])
            if match:
                base_date = match.group(1)
                break
    
    if not base_date:
        print(f"Could not extract base date from {original_file}")
        continue
    
    # Read the current file
    with open(file_path, 'r') as f:
        current_data = [line.strip().split(',') for line in f.readlines()]
    
    # Filter out empty first row if present
    if not current_data[0][0]:
        current_data = current_data[1:]
    
    # Create new data preserving all original values
    new_data = []
    
    # Add the header row
    header = original_header.copy()
    # If the current data has more columns than the original, add generic column names
    while len(header) < len(current_data[0]):
        header.append(f"column_{len(header)}")
    new_data.append(header)
    
    # Process each row of data
    for i, row in enumerate(current_data):
        time_range = row[0].strip()
        
        # Handle the (+1) notation for next day
        next_day = False
        if "(+1)" in time_range:
            next_day = True
            time_range = time_range.replace("(+1)", "").strip()
        
        # Parse the original date
        base_date_obj = datetime.strptime(base_date, "%d/%m/%Y")
        
        # Adjust for next day if needed
        if next_day:
            new_date = f"{(base_date_obj.day + 1):02d}/{base_date_obj.month:02d}/{base_date_obj.year}"
        else:
            new_date = base_date
        
        # Format the time column as in the original file
        formatted_time = f"{new_date} {time_range}"
        
        # Create the new row, starting with the formatted time
        new_row = [formatted_time]
        
        # Add the J_0, gamma, and d_gamma values from current file, ensuring regular decimal notation
        for j in range(1, 4):
            if j < len(row):
                # Convert scientific notation to regular decimal
                try:
                    if 'E' in row[j] or 'e' in row[j]:
                        value = float(row[j])
                        # Use same number of decimal places as in original
                        if j == 1:  # J_0 value should be an integer with .0 
                            new_row.append(f"{value:.1f}")
                        else:  # gamma and d_gamma
                            new_row.append(f"{value:.2f}")
                    else:
                        new_row.append(row[j])
                except ValueError:
                    new_row.append(row[j])
            else:
                new_row.append("")
        
        # Add the remaining columns from the current file
        for j in range(4, len(row)):
            if j < len(row):
                new_row.append(row[j])
        
        new_data.append(new_row)
    
    # Write the fixed data back to the file
    with open(file_path, 'w', newline='') as f:
        for row in new_data:
            f.write(','.join(row) + '\n')
    
    print(f"Fixed: {file_path}")

print("All GLE Spectra files have been fixed.") 