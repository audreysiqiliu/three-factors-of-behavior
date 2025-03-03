import pandas as pd
import numpy as np
import os

# Paths from environment variables
data_file = os.getenv('DATA_FILE')
misc_path = os.path.dirname(__file__)  # Same directory as the script
base_input_path = os.getenv('DATA_PATH')
base_output_path = os.getenv('OUTPUT_PATH')

# Full path for input and output
input_full_path = os.path.join(base_input_path, data_file)
output_file_name = 'wColor_' + data_file
output_full_path = os.path.join(base_output_path, output_file_name)

# Print paths being used
print(f"Loading input file: {input_full_path}")
print(f"Loading legal mappings from: {os.path.join(misc_path, 'Combined_LegalId_Name_Color.csv')}")
print(f"Loading illegal mappings from: {os.path.join(misc_path, 'Combined_IllegalId_Name_Color.csv')}")
print(f"Output will be saved to: {output_full_path}")


# Load the combined legal and illegal data frames
combined_legal_df = pd.read_csv(os.path.join(misc_path, 'Combined_LegalId_Name_Color.csv'))
combined_illegal_df = pd.read_csv(os.path.join(misc_path, 'Combined_IllegalId_Name_Color.csv'))

# Function to apply color and name mapping using file paths for input and output
def apply_color_name_mapping(input_path, output_path):
    print("Loading master data file...")
    # Load the master data frame
    master_df = pd.read_csv(input_path)

    # Print the number of rows in the master data frame
    print(f"Number of rows in the master data frame: {len(master_df)}")

    print("Applying mappings for legal items...")
    # Create dictionaries for mapping IDs to names and colors
    legal_dict = combined_legal_df.set_index('LegalId')[['LegalName', 'Color']].to_dict('index')

    print("Applying mappings for illegal items...")
    illegal_dict = combined_illegal_df.set_index('IllegalId')[['IllegalName', 'Color']].to_dict('index')

    # Apply the mapping function for legal items
    for legal_id_column in [col for col in master_df.columns if 'Legal' in col and 'Id' in col]:
        name_column = legal_id_column.replace('Id', 'Name')
        color_column = legal_id_column.replace('Id', 'Color')
        
        if name_column not in master_df.columns:
            master_df[name_column] = master_df[legal_id_column].map(lambda x: legal_dict.get(x, {}).get('LegalName', np.nan))
        if color_column not in master_df.columns:
            master_df[color_column] = master_df[legal_id_column].map(lambda x: legal_dict.get(x, {}).get('Color', np.nan))

    # Apply the mapping function for illegal items
    for illegal_id_column in [col for col in master_df.columns if 'Illegal' in col and 'Id' in col]:
        name_column = illegal_id_column.replace('Id', 'Name')
        color_column = illegal_id_column.replace('Id', 'Color')
        
        if name_column not in master_df.columns:
            master_df[name_column] = master_df[illegal_id_column].map(lambda x: illegal_dict.get(x, {}).get('IllegalName', np.nan))
        if color_column not in master_df.columns:
            master_df[color_column] = master_df[illegal_id_column].map(lambda x: illegal_dict.get(x, {}).get('Color', np.nan))
    
    print("Saving the updated data to the output file...")
    # Save the updated DataFrame to the output file path
    master_df.to_csv(output_path, index=False, header=True)

    print(f"File successfully saved to: {output_path}")

# Ensure the output directory exists
if not os.path.exists(base_output_path):
    os.makedirs(base_output_path)
    print(f"Created output directory: {base_output_path}")

# Apply the mapping to the specified input file and save to the output file
apply_color_name_mapping(input_full_path, output_full_path)
