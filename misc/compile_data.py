import os
import pandas as pd

# Define the folder containing the CSV files
data_folder_path = os.getenv('DATA_FOLDER_PATH')
data_path = os.getenv('DATA_PATH')

# Create a list to store each DataFrame
dataframes = []

# Iterate through each file in the folder
for file_name in os.listdir(data_folder_path):
    if file_name.endswith(".csv"):  # Check if the file is a CSV file
        file_path = os.path.join(data_folder_path, file_name)
        try:
            # Read the CSV file into a DataFrame and append it to the list
            df = pd.read_csv(file_path)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

# Combine all the DataFrames into one
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    print("All CSV files have been combined into a single DataFrame.")
else:
    print("No CSV files found in the folder.")

# Optional: Save the combined DataFrame to a new CSV file
output_file = os.path.join(data_path, "ASDB_HNL1-2_id6-10_01-21-25.csv")
combined_df.to_csv(output_file, index=False)
print(f"Combined DataFrame saved to {output_file}")
