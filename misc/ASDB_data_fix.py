import pandas as pd

input_path = "/CCAS/groups/mitroffgrp/Audrey/three_factors_final_prereg/data/ASDB_HNL1-2_id6-10_02-12-25_demo.csv"
output_path = "/CCAS/groups/mitroffgrp/Audrey/three_factors_final_prereg/data/ASDB_HNL1-2_id6-10_02-12-25_demo_fixed.csv"

#input_path = "/Users/g39836381/Library/CloudStorage/Box-Box/ASDB_pull/ASDB_HNL1-2_id6-10_02-12-25_demo.csv"
#output_path = "/Users/g39836381/Library/CloudStorage/Box-Box/ASDB_pull/ASDB_HNL1-2_id6-10_02-12-25_demo_fixed.csv"

# Read CSV with flexible handling of inconsistent rows
df = pd.read_csv(input_path, header=None, dtype=str, on_bad_lines='skip')

print(f"Original shape: {df.shape}")  # Check the number of columns

def fix_row_length(row, target_length=193):
    row_list = row.tolist()
    if len(row_list) < target_length:
        return row_list + [''] * (target_length - len(row_list))  # Pad missing values
    else:
        return row_list[:target_length]  # Trim extra columns

df_fixed = df.apply(lambda x: fix_row_length(x), axis=1)
df_fixed = pd.DataFrame(df_fixed.tolist())  # Convert back to DataFrame

df_fixed.to_csv(output_path, index=False, header=False)

print(f"Fixed CSV saved: {output_path}")

# you have to count the number of columns in row 1 first to get the 193 number