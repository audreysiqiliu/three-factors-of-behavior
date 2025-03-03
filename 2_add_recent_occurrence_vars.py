import numpy as np
import pandas as pd
import os

# Adding Recent Occurence Vars
def calculate_trials_since_by_day(df, iv_name, day_col):
    """
    Calculates the number of trials since the last occurrence of the specified event,
    resetting the count at the start of each new day.
    """
    new_col_name = f"TrialsSinceLast_{iv_name}_ByDay"
    cumulative_col_name = f"Cumulative_{iv_name}_ByDay"
    
    df[new_col_name] = np.nan
    df[cumulative_col_name] = 0
    
    for (user_id, day), group in df.groupby(['UserId', day_col], group_keys=False):
        for value in group[iv_name].dropna().unique():
            mask = group[iv_name] == value
            group.loc[mask, new_col_name] = group.loc[mask, 'TrialNumber'].diff()
            group.loc[mask, cumulative_col_name] = group.loc[mask, iv_name].notna().cumsum()
        df.loc[group.index, [new_col_name, cumulative_col_name]] = group[[new_col_name, cumulative_col_name]]

def calculate_color_match_details(df, illegal_color_columns, legal_color_columns, day_col):
    """
    Tracks the number of trials since the last color match, counts the cumulative occurrences of Illegal1Color,
    and adds a flag column for matches occurring in the current trial.
    """
    df['LastColorMatchTrial'] = np.nan
    df['TrialsSinceLast_ColorMatch_ByDay'] = np.nan
    df['Cumulative_Illegal1Color_AsLegal'] = 0
    df['Cumulative_Illegal1Color_AsIllegal'] = 0
    df['CurrentTrial_ColorMatch_Flag'] = False  # Adding a new flag column for current trial matches
    
    for (user_id, day), user_day_group in df.groupby(['UserId', day_col]):
        last_match_trial = None
        user_day_group.sort_values('TrialNumber', inplace=True)  # Ensure TrialNumber is sorted
        
        for index, row in user_day_group.iterrows():
            illegal_colors = {row[col] for col in illegal_color_columns if pd.notnull(row[col])}
            legal_colors = {row[col] for col in legal_color_columns if pd.notnull(row[col])}
            if 'Illegal1Color' in illegal_colors:
                df.at[index, 'Cumulative_Illegal1Color_AsIllegal'] += 1
            if 'Illegal1Color' in legal_colors:
                df.at[index, 'Cumulative_Illegal1Color_AsLegal'] += 1
            
            match_occurred = illegal_colors.intersection(legal_colors)
            df.at[index, 'CurrentTrial_ColorMatch_Flag'] = bool(match_occurred)  # Set flag if there's a match
            
            if match_occurred:
                last_match_trial = row['TrialNumber']
                df.at[index, 'LastColorMatchTrial'] = last_match_trial
        
        # Update TrialsSinceLast_ColorMatch_ByDay based on the last match trial
        if last_match_trial is not None:
            user_day_group['TrialsSinceLast_ColorMatch_ByDay'] = user_day_group['TrialNumber'] - last_match_trial
            df.loc[user_day_group.index, 'TrialsSinceLast_ColorMatch_ByDay'] = user_day_group['TrialsSinceLast_ColorMatch_ByDay'].fillna(method='ffill')

def copy_last_trial_result(df, iv_name):
    """
    Copies the last trial result and IllegalItems for each specific iv_name occurrence 
    using the 'TrialsSinceLast_X_ByDay' column.
    """
    trial_result_col = f"Last_TrialResult_for_{iv_name}"
    illegal_items_col = f"Last_IllegalItems_for_{iv_name}"
    since_col = f"TrialsSinceLast_{iv_name}_ByDay"
    
    df[trial_result_col] = np.nan
    df[illegal_items_col] = np.nan
    
    # Using shift and cumsum to avoid iterrows()
    for index, row in df.iterrows():
        if not np.isnan(row[since_col]):
            lookback_index = int(index - row[since_col])
            if lookback_index >= 0 and lookback_index < len(df):
                df.at[index, trial_result_col] = df.at[lookback_index, 'TrialResult']
                df.at[index, illegal_items_col] = df.at[lookback_index, 'IllegalItems']

# Read
data_path = os.getenv('DATA_PATH', './data')
output_path = os.getenv('OUTPUT_PATH', './output')

file_path = f"{output_path}/df_HNL_1-2.csv"
df = pd.read_csv(file_path, low_memory=False)
print(f"In step 2, reading file: {file_path}")


legal_color_columns = [col for col in df.columns if 'Legal' in col and 'Color' in col]
illegal_color_columns = [col for col in df.columns if 'Illegal' in col and 'Color' in col]

for iv_name in ['Illegal1Name', 'target_present', 'Type']:
    calculate_trials_since_by_day(df, iv_name, 'Day')

calculate_color_match_details(df, illegal_color_columns, legal_color_columns, 'Day')

for iv_name in ['Illegal1Name', 'target_present', 'Type','ColorMatch']:
    copy_last_trial_result(df, iv_name)

df.to_csv(f'{output_path}/df_HNL_1-2_recent_occurrence.csv', index=False)

# save dummy file to see if it is file saving or a problem with the analysis object
# add more print statements in general
