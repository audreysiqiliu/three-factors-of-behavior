import numpy as np
import pandas as pd
import os

def categorize_target_condition(df):
    """ Categorizes trials based on whether a target is present or absent. """
    df['target_present'] = (df['IllegalItems'] > 0).astype(int)
    df['target_absent'] = (df['IllegalItems'] == 0).astype(int)

def assign_trial_numbers(df):
    """ Adds a trial number for each user, which increments per trial. """
    df['TrialNumber'] = df.groupby('UserId').cumcount() + 1

def categorize_trial_results(df):
    """
    Categorizes each trial based on performance into hits, false alarms, correct rejections, and misses.
    - trials with hits + false alarms are false alarms
    - trials with false alarms + misses are false alarms
    - note: false alarm trials can either be target present or absent
    """
    # Condition definitions
    condition_hit = (df['IllegalItems'] > 0) & (df['IllegalItemsMarked'] == df['IllegalItems'])
    condition_false_alarm = (df['UniqueTaps'] > df['IllegalItems'])
    condition_correct_rejection = ((df['IllegalItems'] == 0) | df['IllegalItems'].isna()) & ((df['LegalItemsMarked'] == 0) | df['LegalItemsMarked'].isna())
    condition_miss = (df['IllegalItems'] > 0) & (df['IllegalItemsMarked'] < df['IllegalItems'])

    # Default category set to incorrect, though this should be overwritten by specific conditions
    df['TrialResult'] = 'Incorrect'

    # Assigning categories based on conditions
    df.loc[condition_hit, 'TrialResult'] = 'Hit'
    df.loc[condition_correct_rejection, 'TrialResult'] = 'Correct Rejection'
    df.loc[condition_miss, 'TrialResult'] = 'Miss'
    df.loc[condition_false_alarm, 'TrialResult'] = 'False Alarm'  # Evaluate false alarm last to ensure it catches any excess taps

def calculate_response_times(df):
    """ Calculates and cleans up the response times based on trial results. """
    conditions = [
        df['TrialResult'] == 'Hit',
        df['TrialResult'] == 'Correct Rejection',
        df['TrialResult'] == 'False Alarm',
        df['TrialResult'] == 'Miss'
    ]
    choices = [
        df['Illegal1MarkTime'],
        df['TimeInScanner'],
        df['FirstLegalTapTime'],
        df['TimeInScanner']
            
    ]
    df['RT'] = np.select(conditions, choices, default=np.nan)
    filter_out_invalid_rt(df)

def filter_out_invalid_rt(df):
    """ Filters out invalid response times by row and User IDs associated with RT values that indicate faulty RT recording. """
    # Identify user IDs with invalid RTs marked as 0 or 1
    user_ids_with_invalid_rt = df[df['RT'].isin([0, 1])]['UserId'].unique()
    
    # Drop rows for users with these invalid RTs
    df.drop(df[df['UserId'].isin(user_ids_with_invalid_rt)].index, inplace=True)
    
    # Set RTs outside of a realistic range (less than 250ms or more than 10000ms) to NaN
    df.loc[(df['RT'] <= 250) | (df['RT'] > 10000), 'RT'] = np.nan
    
    # Print the count of unique users with invalid RTs for logging
    print(f"Number of users with invalid RTs: {len(user_ids_with_invalid_rt)}")

def log_response_times(df):
    """ Calculates the logarithm of the response times. """
    df['log_RT'] = np.log(df['RT'])

def calculate_statistics(df):
    """ Calculates and prints statistics related to response times, trial accuracy, and dataset composition. """
    nan_RT_count = df['RT'].isna().sum()
    num_hits = df['TrialResult'].value_counts().get('Hit', 0)
    num_correct_rejections = df['TrialResult'].value_counts().get('Correct Rejection', 0)
    total_trials = len(df)
    total_users = df['UserId'].nunique()
    accuracy_rate = (num_hits + num_correct_rejections) / total_trials if total_trials > 0 else 0

    print(f"Total number of NaN RT values: {nan_RT_count}\n")
    print(f"Total number of trials: {total_trials}")
    print(f"Total number of unique users: {total_users}")
    print(f"Total number of Hits {num_hits}; Correct Rejections {num_correct_rejections}\nOverall accuracy rate {accuracy_rate}\n")

# Main preprocessing workflow
def preprocess_data(df):
    categorize_target_condition(df)        # Categorize target condition
    assign_trial_numbers(df)               # Assign trial numbers to each trial for each user
    categorize_trial_results(df)           # Categorize trials into hits, false alarms, etc.
    calculate_response_times(df)           # Calculate and clean response times
    log_response_times(df)                 # Apply logarithmic transformation to response times
    calculate_statistics(df)               # Calculate and print various statistics

# Load paths
data_path = os.getenv('DATA_PATH', './data')
output_path = os.getenv('OUTPUT_PATH', './output')

# Read
df = pd.read_csv(f'{data_path}/wColor_Honolulu_sandboxId_1-5.csv', low_memory=False)

# Preprocess
preprocess_data(df)

# Save
df.to_csv(f'{output_path}/df_HNL_1-5.csv', index=False)