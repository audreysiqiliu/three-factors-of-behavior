import numpy as np
import pandas as pd
import os

def categorize_target_condition(df):
    """ Categorizes trials based on whether a target is present or absent. """
    df['target_present'] = (df['IllegalItems'] > 0).astype(int)
    df['target_absent'] = (df['IllegalItems'] == 0).astype(int)
    return df

def assign_trial_numbers(df):
    """ Adds a trial number for each user, which increments per trial. """
    df['TrialNumber'] = df.groupby('UserId').cumcount() + 1
    return df

def categorize_trial_results(df):
    """ Categorizes each trial based on performance into hits, false alarms, correct rejections, and misses. """
    condition_hit = (df['IllegalItems'] > 0) & (df['IllegalItemsMarked'] == df['IllegalItems'])
    condition_false_alarm = (df['UniqueTaps'] > df['IllegalItems'])
    condition_correct_rejection = ((df['IllegalItems'] == 0) | df['IllegalItems'].isna()) & ((df['LegalItemsMarked'] == 0) | df['LegalItemsMarked'].isna())
    condition_miss = (df['IllegalItems'] > 0) & (df['IllegalItemsMarked'] < df['IllegalItems'])

    df['TrialResult'] = 'Incorrect'
    df.loc[condition_hit, 'TrialResult'] = 'Hit'
    df.loc[condition_correct_rejection, 'TrialResult'] = 'Correct Rejection'
    df.loc[condition_miss, 'TrialResult'] = 'Miss'
    df.loc[condition_false_alarm, 'TrialResult'] = 'False Alarm'
    return df

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
    df = filter_out_invalid_rt(df)
    return df

def filter_out_invalid_rt(df):
    """ Filters out invalid response times by row and User IDs associated with RT values that indicate faulty RT recording. """
    user_ids_with_invalid_rt = df[df['RT'].isin([0, 1])]['UserId'].unique()
    df.drop(df[df['UserId'].isin(user_ids_with_invalid_rt)].index, inplace=True)
    df.loc[(df['RT'] <= 250) | (df['RT'] > 10000), 'RT'] = np.nan
    print(f"Number of users with invalid RTs: {len(user_ids_with_invalid_rt)}")
    return df

def log_response_times(df):
    """ Calculates the logarithm of the response times. """
    df['log_RT'] = np.log(df['RT'])
    return df

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

def filter_users_by_trial_counts(df):
    """ Filters out users without exactly 24 trials on Day 1 and 36 trials on Day 2. """
    user_day1_trial_counts = df[df['Day'] == 1].groupby('UserId').size()
    user_day2_trial_counts = df[df['Day'] == 2].groupby('UserId').size()

    users_to_keep = user_day1_trial_counts[user_day1_trial_counts == 24].index.intersection(
                    user_day2_trial_counts[user_day2_trial_counts == 36].index)

    df_filtered = df[df['UserId'].isin(users_to_keep)].copy()
    removed_users = df['UserId'].nunique() - df_filtered['UserId'].nunique()
    print(f"Removed {removed_users} users without exactly 24 trials on Day 1 or 36 trials on Day 2.")
    return df_filtered

# Main preprocessing workflow
def preprocess_data(df):
    print(f"Initial number of trials: {len(df)}")
    print(f"Initial number of unique users: {df['UserId'].nunique()}")

    print("Running filter_users_by_trial_counts...")
    df = filter_users_by_trial_counts(df)
    print(f"After filter_users_by_trial_counts: {len(df)} trials, {df['UserId'].nunique()} users")

    print("Running categorize_target_condition...")
    df = categorize_target_condition(df)
    print(f"After categorize_target_condition: {len(df)} trials")

    print("Running assign_trial_numbers...")
    df = assign_trial_numbers(df)
    print(f"After assign_trial_numbers: {len(df)} trials")

    print("Running categorize_trial_results...")
    df = categorize_trial_results(df)
    print(f"After categorize_trial_results: {len(df)} trials")

    print("Running calculate_response_times...")
    df = calculate_response_times(df)
    print(f"After calculate_response_times: {len(df)} trials")

    print("Running log_response_times...")
    df = log_response_times(df)
    print(f"After log_response_times: {len(df)} trials")

    print("Calculating statistics...")
    calculate_statistics(df)  # This function just prints, no need to return

    return df

# Load paths
data_path = os.getenv('DATA_PATH', './data')
output_path = os.getenv('OUTPUT_PATH', './output')
data_file = os.getenv('DATA_FILE')

# Read
file_path = f"{output_path}/wColor_{data_file}"
df = pd.read_csv(file_path, low_memory=False, on_bad_lines='warn')
print(f"In step 1, reading file: {file_path}")

# Preprocess
df = preprocess_data(df)

# Save
output_file = f'{output_path}/df_HNL_1-2.csv'
df.to_csv(output_file, index=False)

# Verify saving
if os.path.exists(output_file):
    print(f"File saved successfully at: {output_file}")
else:
    print(f"File not found. Saving may have failed: {output_file}")
