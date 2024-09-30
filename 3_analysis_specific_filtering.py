import numpy as np
import pandas as pd
import logging
from datetime import datetime
import os

# Load paths
data_path = os.getenv('DATA_PATH', './data')
output_path = os.getenv('OUTPUT_PATH', './output')

# Setup logging
log_filename = f'filtering_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
logging.basicConfig(filename=log_filename,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_and_print(message):
    print(message)
    logging.info(message)

# Load initial data
df = pd.read_csv(f'{output_path}/df_HNL_1-5_recent_occurrence.csv', low_memory=False)
initial_trial_count = len(df)
initial_user_count = df['UserId'].nunique()
log_and_print(f"Initial data loaded: {initial_trial_count} trials from {initial_user_count} unique users.")

# Filtering Process

# 1. Filter for only days 1 and 2
df_day_filtered = df[df['Day'].isin([1, 2])].copy()
filtered_trial_count = len(df_day_filtered)
filtered_user_count = df_day_filtered['UserId'].nunique()
log_and_print(f"After filtering for Days 1 and 2: {filtered_trial_count} trials from {filtered_user_count} unique users.")

# 2. Filter based on allowed upgrades
allowed_bitmask = 8 | 16 | 2048

def is_allowed_upgrade(value):
    return (value & ~allowed_bitmask) == 0

df_day_filtered['AllowedUpgrade'] = df_day_filtered['ActiveUpgradesId'].apply(is_allowed_upgrade)
disallowed_user_ids = df_day_filtered[~df_day_filtered['AllowedUpgrade']]['UserId'].unique()
df_upgrade_filtered = df_day_filtered[~df_day_filtered['UserId'].isin(disallowed_user_ids)].copy()
removed_users_upgrade = len(disallowed_user_ids)
log_and_print(f"Removed {removed_users_upgrade} users due to disallowed upgrades.")

# Update for subsequent processing
df_filtered = df_upgrade_filtered.drop(columns=['AllowedUpgrade'])
current_trial_count = len(df_filtered)
current_user_count = df_filtered['UserId'].nunique()
log_and_print(f"After all initial filters: {current_trial_count} trials from {current_user_count} unique users.")

# 3. Subset day 1 and 2 for individual metrics calculations
df_day1 = df_filtered[df_filtered['Day'] == 1].copy()
df_day2 = df_filtered[df_filtered['Day'] == 2].copy()

# 4. Calculate performance metrics for Day 2 target present trials
df_day2_tp = df_day2[df_day2['IllegalItems'] > 0].copy()
metrics_overall = df_day2_tp.groupby('UserId').agg(
    avg_target_present_accuracy=('TrialResult', lambda x: (x == 'Hit').mean()),
    avg_hit_RT=('RT', 'mean'),
    avg_hit_log_RT=('log_RT', 'mean')
).reset_index()

# Check for users with NaN in avg_hit_RT
nan_rt_users = metrics_overall['UserId'][metrics_overall['avg_hit_RT'].isna()].tolist()
num_nan_rt_users = len(nan_rt_users)
log_and_print(f"Found {num_nan_rt_users} users with NaN avg_hit_RT.")

# 5. Calculate target-specific performance metrics
metrics_target = df_day2_tp.groupby(['UserId', 'Illegal1Name']).agg(
    target_id_accuracy=('TrialResult', lambda x: (x == 'Hit').mean()),
    target_id_hit_RT=('RT', 'mean'),
    target_id_hit_log_RT=('log_RT', 'mean')
).reset_index()

# Pivot target-specific metrics
metrics_target_pivot = metrics_target.pivot_table(
    index='UserId',
    columns='Illegal1Name',
    values=['target_id_accuracy', 'target_id_hit_RT', 'target_id_hit_log_RT']
)
metrics_target_pivot.columns = [f"{col[1]}-{col[0]}" for col in metrics_target_pivot.columns]
metrics_target_pivot.reset_index(inplace=True)

# 6. Combine overall and target-specific metrics
individual_metrics = metrics_overall.merge(metrics_target_pivot, on='UserId', how='left')
individual_metrics.to_csv('/lustre/CCAS/mitroffgrp/Audrey/output/individual_metrics.csv', index=False)
log_and_print("Saved individual metrics to 'individual_metrics.csv'.")

# 7. Merge individual metrics with Day 1 data
df_day1_metrics = df_day1.merge(individual_metrics, on='UserId', how='left')

# 8. Further Filters

# a. Remove multiple target trials
df_single_target = df_day1_metrics[df_day1_metrics['IllegalItems'] == 1].copy()
log_and_print(f"After removing multiple target trials: {len(df_single_target)} trials remain.")

# b. Filter for common bag types (TypeId 1-4)
df_common_bags = df_single_target[df_single_target['TypeId'].isin([1, 2, 3, 4])].copy()
log_and_print(f"After filtering for common bag types: {len(df_common_bags)} trials remain.")

# c. Identify top 10 most frequent targets in common bag types
top_targets = df_common_bags['Illegal1Name'].value_counts().head(10).index.tolist()

# d. Find targets common across all bag types
targets_per_bag = {
    bag_type: set(df_common_bags[df_common_bags['TypeId'] == bag_type]['Illegal1Name'].unique())
    for bag_type in [1, 2, 3, 4]
}
common_targets = set.intersection(*targets_per_bag.values())
final_targets = list(set(top_targets).intersection(common_targets))
log_and_print(f"Identified {len(final_targets)} common targets across all bag types.")

# e. Filter DataFrame to include only these common targets
df_final_targets = df_common_bags[df_common_bags['Illegal1Name'].isin(final_targets)].copy()
log_and_print(f"After filtering for common targets: {len(df_final_targets)} trials remain.")

# f. Filter out small set sizes (LegalItems <=4)
df_large_sets = df_final_targets[df_final_targets['LegalItems'] > 4].copy()
log_and_print(f"After filtering out small set sizes: {len(df_large_sets)} trials remain.")

# 9. Feature Engineering

# a. Calculate cumulative target exposure probability
df_large_sets['Cumulative_Illegal1Name_ByDay_Prob'] = df_large_sets['Cumulative_Illegal1Name_ByDay'] / df_large_sets['TrialNumber']
df_large_sets['Cumulative_target_present_ByDay_Prob'] = df_large_sets['Cumulative_target_present_ByDay'] / df_large_sets['TrialNumber']

# b. Create binary split variables
median_avg_hit_RT = df_large_sets['avg_hit_RT'].median()
df_large_sets['avg_hit_RT_Category'] = np.where(df_large_sets['avg_hit_RT'] > median_avg_hit_RT, 'high', 'low')

df_large_sets['PreviousTargetIdMatch'] = np.where(df_large_sets['TrialsSinceLast_Illegal1Name_ByDay'] == 1, 1, 0)
df_large_sets['PreviousTargetCondMatch'] = np.where(df_large_sets['TrialsSinceLast_target_present_ByDay'] == 1, 1, 0)
df_large_sets['SetSize_Category'] = np.where(df_large_sets['LegalItems'] > 7, 'high', 'low')
df_large_sets['Plane'] = np.where(df_large_sets['TrialNumber'] > 12, 2, 1)

# c. Import target difficulty scores from Day 2; note keep using this (scored from sandbox) in real analysis
difficulty_scores = pd.read_csv(f'{output_path}/target_difficulty_omnibus_lme.csv')
df_feature_engineered = df_large_sets.merge(
    difficulty_scores[['Illegal1Name', 'Difficulty_Score', 'Difficulty_Category']],
    on='Illegal1Name',
    how='left'
)

# 10. Filter out any trials that are not "Hit"
df_hits_only = df_feature_engineered[df_feature_engineered['TrialResult'] == 'Hit'].copy()
removed_non_hit_trials = len(df_feature_engineered) - len(df_hits_only)
remaining_users_after_hit_filter = df_hits_only['UserId'].nunique()

log_and_print(f"Removed {removed_non_hit_trials} non-hit trials. Remaining dataset contains {len(df_hits_only)} hit trials from {remaining_users_after_hit_filter} unique subjects.")

# 11. Final Clean-Up: Remove rows with missing values in key columns
columns_to_check = [
    'avg_hit_RT_Category', 'PreviousTargetIdMatch', 'PreviousTargetCondMatch', 
    'SetSize_Category', 'Difficulty_Category', 'Plane', 'UserId', 'RT',
    'TrialNumber', 'TrialsSinceLast_Illegal1Name_ByDay', 'TrialsSinceLast_target_present_ByDay', 
    'LegalItems', 'Illegal1Name', 'avg_hit_RT', 'Cumulative_Illegal1Name_ByDay_Prob', 
    'Cumulative_target_present_ByDay_Prob'
]
missing_values = df_hits_only[columns_to_check].isnull().sum()
log_and_print("Missing values before final clean-up:")
log_and_print(missing_values.to_string())

df_final_cleaned = df_hits_only.dropna(subset=columns_to_check).copy()
removed_trials = len(df_hits_only) - len(df_final_cleaned)
final_user_count = df_final_cleaned['UserId'].nunique()

log_and_print(f"Removed {removed_trials} trials due to missing values.")
log_and_print(f"Final dataset contains {len(df_final_cleaned)} trials from {final_user_count} unique subjects.")

# Save the final cleaned DataFrame
df_feature_engineered.to_csv(f'{output_path}/df_HNL1_all_final.csv', index=False)
log_and_print("Saved intermediate DataFrame, before hit-filtering and NA removal, to 'df_HNL1_all_final.csv'.")
df_final_cleaned.to_csv(f'{output_path}/df_HNL1_hits_final_cleaned_for_LME.csv', index=False)
log_and_print("Saved final cleaned DataFrame to 'df_HNL1_hits_final_cleaned_for_LME.csv'.")

# End of Script
