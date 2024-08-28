import sys
import statsmodels.formula.api as smf
from scipy.stats import chi2
import pandas as pd
import pickle
import os

def calculate_bic(model):
    """Return the BIC for the model."""
    return model.bic

def save_model_outputs(result, summary_path, model_path, results_path):
    """Save model summary, model object, and fitted results."""
    with open(summary_path, 'w') as f:
        f.write(result.summary().as_text())
    with open(model_path, 'wb') as f:
        pickle.dump(result.model, f)
    with open(results_path, 'wb') as f:
        pickle.dump(result, f)

def likelihood_ratio_test(full_model, reduced_model):
    """Calculate LRT statistic and p-value."""
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_difference = len(full_model.params) - len(reduced_model.params)
    p_value = chi2.sf(lr_stat, df=df_difference)
    return lr_stat, p_value

# Load data and paths
data_path = os.getenv('DATA_PATH', './data')
output_path = os.getenv('OUTPUT_PATH', './output')

df_cleaned_simple = pd.read_csv(f'{output_path}/df_HNL1_hits_final_cleaned_for_LME.csv', low_memory=False)

# Define the log file path
log_path = f'{output_path}/omnibus_lme_median_split_lrt_log.txt'

# Redirect stdout to the log file
original_stdout = sys.stdout  # Save a reference to the original standard output
with open(log_path, 'w') as log_file:
    sys.stdout = log_file  # Change the standard output to the log file

    # Full model
    full_formula = 'RT ~ avg_hit_RT_Category * PreviousTargetCondMatch * Difficulty_Category * C(Plane) + (1|UserId)'
    full_model = smf.mixedlm(full_formula, df_cleaned_simple, groups=df_cleaned_simple['UserId']).fit()

    # Print the full model summary
    print(full_model.summary())

    # Define the reduced formulas and fit models
    reduced_formulas = {
        "Without Interactions": 'RT ~ avg_hit_RT_Category + PreviousTargetCondMatch + Difficulty_Category + C(Plane) + (1|UserId)',
        "Without avg_hit_RT_Category": 'RT ~ PreviousTargetCondMatch*Difficulty_Category*C(Plane) + (1|UserId)',
        "Without PreviousTargetCondMatch": 'RT ~ avg_hit_RT_Category*Difficulty_Category*C(Plane) + (1|UserId)',
        "Without Difficulty_Category": 'RT ~ avg_hit_RT_Category*PreviousTargetCondMatch*C(Plane) + (1|UserId)',
        "Without C(Plane)": 'RT ~ avg_hit_RT_Category*PreviousTargetCondMatch*Difficulty_Category + (1|UserId)',
    }

    # Number of observations
    n = len(df_cleaned_simple)

    # Fit and save reduced models, compare with full model
    for name, formula in reduced_formulas.items():
        reduced_model = smf.mixedlm(formula, df_cleaned_simple, groups=df_cleaned_simple['UserId']).fit()
        lr_stat, p_value = likelihood_ratio_test(full_model, reduced_model)
        bic = calculate_bic(reduced_model)
        print(f"\n{name}:")
        print("Likelihood Ratio Statistic:", lr_stat)
        print("P-Value:", p_value)
        print("BIC:", bic)

    # Save outputs
    save_model_outputs(full_model, f'{output_path}/omnibus_binary_model_summary.txt', f'{output_path}/mitroffgrp/Audrey/output/omnibus_binary_model.pkl', f'{output_path}/omnibus_binary_model_results.pkl')

# Reset stdout to its original setting
sys.stdout = original_stdout

# Notify completion
print(f"Analysis complete. Results and outputs have been saved to {log_path}")