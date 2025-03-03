import sys
import statsmodels.formula.api as smf
from scipy.stats import chi2
import pickle
import numpy as np
import pandas as pd
import os

def calculate_bic(model, n):
    """Calculate BIC manually for a given fitted model."""
    llf = model.llf  # Log-likelihood
    k = len(model.params)  # Number of estimated parameters
    return -2 * llf + np.log(n) * k

def save_model_outputs(result, summary_path, model_path, results_path):
    """Save model summary, model, and results."""
    with open(summary_path, 'w') as f:
        f.write(result.summary().as_text())
    with open(model_path, 'wb') as f:
        pickle.dump(result.model, f)
    with open(results_path, 'wb') as f:
        pickle.dump(result, f)

def likelihood_ratio_test(full_model, reduced_model):
    """Calculate LRT and p-value between full and reduced models."""
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_difference = len(full_model.params) - len(reduced_model.params)
    p_value = chi2.sf(lr_stat, df=df_difference)
    return lr_stat, p_value

def compare_models(full_model, reduced_models, n):
    """Compare full model with a list of reduced models."""
    for name, reduced_model in reduced_models:
        try:
            lr_stat, p_value = likelihood_ratio_test(full_model, reduced_model)
            print(f"LRT for {name}: Stat={lr_stat}, p-value={p_value}")
            print(f"BIC for {name}: {calculate_bic(reduced_model, n)}")
        except Exception as e:
            print(f"Error comparing model '{name}': {str(e)}")

# Load data
data_path = os.getenv('DATA_PATH', './data')
output_path = os.getenv('OUTPUT_PATH', './output')
df_cleaned = pd.read_csv(f'{output_path}/df_HNL1_hits_final_cleaned_for_LME.csv', low_memory=False)

# Define the variables
categorical_ivs = ['TrialNumber','TrialsSinceLast_Illegal1Name_ByDay','TrialsSinceLast_target_present_ByDay','LegalItems','Illegal1Name']
continuous_ivs = ['avg_hit_RT','Cumulative_Illegal1Name_ByDay_Prob','Cumulative_target_present_ByDay_Prob']
grouping_var = 'UserId'
dv = 'RT'

# Order Illegal1Name based on Difficulty_Score
difficulty_order = df_cleaned.groupby("Illegal1Name")["Difficulty_Score"].first().sort_values().index.tolist()
df_cleaned["Illegal1Name"] = pd.Categorical(df_cleaned["Illegal1Name"], categories=difficulty_order, ordered=True)
print("Target (Illegal1Name) difficulty order:", difficulty_order)
# Define the reference level (the Illegal1Name with the lowest Difficulty_Score, i.e., PISTOL)
reference_level = difficulty_order[0]

# Redirect print output to a log file
log_path = f'{output_path}/omnibus_lme_model_analysis_log.txt'
with open(log_path, 'w') as log_file:
    sys.stdout = log_file

    try:
        # Full model
        full_formula = f'RT ~ C(TrialNumber)+C(TrialsSinceLast_Illegal1Name_ByDay)+C(TrialsSinceLast_target_present_ByDay)+C(LegalItems)+C(Illegal1Name, Treatment(reference="{reference_level}")) + avg_hit_RT+Cumulative_Illegal1Name_ByDay_Prob+Cumulative_target_present_ByDay_Prob + (1|UserId)'
        full_model = smf.mixedlm(full_formula, df_cleaned, groups=df_cleaned['UserId']).fit()

        # Save the full model output
        save_model_outputs(full_model, f'{output_path}/omnibus_full_model_summary.txt', f'{output_path}/omnibus_full_model.pkl', f'{output_path}/omnibus_full_model_results.pkl')

        # Save the trial-by-trial fitted values and residuals
        df_cleaned['Fitted_Values'] = full_model.fittedvalues
        df_cleaned['Residuals'] = full_model.resid
        df_cleaned.to_csv(f'{output_path}/df_HNL1_3factors_LME_fitted_values_residuals.csv', index=False)
        print("Saved fitted values and residuals to 'df_HNL1_3factors_LME_fitted_values_residuals.csv'.")

        # Reduced models
        reduced_formulas = {
            "Without Individual Differences": 'RT ~ C(TrialNumber) + C(TrialsSinceLast_Illegal1Name_ByDay) + C(TrialsSinceLast_target_present_ByDay) + C(LegalItems) + C(Illegal1Name) + Cumulative_Illegal1Name_ByDay_Prob + Cumulative_target_present_ByDay_Prob + (1|UserId)',
            "Without Trial History": 'RT ~ C(LegalItems) + C(Illegal1Name) + avg_hit_RT + (1|UserId)',
            "Without Recent Exposure Features": 'RT ~ C(TrialNumber)+C(LegalItems)+C(Illegal1Name) + avg_hit_RT+Cumulative_Illegal1Name_ByDay_Prob+Cumulative_target_present_ByDay_Prob + (1|UserId)',
            "Without Cumulative Exposure Features": 'RT ~ C(TrialNumber)+C(TrialsSinceLast_Illegal1Name_ByDay)+C(TrialsSinceLast_target_present_ByDay)+C(LegalItems)+C(Illegal1Name) + avg_hit_RT + (1|UserId)',
            "Without Stimulus Features": 'RT ~ C(TrialNumber) + C(TrialsSinceLast_Illegal1Name_ByDay) + C(TrialsSinceLast_target_present_ByDay) + avg_hit_RT + Cumulative_Illegal1Name_ByDay_Prob + Cumulative_target_present_ByDay_Prob + (1|UserId)'
        }

        # Number of observations
        n = len(df_cleaned)

        # Fit and save reduced models
        reduced_models = []
        for name, formula in reduced_formulas.items():
            try:
                reduced_model = smf.mixedlm(formula, df_cleaned, groups=df_cleaned['UserId']).fit()
                reduced_models.append((name, reduced_model))
            except Exception as e:
                print(f"Error fitting model '{name}': {str(e)}")

        # Compare models
        compare_models(full_model, reduced_models, n)

        # Var by Var comparison
        all_vars = ['C(TrialNumber)', 'C(TrialsSinceLast_Illegal1Name_ByDay)', 'C(TrialsSinceLast_target_present_ByDay)', 'C(LegalItems)', 'C(Illegal1Name)', 'avg_hit_RT', 'Cumulative_Illegal1Name_ByDay_Prob', 'Cumulative_target_present_ByDay_Prob']
        base_formula = 'RT ~ ' + ' + '.join(all_vars[:-1]) + ' + (1|UserId)'

        # For individual variables
        for var in all_vars:
            remaining_vars = [v for v in all_vars if v != var]
            reduced_formula = f"RT ~ {' + '.join(remaining_vars)} + (1|UserId)"
            try:
                reduced_model = smf.mixedlm(reduced_formula, df_cleaned, groups=df_cleaned['UserId']).fit()
                print(f"\nTesting without {var}")
                lr_stat, p_value = likelihood_ratio_test(full_model, reduced_model)
                print(f"LRT stat: {lr_stat}, p-value: {p_value}")
                print(f"BIC: {calculate_bic(reduced_model, n)}")
            except Exception as e:
                print(f"Error fitting reduced model without {var}: {str(e)}")

    except Exception as e:
        print(f"Critical error with full model setup: {str(e)}")

# Reset stdout to default
sys.stdout = sys.__stdout__

# Confirm where the log has been saved
print(f"Printed outputs saved to {log_path}")