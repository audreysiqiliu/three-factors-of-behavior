# Three Factors of Behavior: Pre-registered Analysis Pipeline

## Overview
Comparing the relative impact of individual differences, prior experience, and stimulus properties on visual search behavior in an airport security screening game. Run scripts sequentially.

## Directory Structure

```plaintext
final_prereg_script/
├── misc/                             
│   ├── Combined_IllegalId_Name_Color.csv # Illegal Item info used by add_color.py
│   ├── Combined_LegalId_Name_Color.csv   # Legal Item info
│   ├── ASDB_data_fix.py                  # Ensures all rows in data have the same number of columns
│   ├── compile_data                      # Use if concatenating multiple raw data files
│   └── add_color.py                      # Script to add item names and color to main dataframe
├── 1_general_data_prep.py            # Initial data preprocessing
├── 2_add_recent_occurrence_vars.py   # Add recent occurrence variables
├── 3_analysis_specific_filtering.py  # Apply analysis-specific filtering
├── 4a_raw-factor_models.py           # Fit and analyze raw factor models
├── 4b_binary-factor_models.py        # Fit and analyze binary factor models
└── run_all_scripts.sh                # (not included) SLURM job script to run all scripts sequentially

```

## Script Descriptions

### 1. `1_general_data_prep.py`

**Purpose**:
- Performs initial preprocessing of the dataset.

**Input**:
- `file/path/in/run_all_scripts/wColor_Honolulu_sandboxId_1-2.csv`: Raw 5% Sandbox data airport scanner honolulu (HNL) Days 1-5

**Cleaning Steps**:
- **Filter out Replays**: flag for trials played more than once
- **Filter users by trial counts**: Filter out users without exactly 24 trials in Day 1 and 36 trials in Day2
- **Categorize Target Condition**: Categorizes trials as `target_present` or `target_absent`.
- **Assign Trial Numbers**: Assigns a sequential trial number to each user’s trials.
- **Categorize Trial Results**: Categorizes trials as `Hit`, `False Alarm`, `Correct Rejection`, or `Miss`.
- **Calculate Response Times**: Calculates and cleans response times (`RT`) and performs a log transformation.
- **Filter Invalid RTs**: Filters out invalid/nonexistent RT values and removes users with invalid RTs.

**Output**:
- `df_HNL_1-2.csv`: The cleaned and preprocessed dataset.

---
### 2. `2_add_recent_occurrence_vars.py`

**Purpose**:
- This script adds variables related to recent occurrences of specific events, such as the number of trials since certain items or colors were last seen. It also tracks cumulative occurrences and flags for color matches within trials.

**Input**:
- **`df_HNL_1-2.csv`**: The cleaned and preprocessed dataset from the first script, which contains categorized trials and response times.

**Steps**:
1. **Calculate Trials Since Last Occurrence**:
   - Adds columns that track the number of trials since the last occurrence of specific variables (`Illegal1Name`, `target_present`, `TypeId`), resetting at the start of each Day (i.e., airport scanner level).
   - Also calculates cumulative occurrences of these variables within the day.

2. **Calculate Color Match Details**:
   - Adds columns to track the number of trials since the last color match between `Illegal1Color` and any `LegalColor`.
   - Flags trials where a color match occurred.
   - Tracks cumulative occurrences of `Illegal1Color` as both legal and illegal colors.

3. **Copy Last Trial Results**:
   - For each occurrence of `Illegal1Name`, `target_present`, `TypeId`, and color match, this script copies the trial result of the last occurrence.

**Output**:
- **`df_HNL_1-2_recent_occurrence.csv`**: The dataset with additional columns related to recent occurrences, cumulative counts, and color match flags, ready for further analysis.

---

### 3. `3_analysis_specific_filtering.py`

**Purpose**:
- This script performs filtering and feature engineering specific to the IVs of interest for the LME. Dataset is reduced from Days 1-5 to Days 1-2, performance metrics are taken from Day 2, merged with Day 1. The final dataset contains Day 1 hit trials only.

**Input**:
- **`df_HNL_1-2_recent_occurrence.csv`**: The dataset containing recent occurrence variables, produced by the previous script.

**Steps**:

1. **Load Initial Data**:
   - Load dataset and log the initial number of trials and users.

2. **Filter for Days 1 and 2**:
   - The dataset is filtered to include only trials from Days 1 and 2.

3. **Filter Based on Allowed Upgrades**:
   - Users with disallowed upgrades, identified using a bitmask, are filtered out.

4. **Subset Data for Day 1 and Day 2**:
   - The dataset is split into Day 1 and Day 2 subsets for individual metrics calculations.

5. **Calculate Individual Performance Metrics using Day 2 Target-Present Trials**:
   - Metrics such as average target-present accuracy and hit RT.

6. **Calculate Target-Specific Day 2 Performance Metrics**:
   - Target-specific metrics (e.g., hit rate, average RT) are calculated and pivoted for each target (Illegal1Name).

7. **Merge Metrics with Day 1 Data**:
   - The calculated metrics are merged back with the Day 1 data to produce a comprehensive dataset. `avg hit RT` is used as the primary individual difference measure.

8. **Further Filters**:
   - **Remove Multiple Target Trials**: Trials with multiple targets are removed.
   - **Filter for Common Bag Types**: The dataset is filtered to include only trials with common bag types (`TypeId` 1-4).
   - **Identify and Retain Common Targets**: The script identifies and retains the top 8 most frequent targets common across all bag types.
   - **Filter Out Small Set Sizes**: Trials with small set sizes (`LegalItems` <= 4) are removed.

9. **Feature Engineering**:
    - **Cumulative Target Exposure Probability**: Calculates the cumulative target exposure as a proportion of current trial number.
    - **Create Binary Split Variables**: Binary variables are created based on median `avg hit RT`, previous target ID match, previous target condition match, set size category, and trial number.

10. **Filter for Hit Trials**:
    - The dataset is filtered to include only hit trials (`TrialResult == 'Hit'`).

11. **Final Clean-Up**:
    - Rows with missing values in key columns (independent variables for subsequent LMEs) are removed, and the final cleaned dataset is prepared for model analysis.

**Output**:
- **`df_HNL1_all_final.csv`**: An intermediate dataset saved before filtering for hit trials and NA removal.
- **`df_HNL1_hits_final_cleaned_for_LME.csv`**: The final cleaned dataset, filtered for hit trials, ready for LME (Linear Mixed Effects) modeling.
- **`filtering_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt`**: log of # trials and users removed through filtering.

### 4a. `4a_raw-factor_models.py`

**Purpose**:
- This script fits the raw-factors linear mixed-effects models (LME). Full and reduced models are compared using likelihood ratio tests (LRT) to quantify the importance of the variables omitted in the reduced models.

**Input**:
- **`df_HNL1_hits_final_cleaned_for_LME.csv`**: The cleaned and filtered dataset produced in the previous steps.

**Steps**:

1. **Load Data**:
   - The script loads the cleaned dataset, which includes the variables to be used in the model.

2. **Define Variables**:
   - Categorical Independent Variables (`categorical_ivs`): `TrialNumber`, `TrialsSinceLast_Illegal1Name_ByDay`, `TrialsSinceLast_target_present_ByDay`, `LegalItems`, `Illegal1Name`.
   - Continuous Independent Variables (`continuous_ivs`): `avg_hit_RT`, `Cumulative_Illegal1Name_ByDay_Prob`, `Cumulative_target_present_ByDay_Prob`.
   - Dependent Variable (`dv`): `RT` (response time).
   - Grouping Variable (`grouping_var`): `UserId`.

3. **Fit the Full Model**:
   - A full mixed-effects model is fitted with all the specified variables. The fitted model is saved to a pickle file, and the summary is written to a text file.
   - Trial-by-trial fitted values and residuals are also saved to a CSV file.

4. **Fit and Compare Reduced Models**:
   - Several reduced models are fitted, each omitting different sets of variables (e.g., without individual differences, without trial history, etc.).
   - The models are compared using likelihood ratio tests (LRT) and Bayesian Information Criterion (BIC).

5. **Var-by-Var Model Comparisons**:
   - For each variable in the full model, a reduced model is fitted excluding that variable to test its contribution to the model fit.

6. **Log Output**:
   - All print statements (including model comparison statistics) are logged to a specified log file.

**Output**:
- **`omnibus_full_model_summary.txt`**: Text file containing the summary of the full LME model.
- **`omnibus_full_model.pkl`**: Pickle file storing the full LME model.
- **`omnibus_full_model_results.pkl`**: Pickle file storing the results of the full LME model.
- **`df_HNL1_3factors_LME_fitted_values_residuals.csv`**: CSV file with trial-by-trial fitted values and residuals.
- **`omnibus_lme_model_analysis_log.txt`**: Log file with detailed output from the model fitting and comparisons.

---

### 4b. `4b_binary-factor_models.py`

**Purpose**:
- This script fits the binary-factor LME to assess the main and interaction effects of binary-split variables on response times (`RT`). Also performs LRTs to quantify variable importance.

**Input**:
- **`df_HNL_1_ByDay_omnibus_lme_cleaned.csv`**: The cleaned and filtered dataset produced in the previous steps.

**Steps**:

1. **Load Data**

2. **Fit the Full Binary Factor Model**:
   - A full mixed-effects model is fitted with interaction terms between `avg_hit_RT_Category`, `PreviousTargetCondMatch`, `Difficulty_Category`, and `Plane`.
   - The summary of the full model is printed and logged.

3. **Fit and Compare Reduced Models**:
   - Several reduced models are fitted, each omitting different sets of variables or interactions.
   - The models are compared using likelihood ratio tests (LRT) and Bayesian Information Criterion (BIC).

4. **Log Output**:
   - All print statements (including model comparison statistics) are logged to a specified log file.

5. **Save Outputs**:
   - The script saves the full model, its summary, and results in specified output files.

**Output**:
- **`omnibus_binary_model_summary.txt`**: Text file containing the summary of the full binary factor model.
- **`omnibus_binary_model.pkl`**: Pickle file storing the full binary factor model.
- **`omnibus_binary_model_results.pkl`**: Pickle file storing the results of the full binary factor model.
- **`omnibus_lme_median_split_lrt_log.txt`**: Log file with detailed output from the model fitting and comparisons.

---

### Dependencies

- **Python 3** with the following libraries:
  - `numpy`
  - `pandas`
  - `logging`
  - `datetime`
  - `statsmodels`
  - `scipy`
  - `pickle`

### Data Inputs and Outputs

- **Input Files**:
  - `wColor_Honolulu_sandboxId_1-5.csv`: Raw dataset with legal/illegal item names and colors.
  - `target_difficulty_omnibus_lme.csv`: Additional dataset for importing target difficulty scores calculated from 5% sandbox.

- **Output Files**:
  - `df_HNL_1-2.csv`: Cleaned and preprocessed dataset from `1_general_data_prep.py`.
  - `df_HNL_1-2_recent_occurrence.csv`: Dataset with added recent occurrence variables from `2_add_recent_occurrence_vars.py`.
  - `df_HNL1_all_final.csv`: Intermediate dataset saved before filtering for hit trials and removing NAs, from `3_analysis_specific_filtering.py`.
  - `df_HNL1_hits_final_cleaned_for_LME.csv`: Final cleaned dataset ready for LME modeling from `3_analysis_specific_filtering.py`.
  - `omnibus_full_model_summary.txt`: Model summaries from raw factor models (`4a_raw-factor_models.py`).
  - `omnibus_full_model.pkl`: Pickle file storing the full LME model from `4a_raw-factor_models.py`.
  - `omnibus_full_model_results.pkl`: Pickle file storing the results of the full LME model from `4a_raw-factor_models.py`.
  - `df_HNL1_3factors_LME_fitted_values_residuals.csv`: Residuals and fitted values from raw factor models (`4a_raw-factor_models.py`).
  - `omnibus_lme_model_analysis_log.txt`: Log file with detailed output from the model fitting and comparisons in `4a_raw-factor_models.py`.
  - `omnibus_binary_model_summary.txt`: Model summaries from binary factor models (`4b_binary-factor_models.py`).
  - `omnibus_binary_model.pkl`: Pickle file storing the full binary factor model from `4b_binary-factor_models.py`.
  - `omnibus_binary_model_results.pkl`: Pickle file storing the results of the full binary factor model from `4b_binary-factor_models.py`.
  - `omnibus_lme_median_split_lrt_log.txt`: Log file with detailed output from the model fitting and comparisons in `4b_binary-factor_models.py`.

### Contact Information

For any questions regarding this project, please contact:

- **Name:** Audrey Liu
- **Email:** audrey.liu@gwu.edu

### License

The MIT License (MIT)
Copyright (c) 2024 Audrey Siqi-Liu
