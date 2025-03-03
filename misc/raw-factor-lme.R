# Load necessary libraries
library(Matrix)
library(lme4)
library(car) # For Anova with Type III tests

# Read the data
df_cleaned <- read.csv("/Users/g39836381/Library/CloudStorage/Box-Box/HPC/three_factors_final_prereg_output_03-03-25/df_HNL1_hits_final_cleaned_for_LME.csv")

options(contrasts = c("contr.sum", "contr.poly"))

# Fit the mixed-effects model
model <- lmer(
  RT ~ C(TrialNumber) + C(TrialsSinceLast_Illegal1Name_ByDay) + 
    C(TrialsSinceLast_target_present_ByDay) + C(LegalItems) + C(Illegal1Name) +
    avg_hit_RT + Cumulative_Illegal1Name_ByDay_Prob + Cumulative_target_present_ByDay_Prob + 
    (1 | UserId),
  data = df_cleaned
)

# Run Type III ANOVA to get p-values for each categorical variable
anova_results <- Anova(model, type = "III", test = "F")
print(anova_results)