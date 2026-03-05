import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

import joblib

df_orig = pd.read_csv('RawClinicalData.csv')
df_proc = pd.read_csv('RiskAugmentedData.csv')

if 'charges' not in df_proc.columns:
    df_proc['charges'] = df_orig['charges']

missing_mask = (df_proc['charges'].isnull()) | (df_proc['charges'] <= 0)
valid_mask = ~missing_mask

train_df = df_proc[valid_mask].copy()
predict_df = df_proc[missing_mask].copy()

formula = "charges ~ slos + log_risk_score + hazard_multiplier + num_co"

df_proc.rename(columns={'num.co': 'num_co'}, inplace=True)
train_df = df_proc[valid_mask].copy()
predict_df = df_proc[missing_mask].copy()

try:
    model = smf.glm(formula=formula, data=train_df,
                    family=sm.families.Gamma(link=sm.families.links.log())).fit()
    print(model.summary())

    # Predict
    predicted_charges = model.predict(predict_df)

    # Fill values
    df_proc.loc[missing_mask, 'charges'] = predicted_charges

    # Visual check
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df_proc.loc[valid_mask, 'charges'], label='Original Charges', fill=True)
    sns.kdeplot(predicted_charges, label='Imputed Charges', fill=True)
    plt.title('Distribution of Original vs Imputed Charges')
    plt.legend()
    plt.savefig('imputation_check.png')

    # Save the updated dataset
    df_proc.to_csv('survival_analysis_imputed.csv', index=False)
    print("Imputation successful. File saved.")

except Exception as e:
    print(f"GLM Fitting Failed: {e}")

joblib.dump(model, 'charge_glm_model.pkl')
print("Charge GLM Modeli (charge_glm_model.pkl) kaydedildi.")
