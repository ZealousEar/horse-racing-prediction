import pandas as pd
import numpy as np
import os

def check_predictions_file():
    #Constructing path to CSV file relative to script's location
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, "predictions_v4.csv")

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    #Checking for required columns
    required_columns = ["Race_ID", "Horse", "Predicted_Probability"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {', '.join(missing_cols)}")
        return

    #Checking for missing values
    if df[required_columns].isnull().values.any():
        print("Error: Missing values found in required columns.")
        print(df[df[required_columns].isnull().any(axis=1)])

    #Checking probability sum per race
    #Converting to numeric, coercing errors
    df['Predicted_Probability'] = pd.to_numeric(df['Predicted_Probability'], errors='coerce')
    
    if df['Predicted_Probability'].isnull().any():
        print("Error: Non-numeric values found in Predicted_Probability after coercion.")
        print(df[df['Predicted_Probability'].isnull()])
    
    #Filtering out rows where Predicted_Probability became NaN before groupby sum
    valid_probs_df = df.dropna(subset=['Predicted_Probability'])
    
    race_probabilities_sum = valid_probs_df.groupby("Race_ID")["Predicted_Probability"].sum()
    
    #Checking if all sums are close to 1
    if not np.allclose(race_probabilities_sum, 1.0, atol=1e-5): 
        print("Error: Probabilities do not sum to 1.0 (with tolerance 1e-5) for following Race_IDs:")
        print(race_probabilities_sum[~np.allclose(race_probabilities_sum, 1.0, atol=1e-5)])
    else:
        print("Success: Probabilities sum to 1.0 for all races (within tolerance 1e-5).")

    #Checking if probabilities are between 0 and 1
    if 'Predicted_Probability' in df.columns:
        if df['Predicted_Probability'].isnull().values.any():
            print("Error: Predicted_Probability contains non-numeric or missing values.")
        elif not ((df["Predicted_Probability"] >= 0) & (df["Predicted_Probability"] <= 1)).all():
            print("Error: Probabilities are not all between 0 and 1.")
            print("Problematic rows (probabilities < 0 or > 1):")
            print(df[~((df["Predicted_Probability"] >= 0) & (df["Predicted_Probability"] <= 1))])
        else:
            print("Success: All probabilities are between 0 and 1.")
    else:
        print("Error: Predicted_Probability column is missing, cannot check range.")

    #Checking for duplicate predictions (same Race_ID and Horse)
    duplicates = df[df.duplicated(subset=['Race_ID', 'Horse'], keep=False)]
    if not duplicates.empty:
        print("Error: Duplicate predictions found for following Race_ID and Horse combinations:")
        print(duplicates)
    else:
        print("Success: No duplicate predictions found.")
        
    print("\nValidation script finished.")

if __name__ == "__main__":
    check_predictions_file() 