#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from model_v5_calibrated import CalibratedModel, FORBIDDEN_COLUMNS
import warnings
warnings.filterwarnings('ignore')

def evaluate_model_cv(train_df, n_splits=5):
    """Evaluate model using stratified K-fold cross-validation"""
    
    # Remove forbidden columns except Position
    forbidden_to_remove = [col for col in FORBIDDEN_COLUMNS if col != 'Position']
    for col in forbidden_to_remove:
        if col in train_df.columns:
            train_df = train_df.drop(col, axis=1)
    
    # Group by race
    race_ids = train_df['Race_ID'].unique()
    race_labels = train_df.groupby('Race_ID')['Position'].apply(lambda x: (x == 1).any()).astype(int)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_log_losses = []
    all_brier_scores = []
    
    print(f"Running {n_splits}-fold cross-validation...")
    print("="*60)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(race_ids, race_labels)):
        train_races = race_ids[train_idx]
        val_races = race_ids[val_idx]
        
        fold_train = train_df[train_df['Race_ID'].isin(train_races)].copy()
        fold_val = train_df[train_df['Race_ID'].isin(val_races)].copy()
        
        # Train model with calibration
        model = CalibratedModel()
        model.train(fold_train, calibrate=True)
        
        # Predict on validation set
        val_predictions = model.predict(fold_val)
        
        # Merge predictions with actual outcomes
        val_with_preds = fold_val.merge(
            val_predictions, 
            on=['Race_ID', 'Horse'], 
            how='left'
        )
        
        # Calculate metrics
        y_true = (val_with_preds['Position'] == 1).astype(int)
        y_pred = val_with_preds['Predicted_Probability']
        
        fold_log_loss = log_loss(y_true, y_pred)
        fold_brier = brier_score_loss(y_true, y_pred)
        
        all_log_losses.append(fold_log_loss)
        all_brier_scores.append(fold_brier)
        
        print(f"Fold {fold+1}: Log-Loss = {fold_log_loss:.4f}, Brier Score = {fold_brier:.4f}")
    
    print("="*60)
    print(f"\nMean Log-Loss: {np.mean(all_log_losses):.4f} (±{np.std(all_log_losses):.4f})")
    print(f"Mean Brier Score: {np.mean(all_brier_scores):.4f} (±{np.std(all_brier_scores):.4f})")
    
    return {
        'mean_log_loss': np.mean(all_log_losses),
        'std_log_loss': np.std(all_log_losses),
        'mean_brier': np.mean(all_brier_scores),
        'std_brier': np.std(all_brier_scores)
    }

if __name__ == "__main__":
    print("Evaluating V5 Calibrated Model Performance")
    print("="*60)
    
    # Load training data
    train_df = pd.read_csv('data/trainData.csv')
    print(f"Training samples: {len(train_df)}")
    print()
    
    # Evaluate
    results = evaluate_model_cv(train_df, n_splits=5)
    
    print("\n" + "="*60)
    print("V5 MODEL PERFORMANCE")
    print("="*60)
    print(f"Log-Loss: {results['mean_log_loss']:.4f} ± {results['std_log_loss']:.4f}")
    print(f"Brier Score: {results['mean_brier']:.4f} ± {results['std_brier']:.4f}")
    print()
    print("Baseline (V4) was:")
    print("Log-Loss: 0.3351 ± 0.0052")
    print("Brier Score: 0.0908 ± 0.0006")
