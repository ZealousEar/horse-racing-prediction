#!/usr/bin/env python3
"""
Properly Calibrated Model with Hold-out Calibration Set
Based on: "On Calibration of Modern Neural Networks" (Guo et al., 2017)

Key: Use separate calibration set, NOT cross-validation
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

FORBIDDEN_COLUMNS = ['betfairSP', 'Position', 'timeSecs', 'pdsBeaten', 'NMFP', 'NMFPLTO']

class CalibratedFinalModel:
    def __init__(self):
        self.model = None
        self.temperature = 1.0
        self.feature_cols = None
        self.specialization_stats = {}
        
    def validate_no_leakage(self, df, stage=""):
        violations = [col for col in df.columns if col in FORBIDDEN_COLUMNS and col != 'Position']
        if violations:
            raise ValueError(f"DATA LEAKAGE at {stage}: Found forbidden columns {violations}")
    
    def calculate_stats(self, train_df):
        """Bayesian smoothed specialization statistics"""
        train_df = train_df.sort_values(['Race_ID', 'Horse'])
        
        # Trainer-Course
        tc = train_df.groupby(['Trainer', 'Course']).agg({
            'Position': ['count', lambda x: (x == 1).sum()]
        }).reset_index()
        tc.columns = ['Trainer', 'Course', 'runs', 'wins']
        
        trainer_overall = train_df.groupby('Trainer').agg({
            'Position': [lambda x: (x == 1).sum(), 'count']
        }).reset_index()
        trainer_overall.columns = ['Trainer', 'wins_overall', 'runs_overall']
        trainer_overall['win_rate_overall'] = trainer_overall['wins_overall'] / trainer_overall['runs_overall']
        
        tc = tc.merge(trainer_overall, on='Trainer', how='left')
        tc['win_rate'] = (tc['wins'] + 10 * tc['win_rate_overall']) / (tc['runs'] + 10)
        
        self.specialization_stats['trainer_course'] = tc
        
        # Jockey-Course
        jc = train_df.groupby(['Jockey', 'Course']).agg({
            'Position': ['count', lambda x: (x == 1).sum()]
        }).reset_index()
        jc.columns = ['Jockey', 'Course', 'runs', 'wins']
        
        jockey_overall = train_df.groupby('Jockey').agg({
            'Position': [lambda x: (x == 1).sum(), 'count']
        }).reset_index()
        jockey_overall.columns = ['Jockey', 'wins_overall', 'runs_overall']
        jockey_overall['win_rate_overall'] = jockey_overall['wins_overall'] / jockey_overall['runs_overall']
        
        jc = jc.merge(jockey_overall, on='Jockey', how='left')
        jc['win_rate'] = (jc['wins'] + 10 * jc['win_rate_overall']) / (jc['runs'] + 10)
        
        self.specialization_stats['jockey_course'] = jc
        
        # Trainer overall
        trainer_stats = train_df.groupby('Trainer').agg({
            'Position': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
        }).reset_index()
        trainer_stats.columns = ['Trainer', 'total_runs', 'total_wins', 'total_places']
        trainer_stats['win_rate'] = trainer_stats['total_wins'] / trainer_stats['total_runs']
        trainer_stats['place_rate'] = trainer_stats['total_places'] / trainer_stats['total_runs']
        
        self.specialization_stats['trainer_stats'] = trainer_stats
    
    def create_features(self, df):
        """Conservative, proven features"""
        df = df.copy()
        
        # Specialization
        df = df.merge(
            self.specialization_stats['trainer_course'][['Trainer', 'Course', 'win_rate', 'runs']],
            on=['Trainer', 'Course'], how='left'
        )
        df.rename(columns={'win_rate': 'tc_wr', 'runs': 'tc_exp'}, inplace=True)
        
        df = df.merge(
            self.specialization_stats['jockey_course'][['Jockey', 'Course', 'win_rate', 'runs']],
            on=['Jockey', 'Course'], how='left'
        )
        df.rename(columns={'win_rate': 'jc_wr', 'runs': 'jc_exp'}, inplace=True)
        
        df = df.merge(
            self.specialization_stats['trainer_stats'][['Trainer', 'win_rate', 'place_rate']],
            on='Trainer', how='left'
        )
        df.rename(columns={'win_rate': 'trainer_wr', 'place_rate': 'trainer_pr'}, inplace=True)
        
        df['tc_wr'] = df['tc_wr'].fillna(0.10)
        df['tc_exp'] = df['tc_exp'].fillna(0)
        df['jc_wr'] = df['jc_wr'].fillna(0.10)
        df['jc_exp'] = df['jc_exp'].fillna(0)
        df['trainer_wr'] = df['trainer_wr'].fillna(0.10)
        df['trainer_pr'] = df['trainer_pr'].fillna(0.30)
        
        # Speed
        df['speed_diff'] = df['Speed_PreviousRun'] - df['Speed_2ndPreviousRun']
        df['speed_avg'] = (df['Speed_PreviousRun'] + df['Speed_2ndPreviousRun']) / 2
        df['speed_improving'] = (df['speed_diff'] > 0).astype(int)
        
        # Ratings
        df['team_rating'] = (df['TrainerRating'] + df['JockeyRating']) / 2
        df['bloodline_rating'] = (df['SireRating'] + df['DamsireRating']) / 2
        df['combined_rating'] = (df['team_rating'] + df['bloodline_rating']) / 2
        
        # Market
        df['hist_prob'] = 1 / (df['MarketOdds_PreviousRun'] + 1)
        df['hist_prob_2'] = 1 / (df['MarketOdds_2ndPreviousRun'] + 1)
        df['hist_prob_avg'] = (df['hist_prob'] + df['hist_prob_2']) / 2
        
        # Class/Field
        df['log_prize'] = np.log1p(df['Prize'])
        df['prize_per_runner'] = df['Prize'] / (df['Runners'] + 1)
        df['field_ratio'] = df['Runners'] / (df['meanRunners'] + 1)
        
        # Distance
        df['distance_miles'] = df['distanceYards'] / 1760
        df['sprint'] = (df['distanceYards'] < 1320).astype(int)
        
        # Rest
        df['log_rest'] = np.log1p(df['daysSinceLastRun'])
        df['optimal_rest'] = (df['daysSinceLastRun'].between(14, 28)).astype(int)
        
        # Race-relative (CRITICAL for calibration)
        for col in ['Speed_PreviousRun', 'TrainerRating', 'JockeyRating', 
                   'combined_rating', 'hist_prob_avg']:
            if col in df.columns:
                df[f'{col}_pct'] = df.groupby('Race_ID')[col].rank(pct=True)
                df[f'{col}_z'] = df.groupby('Race_ID')[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-6)
                )
        
        # Key interactions
        df['tc_jc_synergy'] = df['tc_wr'] * df['jc_wr']
        df['speed_rating'] = df['speed_avg'] * df['combined_rating']
        df['market_rating'] = df['hist_prob_avg'] * df['combined_rating']
        df['rating_field'] = df['combined_rating'] * df['field_ratio']
        
        return df
        
    def temperature_scale(self, logits, temperature):
        """Apply temperature scaling"""
        return logits / temperature
    
    def optimize_temperature(self, y_true, logits):
        """Find optimal temperature to minimize NLL"""
        def nll_loss(T):
            scaled = 1 / (1 + np.exp(-logits / T))
            return log_loss(y_true, scaled)
        
        result = minimize(nll_loss, x0=1.0, bounds=[(0.1, 10.0)], method='L-BFGS-B')
        return result.x[0]
        
    def train(self, train_df):
        self.validate_no_leakage(train_df, "training")
        
        # Split into train (70%) and calibration (30%) by races
        race_ids = train_df['Race_ID'].unique()
        race_labels = train_df.groupby('Race_ID')['Position'].apply(
            lambda x: (x == 1).any()
        ).astype(int)
        
        train_races, cal_races = train_test_split(
            race_ids, test_size=0.3, random_state=42, stratify=race_labels
        )
        
        df_train = train_df[train_df['Race_ID'].isin(train_races)].copy()
        df_cal = train_df[train_df['Race_ID'].isin(cal_races)].copy()
        
        print(f"Training: {len(df_train)} samples, Calibration: {len(df_cal)} samples")
        
        # Calculate stats on training set only
        self.calculate_stats(df_train)
        
        # Create features
        df_train = self.create_features(df_train)
        df_cal = self.create_features(df_cal)
        
        y_train = (df_train['Position'] == 1).astype(int)
        y_cal = (df_cal['Position'] == 1).astype(int)
        
        exclude = ['Race_Time', 'Race_ID', 'Horse', 'Trainer', 'Jockey', 
                  'Course', 'Distance', 'Going'] + FORBIDDEN_COLUMNS
        
        self.feature_cols = [col for col in df_train.columns 
                           if col not in exclude and 
                           df_train[col].dtype in ['int64', 'float64']]
        
        X_train = df_train[self.feature_cols].fillna(0)
        X_cal = df_cal[self.feature_cols].fillna(0)
        
        print(f"Training on {len(self.feature_cols)} features")
        
        # Train model on training set
        self.model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=20,
            max_depth=5,
            min_child_samples=35,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=0.2,
            random_state=42,
            verbose=-1
        )
        self.model.fit(X_train, y_train)
        
        # Get logits on calibration set
        cal_probs = self.model.predict_proba(X_cal)[:, 1]
        cal_logits = np.log(cal_probs / (1 - cal_probs + 1e-10))
        
        # Optimize temperature
        print("Optimizing temperature on calibration set...")
        self.temperature = self.optimize_temperature(y_cal, cal_logits)
        print(f"Optimal temperature: {self.temperature:.3f}")
        
        # Evaluate calibration
        scaled_probs = 1 / (1 + np.exp(-cal_logits / self.temperature))
        cal_brier = brier_score_loss(y_cal, scaled_probs)
        raw_brier = brier_score_loss(y_cal, cal_probs)
        print(f"Calibration Brier: {raw_brier:.4f} -> {cal_brier:.4f}")
        
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Features:")
        print(importance.head(15))
        
    def predict(self, test_df):
        self.validate_no_leakage(test_df, "prediction")
        
        test_df = self.create_features(test_df)
        X_test = test_df[self.feature_cols].fillna(0)
        
        # Get raw probabilities
        raw_probs = self.model.predict_proba(X_test)[:, 1]
        
        # Apply temperature scaling
        logits = np.log(raw_probs / (1 - raw_probs + 1e-10))
        scaled_probs = 1 / (1 + np.exp(-logits / self.temperature))
        
        predictions = pd.DataFrame({
            'Race_ID': test_df['Race_ID'],
            'Horse': test_df['Horse'],
            'raw_prob': scaled_probs
        })
        
        # Normalize
        final_predictions = []
        for race_id, race_data in predictions.groupby('Race_ID'):
            race_probs = race_data['raw_prob'].values
            normalized = race_probs / race_probs.sum()
            
            for i, row in enumerate(race_data.itertuples()):
                final_predictions.append({
                    'Race_ID': race_id,
                    'Horse': row.Horse,
                    'Predicted_Probability': normalized[i]
                })
        
        return pd.DataFrame(final_predictions)

def main():
    print("CALIBRATED MODEL - Proper Hold-out Temperature Scaling")
    print("="*60)
    
    train_df = pd.read_csv('data/trainData.csv')
    test_df = pd.read_csv('data/testData.csv')
    
    for col in [c for c in FORBIDDEN_COLUMNS if c != 'Position']:
        if col in train_df.columns:
            train_df = train_df.drop(col, axis=1)
    
    for col in FORBIDDEN_COLUMNS:
        if col in test_df.columns:
            test_df = test_df.drop(col, axis=1)
    
    model = CalibratedFinalModel()
    model.train(train_df)
    
    predictions = model.predict(test_df)
    predictions.to_csv('predictions_calibrated_final.csv', index=False)
    
    print("\nPredictions saved!")

if __name__ == "__main__":
    main()
