#!/usr/bin/env python3
"""
Ensemble Model - Multiple diverse models averaged for better calibration
Based on: "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

FORBIDDEN_COLUMNS = ['betfairSP', 'Position', 'timeSecs', 'pdsBeaten', 'NMFP', 'NMFPLTO']

class EnsembleModel:
    def __init__(self):
        self.models = []
        self.feature_cols = None
        self.specialization_stats = {}
        
    def validate_no_leakage(self, df, stage=""):
        violations = [col for col in df.columns if col in FORBIDDEN_COLUMNS and col != 'Position']
        if violations:
            raise ValueError(f"DATA LEAKAGE at {stage}: Found forbidden columns {violations}")
    
    def calculate_stats(self, train_df):
        """Same as improved model"""
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
        tc['win_rate'] = (tc['wins'] + 20 * tc['win_rate_overall']) / (tc['runs'] + 20)
        
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
        jc['win_rate'] = (jc['wins'] + 20 * jc['win_rate_overall']) / (jc['runs'] + 20)
        
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
        """Conservative feature set from improved model"""
        df = df.copy()
        
        # Specialization
        df = df.merge(
            self.specialization_stats['trainer_course'][['Trainer', 'Course', 'win_rate', 'runs']],
            on=['Trainer', 'Course'], how='left'
        )
        df.rename(columns={'win_rate': 'tc_wr', 'runs': 'tc_runs'}, inplace=True)
        
        df = df.merge(
            self.specialization_stats['jockey_course'][['Jockey', 'Course', 'win_rate', 'runs']],
            on=['Jockey', 'Course'], how='left'
        )
        df.rename(columns={'win_rate': 'jc_wr', 'runs': 'jc_runs'}, inplace=True)
        
        df = df.merge(
            self.specialization_stats['trainer_stats'][['Trainer', 'win_rate', 'place_rate']],
            on='Trainer', how='left'
        )
        df.rename(columns={'win_rate': 'trainer_wr', 'place_rate': 'trainer_pr'}, inplace=True)
        
        df['tc_wr'] = df['tc_wr'].fillna(0.10)
        df['tc_runs'] = df['tc_runs'].fillna(0)
        df['jc_wr'] = df['jc_wr'].fillna(0.10)
        df['jc_runs'] = df['jc_runs'].fillna(0)
        df['trainer_wr'] = df['trainer_wr'].fillna(0.10)
        df['trainer_pr'] = df['trainer_pr'].fillna(0.30)
        
        # Speed
        df['speed_diff'] = df['Speed_PreviousRun'] - df['Speed_2ndPreviousRun']
        df['speed_avg'] = (df['Speed_PreviousRun'] + df['Speed_2ndPreviousRun']) / 2
        
        # Ratings
        df['team_rating'] = (df['TrainerRating'] + df['JockeyRating']) / 2
        df['bloodline_rating'] = (df['SireRating'] + df['DamsireRating']) / 2
        df['combined_rating'] = (df['team_rating'] + df['bloodline_rating']) / 2
        
        # Market
        df['hist_prob_1'] = 1 / (df['MarketOdds_PreviousRun'] + 1)
        df['hist_prob_2'] = 1 / (df['MarketOdds_2ndPreviousRun'] + 1)
        df['hist_prob_avg'] = (df['hist_prob_1'] + df['hist_prob_2']) / 2
        
        # Field
        df['field_ratio'] = df['Runners'] / (df['meanRunners'] + 1)
        df['log_prize'] = np.log1p(df['Prize'])
        
        # Distance & rest
        df['log_distance'] = np.log1p(df['distanceYards'])
        df['log_rest'] = np.log1p(df['daysSinceLastRun'])
        
        # Race-relative (key for calibration)
        for col in ['Speed_PreviousRun', 'TrainerRating', 'JockeyRating', 
                   'combined_rating', 'hist_prob_avg']:
            if col in df.columns:
                df[f'{col}_pct'] = df.groupby('Race_ID')[col].rank(pct=True)
                df[f'{col}_z'] = df.groupby('Race_ID')[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-6)
                )
        
        # Simple interactions
        df['tc_jc_synergy'] = df['tc_wr'] * df['jc_wr']
        df['speed_rating'] = df['speed_avg'] * df['combined_rating']
        df['market_rating'] = df['hist_prob_avg'] * df['combined_rating']
        
        return df
        
    def train(self, train_df):
        self.validate_no_leakage(train_df, "training")
        
        self.calculate_stats(train_df)
        train_df = self.create_features(train_df)
        
        y = (train_df['Position'] == 1).astype(int)
        
        exclude = ['Race_Time', 'Race_ID', 'Horse', 'Trainer', 'Jockey', 
                  'Course', 'Distance', 'Going'] + FORBIDDEN_COLUMNS
        
        self.feature_cols = [col for col in train_df.columns 
                           if col not in exclude and 
                           train_df[col].dtype in ['int64', 'float64']]
        
        X = train_df[self.feature_cols].fillna(0)
        
        print(f"Training ensemble on {len(self.feature_cols)} features")
        
        # Train diverse models
        model_configs = [
            {'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 20, 'max_depth': 4, 'seed': 42},
            {'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 25, 'max_depth': 5, 'seed': 123},
            {'n_estimators': 250, 'learning_rate': 0.04, 'num_leaves': 30, 'max_depth': 5, 'seed': 456},
            {'n_estimators': 200, 'learning_rate': 0.06, 'num_leaves': 15, 'max_depth': 4, 'seed': 789},
            {'n_estimators': 350, 'learning_rate': 0.02, 'num_leaves': 20, 'max_depth': 6, 'seed': 999},
        ]
        
        for i, config in enumerate(model_configs):
            print(f"Training model {i+1}/5...")
            model = lgb.LGBMClassifier(
                objective='binary',
                metric='binary_logloss',
                n_estimators=config['n_estimators'],
                learning_rate=config['learning_rate'],
                num_leaves=config['num_leaves'],
                max_depth=config['max_depth'],
                min_child_samples=30,
                subsample=0.75,
                colsample_bytree=0.75,
                reg_alpha=0.15,
                reg_lambda=0.15,
                random_state=config['seed'],
                verbose=-1
            )
            model.fit(X, y)
            self.models.append(model)
        
        print(f"Ensemble of {len(self.models)} models trained")
        
    def predict(self, test_df):
        self.validate_no_leakage(test_df, "prediction")
        
        test_df = self.create_features(test_df)
        X_test = test_df[self.feature_cols].fillna(0)
        
        # Average predictions from all models
        all_probs = []
        for model in self.models:
            probs = model.predict_proba(X_test)[:, 1]
            all_probs.append(probs)
        
        avg_probs = np.mean(all_probs, axis=0)
        
        predictions = pd.DataFrame({
            'Race_ID': test_df['Race_ID'],
            'Horse': test_df['Horse'],
            'raw_prob': avg_probs
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
    print("ENSEMBLE Model - 5 Diverse Models Averaged")
    print("="*60)
    
    train_df = pd.read_csv('data/trainData.csv')
    test_df = pd.read_csv('data/testData.csv')
    
    for col in [c for c in FORBIDDEN_COLUMNS if c != 'Position']:
        if col in train_df.columns:
            train_df = train_df.drop(col, axis=1)
    
    for col in FORBIDDEN_COLUMNS:
        if col in test_df.columns:
            test_df = test_df.drop(col, axis=1)
    
    model = EnsembleModel()
    model.train(train_df)
    
    predictions = model.predict(test_df)
    predictions.to_csv('predictions_ensemble.csv', index=False)
    
    print("\nPredictions saved!")

if __name__ == "__main__":
    main()
