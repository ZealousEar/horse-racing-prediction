#!/usr/bin/env python3
"""
Ultra-Simple Model - Minimum complexity for best calibration
Hypothesis: V4 might still be slightly overfitting. Go even simpler.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

FORBIDDEN_COLUMNS = ['betfairSP', 'Position', 'timeSecs', 'pdsBeaten', 'NMFP', 'NMFPLTO']

class UltraSimpleModel:
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.specialization_stats = {}
        
    def validate_no_leakage(self, df, stage=""):
        violations = [col for col in df.columns if col in FORBIDDEN_COLUMNS and col != 'Position']
        if violations:
            raise ValueError(f"DATA LEAKAGE at {stage}: Found forbidden columns {violations}")
    
    def calculate_stats(self, train_df):
        """Minimal stats with heavy smoothing"""
        train_df = train_df.sort_values(['Race_ID', 'Horse'])
        
        # Just trainer-course with VERY heavy smoothing
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
        # EXTREME smoothing - factor 50!
        tc['win_rate'] = (tc['wins'] + 50 * tc['win_rate_overall']) / (tc['runs'] + 50)
        
        self.specialization_stats['trainer_course'] = tc
        
        # Trainer overall only
        trainer_stats = train_df.groupby('Trainer').agg({
            'Position': ['count', lambda x: (x == 1).sum()]
        }).reset_index()
        trainer_stats.columns = ['Trainer', 'total_runs', 'total_wins']
        trainer_stats['win_rate'] = trainer_stats['total_wins'] / trainer_stats['total_runs']
        
        self.specialization_stats['trainer_stats'] = trainer_stats
    
    def create_features(self, df):
        """Minimal feature set - only the essentials"""
        df = df.copy()
        
        # Just core specialization
        df = df.merge(
            self.specialization_stats['trainer_course'][['Trainer', 'Course', 'win_rate']],
            on=['Trainer', 'Course'], how='left'
        )
        df.rename(columns={'win_rate': 'tc_wr'}, inplace=True)
        
        df = df.merge(
            self.specialization_stats['trainer_stats'][['Trainer', 'win_rate']],
            on='Trainer', how='left'
        )
        df.rename(columns={'win_rate': 'trainer_wr'}, inplace=True)
        
        df['tc_wr'] = df['tc_wr'].fillna(0.10)
        df['trainer_wr'] = df['trainer_wr'].fillna(0.10)
        
        # Core features only
        df['speed_avg'] = (df['Speed_PreviousRun'] + df['Speed_2ndPreviousRun']) / 2
        df['team_rating'] = (df['TrainerRating'] + df['JockeyRating']) / 2
        df['hist_prob'] = 1 / (df['MarketOdds_PreviousRun'] + 1)
        
        # Race-relative (essential for calibration)
        for col in ['Speed_PreviousRun', 'TrainerRating', 'JockeyRating', 'hist_prob']:
            if col in df.columns:
                df[f'{col}_pct'] = df.groupby('Race_ID')[col].rank(pct=True)
        
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
        
        print(f"Training ultra-simple model on {len(self.feature_cols)} features")
        
        # EXTREMELY conservative hyperparameters
        self.model = lgb.LGBMClassifier(
            n_estimators=100,  # Much fewer
            learning_rate=0.05,  # Slower
            num_leaves=10,  # Much simpler trees
            max_depth=3,  # Very shallow
            min_child_samples=100,  # Much more conservative
            subsample=0.6,
            colsample_bytree=0.6,
            reg_alpha=0.5,  # Much stronger regularization
            reg_lambda=0.5,
            random_state=42,
            verbose=-1
        )
        self.model.fit(X, y)
        
        print(f"Model trained with extreme regularization")
        
    def predict(self, test_df):
        self.validate_no_leakage(test_df, "prediction")
        
        test_df = self.create_features(test_df)
        X_test = test_df[self.feature_cols].fillna(0)
        
        raw_probs = self.model.predict_proba(X_test)[:, 1]
        
        predictions = pd.DataFrame({
            'Race_ID': test_df['Race_ID'],
            'Horse': test_df['Horse'],
            'raw_prob': raw_probs
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
    print("ULTRA-SIMPLE Model - Extreme Regularization")
    print("="*60)
    
    train_df = pd.read_csv('data/trainData.csv')
    test_df = pd.read_csv('data/testData.csv')
    
    for col in [c for c in FORBIDDEN_COLUMNS if c != 'Position']:
        if col in train_df.columns:
            train_df = train_df.drop(col, axis=1)
    
    for col in FORBIDDEN_COLUMNS:
        if col in test_df.columns:
            test_df = test_df.drop(col, axis=1)
    
    model = UltraSimpleModel()
    model.train(train_df)
    
    predictions = model.predict(test_df)
    predictions.to_csv('predictions_ultra_simple.csv', index=False)
    
    print("\nPredictions saved!")

if __name__ == "__main__":
    main()
