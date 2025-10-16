#!/usr/bin/env python3
"""
Market-Blended Model
Blend model predictions with historical market odds for better calibration
Based on: Market odds are generally well-calibrated
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss
import warnings
warnings.filterwarnings('ignore')

FORBIDDEN_COLUMNS = ['betfairSP', 'Position', 'timeSecs', 'pdsBeaten', 'NMFP', 'NMFPLTO']

class MarketBlendModel:
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.specialization_stats = {}
        self.blend_alpha = 0.7  # Will optimize this
        
    def validate_no_leakage(self, df, stage=""):
        violations = [col for col in df.columns if col in FORBIDDEN_COLUMNS and col != 'Position']
        if violations:
            raise ValueError(f"DATA LEAKAGE at {stage}: Found forbidden columns {violations}")
    
    def calculate_stats(self, train_df):
        """Same as V4 - proven stats"""
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
        tc['win_rate'] = tc['win_rate'].clip(upper=0.30)
        
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
        jc['win_rate'] = jc['win_rate'].clip(upper=0.30)
        
        self.specialization_stats['jockey_course'] = jc
        
        # Trainer overall
        trainer_stats = train_df.groupby('Trainer').agg({
            'Position': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
        }).reset_index()
        trainer_stats.columns = ['Trainer', 'total_runs', 'total_wins', 'total_places']
        trainer_stats['win_rate'] = (trainer_stats['total_wins'] / trainer_stats['total_runs']).clip(upper=0.30)
        trainer_stats['place_rate'] = (trainer_stats['total_places'] / trainer_stats['total_runs']).clip(upper=0.50)
        
        self.specialization_stats['trainer_stats'] = trainer_stats
    
    def create_features(self, df):
        """V4 features - proven to work"""
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
        
        df['tc_wr'] = df['tc_wr'].fillna(0.08)
        df['tc_runs'] = df['tc_runs'].fillna(0)
        df['jc_wr'] = df['jc_wr'].fillna(0.08)
        df['jc_runs'] = df['jc_runs'].fillna(0)
        df['trainer_wr'] = df['trainer_wr'].fillna(0.08)
        df['trainer_pr'] = df['trainer_pr'].fillna(0.25)
        
        # Core features
        df['speed_consistency'] = df['Speed_PreviousRun'] - df['Speed_2ndPreviousRun']
        df['speed_improving'] = (df['speed_consistency'] > 0).astype(int)
        df['speed_avg'] = (df['Speed_PreviousRun'] + df['Speed_2ndPreviousRun']) / 2
        
        df['team_rating'] = (df['TrainerRating'] + df['JockeyRating']) / 2
        df['bloodline_rating'] = (df['SireRating'] + df['DamsireRating']) / 2
        df['combined_rating'] = (df['team_rating'] + df['bloodline_rating']) / 2
        
        df['class_indicator'] = df['Prize'] / 1000
        df['prize_per_runner'] = df['Prize'] / (df['Runners'] + 1)
        df['field_pressure'] = df['Runners'] / df['meanRunners']
        
        df['distance_miles'] = df['distanceYards'] / 1760
        df['sprint_distance'] = (df['distanceYards'] < 1320).astype(int)
        
        df['optimal_rest'] = (df['daysSinceLastRun'].between(14, 28)).astype(int)
        
        df['prev_market_prob'] = 1 / (df['MarketOdds_PreviousRun'] + 1)
        df['prev_market_prob_2nd'] = 1 / (df['MarketOdds_2ndPreviousRun'] + 1)
        df['market_avg'] = (df['prev_market_prob'] + df['prev_market_prob_2nd']) / 2
        
        # Race-relative
        for col in ['Speed_PreviousRun', 'TrainerRating', 'JockeyRating', 'prev_market_prob']:
            if col in df.columns:
                df[f'{col}_percentile'] = df.groupby('Race_ID')[col].rank(pct=True)
                df[f'{col}_zscore'] = df.groupby('Race_ID')[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-6)
                )
        
        # Interactions
        df['speed_x_rest'] = df['Speed_PreviousRun'] * df['optimal_rest']
        df['rating_x_field'] = df['team_rating'] * df['field_pressure']
        df['tc_jc_synergy'] = df['tc_wr'] * df['jc_wr']
        
        return df
        
    def train(self, train_df, optimize_blend=True):
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
        
        print(f"Training on {len(self.feature_cols)} features")
        
        # V4 hyperparameters
        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.025,
            num_leaves=25,
            max_depth=5,
            min_child_samples=30,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.2,
            reg_lambda=0.2,
            random_state=42,
            verbose=-1
        )
        self.model.fit(X, y)
        
        # Optimize blend weight if requested
        if optimize_blend:
            print("Optimizing market blend weight...")
            # Use 3-fold CV to find best alpha
            race_ids = train_df['Race_ID'].unique()
            race_labels = train_df.groupby('Race_ID')['Position'].apply(
                lambda x: (x == 1).any()
            ).astype(int)
            
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            best_alpha = 0.7
            best_brier = float('inf')
            
            for alpha in np.arange(0.3, 1.0, 0.05):
                briers = []
                for train_idx, val_idx in skf.split(race_ids, race_labels):
                    val_races = race_ids[val_idx]
                    val_data = train_df[train_df['Race_ID'].isin(val_races)].copy()
                    
                    X_val = val_data[self.feature_cols].fillna(0)
                    y_val = (val_data['Position'] == 1).astype(int)
                    
                    model_probs = self.model.predict_proba(X_val)[:, 1]
                    market_probs = val_data['market_avg'].fillna(0.10).values
                    
                    blended = alpha * model_probs + (1 - alpha) * market_probs
                    brier = brier_score_loss(y_val, blended)
                    briers.append(brier)
                
                avg_brier = np.mean(briers)
                if avg_brier < best_brier:
                    best_brier = avg_brier
                    best_alpha = alpha
            
            self.blend_alpha = best_alpha
            print(f"Optimal blend: {self.blend_alpha:.2f} * model + {1-self.blend_alpha:.2f} * market")
            print(f"Blend Brier: {best_brier:.4f}")
        
    def predict(self, test_df):
        self.validate_no_leakage(test_df, "prediction")
        
        test_df = self.create_features(test_df)
        X_test = test_df[self.feature_cols].fillna(0)
        
        # Model predictions
        model_probs = self.model.predict_proba(X_test)[:, 1]
        
        # Market predictions (historical)
        market_probs = test_df['market_avg'].fillna(0.10).values
        
        # Blend
        blended_probs = self.blend_alpha * model_probs + (1 - self.blend_alpha) * market_probs
        
        predictions = pd.DataFrame({
            'Race_ID': test_df['Race_ID'],
            'Horse': test_df['Horse'],
            'raw_prob': blended_probs
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
    print("MARKET-BLENDED Model")
    print("="*60)
    
    train_df = pd.read_csv('data/trainData.csv')
    test_df = pd.read_csv('data/testData.csv')
    
    for col in [c for c in FORBIDDEN_COLUMNS if c != 'Position']:
        if col in train_df.columns:
            train_df = train_df.drop(col, axis=1)
    
    for col in FORBIDDEN_COLUMNS:
        if col in test_df.columns:
            test_df = test_df.drop(col, axis=1)
    
    model = MarketBlendModel()
    model.train(train_df, optimize_blend=True)
    
    predictions = model.predict(test_df)
    predictions.to_csv('predictions_market_blend.csv', index=False)
    
    print("\nPredictions saved!")

if __name__ == "__main__":
    main()
