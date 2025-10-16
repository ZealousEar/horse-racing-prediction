#!/usr/bin/env python3
"""
Advanced Model with Research-Backed Improvements
Based on calibration research and horse racing literature
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

FORBIDDEN_COLUMNS = ['betfairSP', 'Position', 'timeSecs', 'pdsBeaten', 'NMFP', 'NMFPLTO']

class AdvancedModel:
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.specialization_stats = {}
        
    def validate_no_leakage(self, df, stage=""):
        violations = [col for col in df.columns if col in FORBIDDEN_COLUMNS and col != 'Position']
        if violations:
            raise ValueError(f"DATA LEAKAGE at {stage}: Found forbidden columns {violations}")
    
    def calculate_advanced_stats(self, train_df):
        """Calculate advanced specialization statistics"""
        print("Calculating advanced specialization statistics...")
        train_df = train_df.sort_values(['Race_ID', 'Horse'])
        
        # 1. Trainer-Course with Bayesian smoothing
        tc = train_df.groupby(['Trainer', 'Course']).agg({
            'Position': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
        }).reset_index()
        tc.columns = ['Trainer', 'Course', 'runs', 'wins', 'places']
        
        trainer_overall = train_df.groupby('Trainer').agg({
            'Position': [lambda x: (x == 1).sum(), 'count']
        }).reset_index()
        trainer_overall.columns = ['Trainer', 'wins_overall', 'runs_overall']
        trainer_overall['win_rate_overall'] = trainer_overall['wins_overall'] / trainer_overall['runs_overall']
        
        tc = tc.merge(trainer_overall, on='Trainer', how='left')
        # Bayesian smoothing
        tc['win_rate'] = (tc['wins'] + 10 * tc['win_rate_overall']) / (tc['runs'] + 10)
        tc['place_rate'] = tc['places'] / tc['runs']
        
        self.specialization_stats['trainer_course'] = tc
        
        # 2. Jockey-Course
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
        
        # 3. Trainer-Jockey combinations (new!)
        tj = train_df.groupby(['Trainer', 'Jockey']).agg({
            'Position': ['count', lambda x: (x == 1).sum()]
        }).reset_index()
        tj.columns = ['Trainer', 'Jockey', 'runs', 'wins']
        tj['win_rate'] = tj['wins'] / tj['runs']
        
        self.specialization_stats['trainer_jockey'] = tj
        
        # 4. Trainer overall stats
        trainer_stats = train_df.groupby('Trainer').agg({
            'Position': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
        }).reset_index()
        trainer_stats.columns = ['Trainer', 'total_runs', 'total_wins', 'total_places']
        trainer_stats['win_rate'] = trainer_stats['total_wins'] / trainer_stats['total_runs']
        trainer_stats['place_rate'] = trainer_stats['total_places'] / trainer_stats['total_runs']
        
        self.specialization_stats['trainer_stats'] = trainer_stats
        
        # 5. Historical market calibration (use past market odds patterns)
        train_df['hist_market_prob'] = 1 / (train_df['MarketOdds_PreviousRun'] + 1)
        market_calibration = train_df.groupby(pd.cut(train_df['hist_market_prob'], bins=10)).agg({
            'Position': lambda x: (x == 1).mean()
        }).reset_index()
        market_calibration.columns = ['prob_bin', 'actual_win_rate']
        
        self.specialization_stats['market_calibration'] = market_calibration
    
    def create_advanced_features(self, df):
        """Advanced feature engineering based on research"""
        df = df.copy()
        
        # === SPECIALIZATION FEATURES ===
        
        # Trainer-Course
        df = df.merge(
            self.specialization_stats['trainer_course'][['Trainer', 'Course', 'win_rate', 'place_rate', 'runs']],
            on=['Trainer', 'Course'],
            how='left'
        )
        df.rename(columns={'win_rate': 'tc_win_rate', 'place_rate': 'tc_place_rate', 'runs': 'tc_runs'}, inplace=True)
        
        # Jockey-Course
        df = df.merge(
            self.specialization_stats['jockey_course'][['Jockey', 'Course', 'win_rate', 'runs']],
            on=['Jockey', 'Course'],
            how='left'
        )
        df.rename(columns={'win_rate': 'jc_win_rate', 'runs': 'jc_runs'}, inplace=True)
        
        # Trainer-Jockey (NEW)
        df = df.merge(
            self.specialization_stats['trainer_jockey'][['Trainer', 'Jockey', 'win_rate', 'runs']],
            on=['Trainer', 'Jockey'],
            how='left'
        )
        df.rename(columns={'win_rate': 'tj_win_rate', 'runs': 'tj_runs'}, inplace=True)
        
        # Trainer overall
        df = df.merge(
            self.specialization_stats['trainer_stats'][['Trainer', 'win_rate', 'place_rate']],
            on='Trainer',
            how='left'
        )
        df.rename(columns={'win_rate': 'trainer_wr', 'place_rate': 'trainer_pr'}, inplace=True)
        
        # Fill NaN
        df['tc_win_rate'] = df['tc_win_rate'].fillna(0.10)
        df['tc_place_rate'] = df['tc_place_rate'].fillna(0.30)
        df['tc_runs'] = df['tc_runs'].fillna(0)
        df['jc_win_rate'] = df['jc_win_rate'].fillna(0.10)
        df['jc_runs'] = df['jc_runs'].fillna(0)
        df['tj_win_rate'] = df['tj_win_rate'].fillna(0.10)
        df['tj_runs'] = df['tj_runs'].fillna(0)
        df['trainer_wr'] = df['trainer_wr'].fillna(0.10)
        df['trainer_pr'] = df['trainer_pr'].fillna(0.30)
        
        # === SPEED & PERFORMANCE FEATURES ===
        
        df['speed_diff'] = df['Speed_PreviousRun'] - df['Speed_2ndPreviousRun']
        df['speed_avg'] = (df['Speed_PreviousRun'] + df['Speed_2ndPreviousRun']) / 2
        df['speed_max'] = df[['Speed_PreviousRun', 'Speed_2ndPreviousRun']].max(axis=1)
        df['speed_improving'] = (df['speed_diff'] > 0).astype(int)
        df['speed_stable'] = (abs(df['speed_diff']) < 5).astype(int)
        
        # === RATING FEATURES ===
        
        df['team_rating'] = (df['TrainerRating'] + df['JockeyRating']) / 2
        df['bloodline_rating'] = (df['SireRating'] + df['DamsireRating']) / 2
        df['combined_rating'] = (df['team_rating'] + df['bloodline_rating']) / 2
        df['rating_diff'] = df['TrainerRating'] - df['JockeyRating']
        
        # === MARKET-BASED FEATURES (Historical) ===
        
        df['hist_prob_1'] = 1 / (df['MarketOdds_PreviousRun'] + 1)
        df['hist_prob_2'] = 1 / (df['MarketOdds_2ndPreviousRun'] + 1)
        df['hist_prob_avg'] = (df['hist_prob_1'] + df['hist_prob_2']) / 2
        df['hist_prob_max'] = df[['hist_prob_1', 'hist_prob_2']].max(axis=1)
        df['hist_prob_trend'] = df['hist_prob_1'] - df['hist_prob_2']
        df['hist_prob_improving'] = (df['hist_prob_trend'] > 0).astype(int)
        df['hist_prob_stable'] = (abs(df['hist_prob_trend']) < 0.05).astype(int)
        
        # Market volatility
        df['market_volatility'] = abs(df['hist_prob_1'] - df['hist_prob_2'])
        
        # === CLASS & FIELD FEATURES ===
        
        df['log_prize'] = np.log1p(df['Prize'])
        df['prize_per_runner'] = df['Prize'] / (df['Runners'] + 1)
        df['field_size_ratio'] = df['Runners'] / (df['meanRunners'] + 1)
        df['competitive_field'] = (df['Runners'] >= 10).astype(int)
        df['small_field'] = (df['Runners'] <= 7).astype(int)
        
        # === DISTANCE FEATURES ===
        
        df['log_distance'] = np.log1p(df['distanceYards'])
        df['distance_furlongs'] = df['distanceYards'] / 220
        df['sprint'] = (df['distanceYards'] < 1320).astype(int)
        df['marathon'] = (df['distanceYards'] > 2640).astype(int)
        
        # === REST & RECENCY ===
        
        df['log_rest'] = np.log1p(df['daysSinceLastRun'])
        df['fresh'] = (df['daysSinceLastRun'] > 45).astype(int)
        df['quick_return'] = (df['daysSinceLastRun'] < 14).astype(int)
        df['optimal_rest'] = (df['daysSinceLastRun'].between(14, 28)).astype(int)
        
        # === AGE (Continuous only) ===
        
        df['age_sq'] = df['Age'] ** 2
        df['age_normalized'] = (df['Age'] - 5) / 2  # Center around prime
        
        # === RACE-RELATIVE FEATURES (Critical for calibration) ===
        
        key_features = ['Speed_PreviousRun', 'TrainerRating', 'JockeyRating', 
                       'team_rating', 'combined_rating', 'hist_prob_avg', 
                       'bloodline_rating', 'Prize']
        
        for col in key_features:
            if col in df.columns:
                # Percentile
                df[f'{col}_pct'] = df.groupby('Race_ID')[col].rank(pct=True)
                # Z-score
                df[f'{col}_z'] = df.groupby('Race_ID')[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-6)
                )
                # Normalized (0-1 range)
                df[f'{col}_norm'] = df.groupby('Race_ID')[col].transform(
                    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
                )
        
        # === INTERACTION FEATURES ===
        
        # Speed interactions
        df['speed_rating'] = df['speed_avg'] * df['combined_rating']
        df['speed_rest'] = df['Speed_PreviousRun'] * df['log_rest']
        
        # Market interactions
        df['market_rating'] = df['hist_prob_avg'] * df['combined_rating']
        df['market_speed'] = df['hist_prob_avg'] * df['speed_avg']
        
        # Specialization interactions
        df['tc_jc_synergy'] = df['tc_win_rate'] * df['jc_win_rate']
        df['tj_experience'] = np.log1p(df['tj_runs'])
        df['total_specialization'] = (df['tc_win_rate'] + df['jc_win_rate'] + df['tj_win_rate']) / 3
        
        # Rating interactions
        df['rating_field'] = df['combined_rating'] * df['field_size_ratio']
        df['age_distance'] = df['Age'] * df['log_distance']
        
        # 3-way interactions (NEW)
        df['speed_rating_market'] = df['speed_avg'] * df['combined_rating'] * df['hist_prob_avg']
        
        return df
        
    def train(self, train_df):
        self.validate_no_leakage(train_df, "training")
        
        self.calculate_advanced_stats(train_df)
        train_df = self.create_advanced_features(train_df)
        
        y = (train_df['Position'] == 1).astype(int)
        
        exclude = ['Race_Time', 'Race_ID', 'Horse', 'Trainer', 'Jockey', 
                  'Course', 'Distance', 'Going'] + FORBIDDEN_COLUMNS
        
        self.feature_cols = [col for col in train_df.columns 
                           if col not in exclude and 
                           train_df[col].dtype in ['int64', 'float64']]
        
        X = train_df[self.feature_cols].fillna(0)
        
        print(f"Training on {len(self.feature_cols)} advanced features")
        
        # Optimized hyperparameters
        self.model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            min_child_samples=25,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
            min_split_gain=0.01
        )
        self.model.fit(X, y)
        
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 Features:")
        print(importance.head(20))
        
    def predict(self, test_df):
        self.validate_no_leakage(test_df, "prediction")
        
        test_df = self.create_advanced_features(test_df)
        
        X_test = test_df[self.feature_cols].fillna(0)
        raw_probs = self.model.predict_proba(X_test)[:, 1]
        
        predictions = pd.DataFrame({
            'Race_ID': test_df['Race_ID'],
            'Horse': test_df['Horse'],
            'raw_prob': raw_probs
        })
        
        # Normalize by race
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
    print("ADVANCED Model - Research-Based Improvements")
    print("="*60)
    
    train_df = pd.read_csv('data/trainData.csv')
    test_df = pd.read_csv('data/testData.csv')
    
    for col in [c for c in FORBIDDEN_COLUMNS if c != 'Position']:
        if col in train_df.columns:
            train_df = train_df.drop(col, axis=1)
    
    for col in FORBIDDEN_COLUMNS:
        if col in test_df.columns:
            test_df = test_df.drop(col, axis=1)
    
    model = AdvancedModel()
    model.train(train_df)
    
    predictions = model.predict(test_df)
    predictions.to_csv('predictions_advanced.csv', index=False)
    
    print("\nPredictions saved to predictions_advanced.csv!")
    
    for race_id, race_data in predictions.groupby('Race_ID'):
        total = race_data['Predicted_Probability'].sum()
        assert abs(total - 1.0) < 1e-6
    print("All probabilities sum to 1.0")

if __name__ == "__main__":
    main()
