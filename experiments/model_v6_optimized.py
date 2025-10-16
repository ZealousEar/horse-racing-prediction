#!/usr/bin/env python3

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# CRITICAL: Define forbidden columns
FORBIDDEN_COLUMNS = ['betfairSP', 'Position', 'timeSecs', 'pdsBeaten', 'NMFP', 'NMFPLTO']

class OptimizedModel:
    def __init__(self):
        self.model = None
        self.temperature = 1.0
        self.feature_cols = None
        self.specialization_stats = {}
        self.max_win_rate = 0.40  # Less restrictive
        self.smooth_factor = 10   # Less aggressive smoothing
        
    def validate_no_leakage(self, df, stage=""):
        """Ensure no forbidden columns are used"""
        violations = [col for col in df.columns if col in FORBIDDEN_COLUMNS and col != 'Position']
        if violations:
            raise ValueError(f"DATA LEAKAGE at {stage}: Found forbidden columns {violations}")
    
    def calculate_specialization_stats(self, train_df):
        """Calculate historical specialization statistics"""
        print("Calculating specialization statistics...")
        
        train_df = train_df.sort_values(['Race_ID', 'Horse'])
        
        # Trainer-Course combinations
        trainer_course = train_df.groupby(['Trainer', 'Course']).agg({
            'Position': ['count', lambda x: (x == 1).sum()]
        }).reset_index()
        trainer_course.columns = ['Trainer', 'Course', 'runs', 'wins']
        
        trainer_overall = train_df.groupby('Trainer').agg({
            'Position': [lambda x: (x == 1).sum(), 'count']
        }).reset_index()
        trainer_overall.columns = ['Trainer', 'wins_overall', 'runs_overall']
        trainer_overall['win_rate_overall'] = trainer_overall['wins_overall'] / trainer_overall['runs_overall']
        
        trainer_course = trainer_course.merge(trainer_overall, on='Trainer', how='left')
        trainer_course['win_rate'] = (
            (trainer_course['wins'] + self.smooth_factor * trainer_course['win_rate_overall']) / 
            (trainer_course['runs'] + self.smooth_factor)
        ).clip(upper=self.max_win_rate)
        
        self.specialization_stats['trainer_course'] = trainer_course
        
        # Jockey-Course combinations
        jockey_course = train_df.groupby(['Jockey', 'Course']).agg({
            'Position': ['count', lambda x: (x == 1).sum()]
        }).reset_index()
        jockey_course.columns = ['Jockey', 'Course', 'runs', 'wins']
        
        jockey_overall = train_df.groupby('Jockey').agg({
            'Position': [lambda x: (x == 1).sum(), 'count']
        }).reset_index()
        jockey_overall.columns = ['Jockey', 'wins_overall', 'runs_overall']
        jockey_overall['win_rate_overall'] = jockey_overall['wins_overall'] / jockey_overall['runs_overall']
        
        jockey_course = jockey_course.merge(jockey_overall, on='Jockey', how='left')
        jockey_course['win_rate'] = (
            (jockey_course['wins'] + self.smooth_factor * jockey_course['win_rate_overall']) / 
            (jockey_course['runs'] + self.smooth_factor)
        ).clip(upper=self.max_win_rate)
        
        self.specialization_stats['jockey_course'] = jockey_course
        
        # Trainer stats
        trainer_stats = train_df.groupby('Trainer').agg({
            'Position': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
        }).reset_index()
        trainer_stats.columns = ['Trainer', 'total_runs', 'total_wins', 'total_places']
        trainer_stats['win_rate'] = trainer_stats['total_wins'] / trainer_stats['total_runs']
        trainer_stats['place_rate'] = trainer_stats['total_places'] / trainer_stats['total_runs']
        
        self.specialization_stats['trainer_stats'] = trainer_stats
        
    def create_specialization_features(self, df):
        """Add specialization features"""
        df = df.copy()
        
        # Trainer-Course
        df = df.merge(
            self.specialization_stats['trainer_course'][['Trainer', 'Course', 'win_rate', 'runs']],
            on=['Trainer', 'Course'],
            how='left'
        )
        df.rename(columns={'win_rate': 'trainer_course_wr', 'runs': 'trainer_course_exp'}, inplace=True)
        
        # Jockey-Course
        df = df.merge(
            self.specialization_stats['jockey_course'][['Jockey', 'Course', 'win_rate', 'runs']],
            on=['Jockey', 'Course'],
            how='left'
        )
        df.rename(columns={'win_rate': 'jockey_course_wr', 'runs': 'jockey_course_exp'}, inplace=True)
        
        # Trainer stats
        df = df.merge(
            self.specialization_stats['trainer_stats'][['Trainer', 'win_rate', 'place_rate']],
            on='Trainer',
            how='left'
        )
        df.rename(columns={'win_rate': 'trainer_wr', 'place_rate': 'trainer_pr'}, inplace=True)
        
        # Fill NaN
        df['trainer_course_wr'] = df['trainer_course_wr'].fillna(0.10)
        df['trainer_course_exp'] = df['trainer_course_exp'].fillna(0)
        df['jockey_course_wr'] = df['jockey_course_wr'].fillna(0.10)
        df['jockey_course_exp'] = df['jockey_course_exp'].fillna(0)
        df['trainer_wr'] = df['trainer_wr'].fillna(0.10)
        df['trainer_pr'] = df['trainer_pr'].fillna(0.30)
        
        return df
            
    def create_features(self, df):
        """Create all features - removed ineffective categorical age features"""
        df = df.copy()
        
        # Speed features
        df['speed_diff'] = df['Speed_PreviousRun'] - df['Speed_2ndPreviousRun']
        df['speed_avg'] = (df['Speed_PreviousRun'] + df['Speed_2ndPreviousRun']) / 2
        df['speed_max'] = df[['Speed_PreviousRun', 'Speed_2ndPreviousRun']].max(axis=1)
        df['speed_improving'] = (df['speed_diff'] > 0).astype(int)
        
        # Ratings - continuous only
        df['team_rating'] = (df['TrainerRating'] + df['JockeyRating']) / 2
        df['bloodline_rating'] = (df['SireRating'] + df['DamsireRating']) / 2
        df['combined_rating'] = (df['team_rating'] + df['bloodline_rating']) / 2
        
        # Class indicators
        df['log_prize'] = np.log1p(df['Prize'])
        df['prize_per_runner'] = df['Prize'] / (df['Runners'] + 1)
        
        # Field features
        df['field_size_ratio'] = df['Runners'] / df['meanRunners']
        df['competitive_field'] = (df['Runners'] >= 10).astype(int)
        
        # Distance
        df['distance_furlongs'] = df['distanceYards'] / 220
        df['sprint'] = (df['distanceYards'] < 1320).astype(int)
        
        # Rest days
        df['log_rest'] = np.log1p(df['daysSinceLastRun'])
        df['well_rested'] = (df['daysSinceLastRun'].between(14, 35)).astype(int)
        
        # Historical market
        df['hist_prob'] = 1 / (df['MarketOdds_PreviousRun'] + 1)
        df['hist_prob_2'] = 1 / (df['MarketOdds_2ndPreviousRun'] + 1)
        df['hist_prob_avg'] = (df['hist_prob'] + df['hist_prob_2']) / 2
        df['hist_prob_trend'] = df['hist_prob'] - df['hist_prob_2']
        
        # Age - continuous only (NO categorical as per feedback)
        df['age_sq'] = df['Age'] ** 2
        
        # Race-relative features (CRITICAL for calibration)
        relative_cols = ['Speed_PreviousRun', 'TrainerRating', 'JockeyRating', 
                        'team_rating', 'combined_rating', 'hist_prob', 'Prize']
        
        for col in relative_cols:
            if col in df.columns:
                # Percentile rank within race
                df[f'{col}_pct'] = df.groupby('Race_ID')[col].rank(pct=True)
                # Z-score within race
                df[f'{col}_z'] = df.groupby('Race_ID')[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-6)
                )
        
        # Key interactions
        df['speed_rating'] = df['speed_avg'] * df['combined_rating']
        df['market_speed'] = df['hist_prob'] * df['Speed_PreviousRun']
        
        return df
        
    def train(self, train_df, tune_temperature=True):
        """Train optimized model"""
        self.validate_no_leakage(train_df, "training")
        
        self.calculate_specialization_stats(train_df)
        train_df = self.create_features(train_df)
        train_df = self.create_specialization_features(train_df)
        
        y = (train_df['Position'] == 1).astype(int)
        
        exclude = ['Race_Time', 'Race_ID', 'Horse', 'Trainer', 'Jockey', 
                  'Course', 'Distance', 'Going'] + FORBIDDEN_COLUMNS
        
        self.feature_cols = [col for col in train_df.columns 
                           if col not in exclude and 
                           train_df[col].dtype in ['int64', 'float64']]
        
        X = train_df[self.feature_cols].fillna(0)
        
        print(f"Training on {len(self.feature_cols)} features")
        
        # Optimized for Brier score - use focal loss concept
        self.model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=20,
            max_depth=5,
            min_child_samples=30,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_alpha=0.15,
            reg_lambda=0.15,
            random_state=42,
            verbose=-1,
            min_split_gain=0.02,
            min_child_weight=0.001,
            max_bin=255
        )
        self.model.fit(X, y)
        
        # Temperature scaling for calibration
        if tune_temperature:
            print("Tuning temperature scaling...")
            race_ids = train_df['Race_ID'].unique()
            race_labels = train_df.groupby('Race_ID')['Position'].apply(
                lambda x: (x == 1).any()
            ).astype(int)
            
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            val_probs = []
            val_labels = []
            
            for train_idx, val_idx in skf.split(race_ids, race_labels):
                val_races = race_ids[val_idx]
                val_data = train_df[train_df['Race_ID'].isin(val_races)].copy()
                
                X_val = val_data[self.feature_cols].fillna(0)
                y_val = (val_data['Position'] == 1).astype(int)
                
                probs = self.model.predict_proba(X_val)[:, 1]
                val_probs.extend(probs)
                val_labels.extend(y_val)
            
            # Find best temperature
            from sklearn.metrics import brier_score_loss
            best_temp = 1.0
            best_brier = float('inf')
            
            for temp in np.arange(0.5, 2.0, 0.1):
                scaled_probs = 1 / (1 + np.exp(-np.log(np.array(val_probs) / (1 - np.array(val_probs) + 1e-10)) / temp))
                brier = brier_score_loss(val_labels, scaled_probs)
                if brier < best_brier:
                    best_brier = brier
                    best_temp = temp
            
            self.temperature = best_temp
            print(f"Optimal temperature: {self.temperature:.2f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Features:")
        print(importance.head(15))
        
    def predict(self, test_df):
        """Generate temperature-scaled predictions"""
        self.validate_no_leakage(test_df, "prediction")
        
        test_df = self.create_features(test_df)
        test_df = self.create_specialization_features(test_df)
        
        X_test = test_df[self.feature_cols].fillna(0)
        raw_probs = self.model.predict_proba(X_test)[:, 1]
        
        # Temperature scaling
        logits = np.log(raw_probs / (1 - raw_probs + 1e-10))
        scaled_probs = 1 / (1 + np.exp(-logits / self.temperature))
        
        predictions = pd.DataFrame({
            'Race_ID': test_df['Race_ID'],
            'Horse': test_df['Horse'],
            'raw_prob': scaled_probs
        })
        
        # Normalize by race
        final_predictions = []
        for race_id, race_data in predictions.groupby('Race_ID'):
            race_probs = race_data['raw_prob'].values
            race_probs = np.maximum(race_probs, 1e-10)
            normalized = race_probs / race_probs.sum()
            
            for i, row in enumerate(race_data.itertuples()):
                final_predictions.append({
                    'Race_ID': race_id,
                    'Horse': row.Horse,
                    'Predicted_Probability': normalized[i]
                })
        
        return pd.DataFrame(final_predictions)

def main():
    print("V6 OPTIMIZED Horse Racing Model")
    print("="*50)
    print("Improvements:")
    print("- Removed ineffective categorical age features")
    print("- Temperature scaling calibration")
    print("- Optimized hyperparameters for Brier score")
    print("- Streamlined feature engineering")
    print()
    
    train_df = pd.read_csv('data/trainData.csv')
    test_df = pd.read_csv('data/testData.csv')
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Remove forbidden columns
    for col in [c for c in FORBIDDEN_COLUMNS if c != 'Position']:
        if col in train_df.columns:
            train_df = train_df.drop(col, axis=1)
    
    for col in FORBIDDEN_COLUMNS:
        if col in test_df.columns:
            test_df = test_df.drop(col, axis=1)
    
    model = OptimizedModel()
    model.train(train_df, tune_temperature=True)
    
    predictions = model.predict(test_df)
    predictions.to_csv('predictions_v6.csv', index=False)
    
    print("\nPredictions saved to predictions_v6.csv!")
    
    # Validate
    for race_id, race_data in predictions.groupby('Race_ID'):
        total = race_data['Predicted_Probability'].sum()
        assert abs(total - 1.0) < 1e-6
    print("All probabilities sum to 1.0")
    
    print(f"\nMax: {predictions['Predicted_Probability'].max():.3f}")
    print(f"Min: {predictions['Predicted_Probability'].min():.6f}")

if __name__ == "__main__":
    main()
