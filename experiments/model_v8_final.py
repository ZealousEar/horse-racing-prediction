#!/usr/bin/env python3

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

FORBIDDEN_COLUMNS = ['betfairSP', 'Position', 'timeSecs', 'pdsBeaten', 'NMFP', 'NMFPLTO']

class FinalModel:
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.specialization_stats = {}
        
    def validate_no_leakage(self, df, stage=""):
        violations = [col for col in df.columns if col in FORBIDDEN_COLUMNS and col != 'Position']
        if violations:
            raise ValueError(f"DATA LEAKAGE at {stage}: Found forbidden columns {violations}")
    
    def calculate_specialization_stats(self, train_df):
        print("Calculating specialization statistics...")
        train_df = train_df.sort_values(['Race_ID', 'Horse'])
        
        # Trainer-Course with moderate smoothing
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
        # Moderate smoothing
        trainer_course['win_rate'] = (
            (trainer_course['wins'] + 5 * trainer_course['win_rate_overall']) / 
            (trainer_course['runs'] + 5)
        )
        
        self.specialization_stats['trainer_course'] = trainer_course
        
        # Jockey-Course
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
            (jockey_course['wins'] + 5 * jockey_course['win_rate_overall']) / 
            (jockey_course['runs'] + 5)
        )
        
        self.specialization_stats['jockey_course'] = jockey_course
        
        # Trainer overall
        trainer_stats = train_df.groupby('Trainer').agg({
            'Position': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
        }).reset_index()
        trainer_stats.columns = ['Trainer', 'total_runs', 'total_wins', 'total_places']
        trainer_stats['win_rate'] = trainer_stats['total_wins'] / trainer_stats['total_runs']
        trainer_stats['place_rate'] = trainer_stats['total_places'] / trainer_stats['total_runs']
        
        self.specialization_stats['trainer_stats'] = trainer_stats
        
    def create_specialization_features(self, df):
        df = df.copy()
        
        df = df.merge(
            self.specialization_stats['trainer_course'][['Trainer', 'Course', 'win_rate']],
            on=['Trainer', 'Course'],
            how='left'
        )
        df.rename(columns={'win_rate': 'trainer_course_wr'}, inplace=True)
        
        df = df.merge(
            self.specialization_stats['jockey_course'][['Jockey', 'Course', 'win_rate']],
            on=['Jockey', 'Course'],
            how='left'
        )
        df.rename(columns={'win_rate': 'jockey_course_wr'}, inplace=True)
        
        df = df.merge(
            self.specialization_stats['trainer_stats'][['Trainer', 'win_rate', 'place_rate']],
            on='Trainer',
            how='left'
        )
        df.rename(columns={'win_rate': 'trainer_wr', 'place_rate': 'trainer_pr'}, inplace=True)
        
        df['trainer_course_wr'] = df['trainer_course_wr'].fillna(0.10)
        df['jockey_course_wr'] = df['jockey_course_wr'].fillna(0.10)
        df['trainer_wr'] = df['trainer_wr'].fillna(0.10)
        df['trainer_pr'] = df['trainer_pr'].fillna(0.30)
        
        return df
            
    def create_features(self, df):
        """Streamlined features - removed age categorical per feedback"""
        df = df.copy()
        
        # Speed features
        df['speed_diff'] = df['Speed_PreviousRun'] - df['Speed_2ndPreviousRun']
        df['speed_avg'] = (df['Speed_PreviousRun'] + df['Speed_2ndPreviousRun']) / 2
        
        # Ratings
        df['team_rating'] = (df['TrainerRating'] + df['JockeyRating']) / 2
        df['bloodline_rating'] = (df['SireRating'] + df['DamsireRating']) / 2
        df['combined_rating'] = (df['team_rating'] + df['bloodline_rating']) / 2
        
        # Historical market probabilities - KEY for calibration
        df['hist_prob_1'] = 1 / (df['MarketOdds_PreviousRun'] + 1)
        df['hist_prob_2'] = 1 / (df['MarketOdds_2ndPreviousRun'] + 1)
        df['hist_prob_avg'] = (df['hist_prob_1'] + df['hist_prob_2']) / 2
        df['hist_prob_max'] = df[['hist_prob_1', 'hist_prob_2']].max(axis=1)
        df['hist_prob_stable'] = (abs(df['hist_prob_1'] - df['hist_prob_2']) < 0.1).astype(int)
        
        # Distance and field
        df['log_distance'] = np.log1p(df['distanceYards'])
        df['field_ratio'] = df['Runners'] / df['meanRunners']
        
        # Rest
        df['log_rest'] = np.log1p(df['daysSinceLastRun'])
        df['well_rested'] = (df['daysSinceLastRun'].between(14, 35)).astype(int)
        
        # Age - continuous only
        df['age_sq'] = df['Age'] ** 2
        
        # CRITICAL: Race-relative features for proper calibration
        key_features = ['Speed_PreviousRun', 'TrainerRating', 'JockeyRating', 
                       'combined_rating', 'hist_prob_avg', 'hist_prob_1', 'team_rating']
        
        for col in key_features:
            if col in df.columns:
                # Within-race percentile
                df[f'{col}_pct'] = df.groupby('Race_ID')[col].rank(pct=True)
                # Within-race normalized score
                df[f'{col}_norm'] = df.groupby('Race_ID')[col].transform(
                    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
                )
        
        # Simple interactions
        df['market_rating'] = df['hist_prob_avg'] * df['combined_rating']
        
        return df
        
    def train(self, train_df):
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
        
        # Optimized for both log-loss and Brier
        self.model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            n_estimators=150,
            learning_rate=0.08,
            num_leaves=16,
            max_depth=5,
            min_child_samples=40,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.2,
            reg_lambda=0.2,
            random_state=42,
            verbose=-1,
            boosting_type='gbdt'
        )
        self.model.fit(X, y)
        
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Features:")
        print(importance.head(15))
        
    def predict(self, test_df):
        self.validate_no_leakage(test_df, "prediction")
        
        test_df = self.create_features(test_df)
        test_df = self.create_specialization_features(test_df)
        
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
    print("V8 FINAL Model")
    print("="*50)
    print("Optimizations:")
    print("- Removed age categorical features")
    print("- Enhanced market probability features")
    print("- Race-relative normalization features")
    print("- Tuned for Brier score")
    print()
    
    train_df = pd.read_csv('data/trainData.csv')
    test_df = pd.read_csv('data/testData.csv')
    
    for col in [c for c in FORBIDDEN_COLUMNS if c != 'Position']:
        if col in train_df.columns:
            train_df = train_df.drop(col, axis=1)
    
    for col in FORBIDDEN_COLUMNS:
        if col in test_df.columns:
            test_df = test_df.drop(col, axis=1)
    
    model = FinalModel()
    model.train(train_df)
    
    predictions = model.predict(test_df)
    predictions.to_csv('predictions_final.csv', index=False)
    
    print("\nPredictions saved to predictions_final.csv!")
    
    for race_id, race_data in predictions.groupby('Race_ID'):
        total = race_data['Predicted_Probability'].sum()
        assert abs(total - 1.0) < 1e-6
    print("All probabilities sum to 1.0")
    
    print(f"\nMax: {predictions['Predicted_Probability'].max():.3f}")
    print(f"Mean: {predictions['Predicted_Probability'].mean():.3f}")

if __name__ == "__main__":
    main()
