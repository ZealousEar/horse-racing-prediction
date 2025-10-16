#!/usr/bin/env python3
"""
Improved model based on V4 with feedback addressed:
- Removed age categorical features (young_horse, prime_age, veteran)
- Kept everything else that was working
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

FORBIDDEN_COLUMNS = ['betfairSP', 'Position', 'timeSecs', 'pdsBeaten', 'NMFP', 'NMFPLTO']

class ImprovedModel:
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.specialization_stats = {}
        self.max_win_rate = 0.30
        self.smooth_factor = 20
        
    def validate_no_leakage(self, df, stage=""):
        violations = [col for col in df.columns if col in FORBIDDEN_COLUMNS and col != 'Position']
        if violations:
            raise ValueError(f"DATA LEAKAGE at {stage}: Found forbidden columns {violations}")
    
    def calculate_specialization_stats(self, train_df):
        print("Calculating specialization statistics...")
        train_df = train_df.sort_values(['Race_ID', 'Horse'])
        
        # Trainer-Course combinations
        trainer_course = train_df.groupby(['Trainer', 'Course']).agg({
            'Position': ['count', lambda x: (x == 1).sum(), 'mean']
        }).reset_index()
        trainer_course.columns = ['Trainer', 'Course', 'runs', 'wins', 'avg_position']
        trainer_course['raw_win_rate'] = trainer_course['wins'] / trainer_course['runs']
        
        trainer_overall = train_df.groupby('Trainer').agg({
            'Position': [lambda x: (x == 1).sum(), 'count']
        }).reset_index()
        trainer_overall.columns = ['Trainer', 'wins_overall', 'runs_overall']
        trainer_overall['win_rate_overall'] = trainer_overall['wins_overall'] / trainer_overall['runs_overall']
        
        trainer_course = trainer_course.merge(trainer_overall, on='Trainer', how='left')
        trainer_course['win_rate_smooth'] = (
            (trainer_course['wins'] + self.smooth_factor * trainer_overall['win_rate_overall']) / 
            (trainer_course['runs'] + self.smooth_factor)
        ).clip(upper=self.max_win_rate)
        
        trainer_course['win_rate_logodds'] = np.log(
            (trainer_course['win_rate_smooth'] + 0.01) / (1 - trainer_course['win_rate_smooth'] + 0.01)
        )
        
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
        jockey_course['win_rate_smooth'] = (
            (jockey_course['wins'] + self.smooth_factor * jockey_overall['win_rate_overall']) / 
            (jockey_course['runs'] + self.smooth_factor)
        ).clip(upper=self.max_win_rate)
        
        jockey_course['win_rate_logodds'] = np.log(
            (jockey_course['win_rate_smooth'] + 0.01) / (1 - jockey_course['win_rate_smooth'] + 0.01)
        )
        
        self.specialization_stats['jockey_course'] = jockey_course
        
        # Trainer-Distance
        train_df['distance_band'] = pd.cut(train_df['distanceYards'], 
                                          bins=[0, 1320, 1760, 2200, 5000],
                                          labels=['sprint', 'mile', 'middle', 'long'])
        
        trainer_distance = train_df.groupby(['Trainer', 'distance_band']).agg({
            'Position': ['count', lambda x: (x == 1).sum()]
        }).reset_index()
        trainer_distance.columns = ['Trainer', 'distance_band', 'runs', 'wins']
        
        trainer_distance = trainer_distance.merge(
            trainer_overall[['Trainer', 'win_rate_overall']], on='Trainer', how='left'
        )
        trainer_distance['win_rate'] = (
            (trainer_distance['wins'] + self.smooth_factor * trainer_distance['win_rate_overall']) / 
            (trainer_distance['runs'] + self.smooth_factor)
        ).clip(upper=self.max_win_rate)
        
        self.specialization_stats['trainer_distance'] = trainer_distance
        
        # Going preferences
        trainer_going = train_df.groupby(['Trainer', 'Going']).agg({
            'Position': ['count', lambda x: (x == 1).sum()]
        }).reset_index()
        trainer_going.columns = ['Trainer', 'Going', 'runs', 'wins']
        
        trainer_going = trainer_going.merge(
            trainer_overall[['Trainer', 'win_rate_overall']], on='Trainer', how='left'
        )
        trainer_going['win_rate'] = (
            (trainer_going['wins'] + self.smooth_factor * trainer_going['win_rate_overall']) / 
            (trainer_going['runs'] + self.smooth_factor)
        ).clip(upper=self.max_win_rate)
        
        self.specialization_stats['trainer_going'] = trainer_going
        
        # Overall trainer stats
        trainer_stats = train_df.groupby('Trainer').agg({
            'Position': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
        }).reset_index()
        trainer_stats.columns = ['Trainer', 'total_runs', 'total_wins', 'total_places']
        trainer_stats['win_rate'] = (trainer_stats['total_wins'] / trainer_stats['total_runs']).clip(upper=self.max_win_rate)
        trainer_stats['place_rate'] = (trainer_stats['total_places'] / trainer_stats['total_runs']).clip(upper=0.50)
        
        self.specialization_stats['trainer_stats'] = trainer_stats
        
    def create_specialization_features(self, df):
        df = df.copy()
        
        df['distance_band'] = pd.cut(df['distanceYards'], 
                                    bins=[0, 1320, 1760, 2200, 5000],
                                    labels=['sprint', 'mile', 'middle', 'long'])
        
        df = df.merge(
            self.specialization_stats['trainer_course'][['Trainer', 'Course', 'win_rate_smooth', 'win_rate_logodds', 'runs']],
            on=['Trainer', 'Course'],
            how='left',
            suffixes=('', '_tc')
        )
        df.rename(columns={
            'win_rate_smooth': 'trainer_course_win_rate',
            'win_rate_logodds': 'trainer_course_logodds',
            'runs': 'trainer_course_experience'
        }, inplace=True)
        
        df = df.merge(
            self.specialization_stats['jockey_course'][['Jockey', 'Course', 'win_rate_smooth', 'win_rate_logodds', 'runs']],
            on=['Jockey', 'Course'],
            how='left',
            suffixes=('', '_jc')
        )
        df.rename(columns={
            'win_rate_smooth': 'jockey_course_win_rate',
            'win_rate_logodds': 'jockey_course_logodds',
            'runs': 'jockey_course_experience'
        }, inplace=True)
        
        df = df.merge(
            self.specialization_stats['trainer_distance'][['Trainer', 'distance_band', 'win_rate']],
            on=['Trainer', 'distance_band'],
            how='left'
        )
        df.rename(columns={'win_rate': 'trainer_distance_win_rate'}, inplace=True)
        
        df = df.merge(
            self.specialization_stats['trainer_going'][['Trainer', 'Going', 'win_rate']],
            on=['Trainer', 'Going'],
            how='left'
        )
        df.rename(columns={'win_rate': 'trainer_going_win_rate'}, inplace=True)
        
        df = df.merge(
            self.specialization_stats['trainer_stats'][['Trainer', 'win_rate', 'place_rate', 'total_runs']],
            on='Trainer',
            how='left'
        )
        df.rename(columns={
            'win_rate': 'trainer_overall_win_rate',
            'place_rate': 'trainer_place_rate',
            'total_runs': 'trainer_experience'
        }, inplace=True)
        
        df['trainer_course_win_rate'] = df['trainer_course_win_rate'].fillna(0.08)
        df['trainer_course_logodds'] = df['trainer_course_logodds'].fillna(-2.4)
        df['trainer_course_experience'] = df['trainer_course_experience'].fillna(0)
        df['jockey_course_win_rate'] = df['jockey_course_win_rate'].fillna(0.08)
        df['jockey_course_logodds'] = df['jockey_course_logodds'].fillna(-2.4)
        df['jockey_course_experience'] = df['jockey_course_experience'].fillna(0)
        df['trainer_distance_win_rate'] = df['trainer_distance_win_rate'].fillna(0.08)
        df['trainer_going_win_rate'] = df['trainer_going_win_rate'].fillna(0.08)
        df['trainer_overall_win_rate'] = df['trainer_overall_win_rate'].fillna(0.08)
        df['trainer_place_rate'] = df['trainer_place_rate'].fillna(0.25)
        df['trainer_experience'] = df['trainer_experience'].fillna(0)
        
        df['trainer_jockey_course_synergy'] = df['trainer_course_win_rate'] * df['jockey_course_win_rate'] * 0.5
        df['experience_advantage'] = np.log1p(df['trainer_course_experience'] + df['jockey_course_experience'])
        df['trainer_specialization'] = (df['trainer_course_win_rate'] - df['trainer_overall_win_rate']).clip(-0.1, 0.1)
        
        df = df.drop('distance_band', axis=1)
        
        return df
            
    def create_compliant_features(self, df):
        """Same as V4 but WITHOUT age categorical features"""
        df = df.copy()
        
        # Speed features
        df['speed_consistency'] = df['Speed_PreviousRun'] - df['Speed_2ndPreviousRun']
        df['speed_improving'] = (df['speed_consistency'] > 0).astype(int)
        df['speed_avg'] = (df['Speed_PreviousRun'] + df['Speed_2ndPreviousRun']) / 2
        df['speed_trend'] = df['Speed_PreviousRun'] / (df['Speed_2ndPreviousRun'] + 1e-6)
        
        # Ratings features
        df['team_rating'] = (df['TrainerRating'] + df['JockeyRating']) / 2
        df['bloodline_rating'] = (df['SireRating'] + df['DamsireRating']) / 2
        df['combined_rating'] = (df['team_rating'] + df['bloodline_rating']) / 2
        
        # Class and prize indicators
        df['class_indicator'] = df['Prize'] / 1000
        df['prize_per_runner'] = df['Prize'] / (df['Runners'] + 1)
        
        # Field size features
        df['field_pressure'] = df['Runners'] / df['meanRunners']
        df['large_field'] = (df['Runners'] > 12).astype(int)
        df['small_field'] = (df['Runners'] < 8).astype(int)
        
        # Distance features
        df['distance_miles'] = df['distanceYards'] / 1760
        df['sprint_distance'] = (df['distanceYards'] < 1320).astype(int)
        df['middle_distance'] = (df['distanceYards'].between(1540, 2200)).astype(int)
        df['long_distance'] = (df['distanceYards'] > 2200).astype(int)
        
        # Rest patterns
        df['optimal_rest'] = (df['daysSinceLastRun'].between(14, 28)).astype(int)
        df['fresh'] = (df['daysSinceLastRun'] > 60).astype(int)
        df['quick_return'] = (df['daysSinceLastRun'] < 14).astype(int)
        
        # Historical market
        df['prev_market_prob'] = 1 / (df['MarketOdds_PreviousRun'] + 1)
        df['prev_market_prob_2nd'] = 1 / (df['MarketOdds_2ndPreviousRun'] + 1)
        df['market_consistency'] = abs(df['prev_market_prob'] - df['prev_market_prob_2nd'])
        
        # Age features - REMOVED categorical (young_horse, prime_age, veteran)
        # Keep Age as continuous feature only
        
        # Race-relative features
        for col in ['Speed_PreviousRun', 'TrainerRating', 'JockeyRating', 
                    'Prize', 'daysSinceLastRun', 'prev_market_prob']:
            if col in df.columns:
                df[f'{col}_percentile'] = df.groupby('Race_ID')[col].rank(pct=True)
                df[f'{col}_zscore'] = df.groupby('Race_ID')[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-6)
                )
        
        # Interaction features
        df['speed_x_rest'] = df['Speed_PreviousRun'] * df['optimal_rest']
        df['rating_x_field'] = df['team_rating'] * df['field_pressure']
        df['age_x_distance'] = df['Age'] * df['distance_miles']
        
        return df
        
    def train(self, train_df):
        self.validate_no_leakage(train_df, "training")
        
        self.calculate_specialization_stats(train_df)
        train_df = self.create_compliant_features(train_df)
        train_df = self.create_specialization_features(train_df)
        
        y = (train_df['Position'] == 1).astype(int)
        
        exclude = ['Race_Time', 'Race_ID', 'Horse', 'Trainer', 'Jockey', 
                  'Course', 'Distance', 'Going'] + FORBIDDEN_COLUMNS
        
        self.feature_cols = [col for col in train_df.columns 
                           if col not in exclude and 
                           train_df[col].dtype in ['int64', 'float64']]
        
        X = train_df[self.feature_cols].fillna(0)
        
        print(f"Training on {len(self.feature_cols)} features (removed age categorical features)")
        
        # Same hyperparameters as V4
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
        
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Features:")
        print(importance.head(15))
        
    def predict(self, test_df):
        self.validate_no_leakage(test_df, "prediction")
        
        test_df = self.create_compliant_features(test_df)
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
    print("IMPROVED Model - Feedback Addressed")
    print("="*50)
    print("Changes from V4:")
    print("- Removed age categorical features (young_horse, prime_age, veteran)")
    print("- Kept all other working features")
    print()
    
    train_df = pd.read_csv('data/trainData.csv')
    test_df = pd.read_csv('data/testData.csv')
    
    for col in [c for c in FORBIDDEN_COLUMNS if c != 'Position']:
        if col in train_df.columns:
            train_df = train_df.drop(col, axis=1)
    
    for col in FORBIDDEN_COLUMNS:
        if col in test_df.columns:
            test_df = test_df.drop(col, axis=1)
    
    model = ImprovedModel()
    model.train(train_df)
    
    predictions = model.predict(test_df)
    predictions.to_csv('predictions_improved.csv', index=False)
    
    print("\nPredictions saved to predictions_improved.csv!")
    
    for race_id, race_data in predictions.groupby('Race_ID'):
        total = race_data['Predicted_Probability'].sum()
        assert abs(total - 1.0) < 1e-6
    print("All probabilities sum to 1.0")
    
    print(f"\nMax: {predictions['Predicted_Probability'].max():.3f}")

if __name__ == "__main__":
    main()
