#!/usr/bin/env python3

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# CRITICAL: Define forbidden columns
FORBIDDEN_COLUMNS = ['betfairSP', 'Position', 'timeSecs', 'pdsBeaten', 'NMFP', 'NMFPLTO']

class CalibratedModel:
    def __init__(self):
        self.model = None
        self.calibrator = None
        self.feature_cols = None
        self.specialization_stats = {}
        # Improved parameters for better calibration
        self.max_win_rate = 0.35  # Slightly less restrictive
        self.smooth_factor = 15   # Balanced smoothing
        
    def validate_no_leakage(self, df, stage=""):
        """Ensure no forbidden columns are used"""
        violations = [col for col in df.columns if col in FORBIDDEN_COLUMNS and col != 'Position']
        if violations:
            raise ValueError(f"DATA LEAKAGE at {stage}: Found forbidden columns {violations}")
    
    def calculate_specialization_stats(self, train_df):
        """Calculate historical specialization statistics"""
        print("Calculating specialization statistics...")
        
        train_df = train_df.sort_values(['Race_ID', 'Horse'])
        
        # 1. Trainer-Course combinations
        trainer_course = train_df.groupby(['Trainer', 'Course']).agg({
            'Position': ['count', lambda x: (x == 1).sum(), 'mean']
        }).reset_index()
        trainer_course.columns = ['Trainer', 'Course', 'runs', 'wins', 'avg_position']
        trainer_course['raw_win_rate'] = trainer_course['wins'] / trainer_course['runs']
        
        # Overall trainer performance
        trainer_overall = train_df.groupby('Trainer').agg({
            'Position': [lambda x: (x == 1).sum(), 'count']
        }).reset_index()
        trainer_overall.columns = ['Trainer', 'wins_overall', 'runs_overall']
        trainer_overall['win_rate_overall'] = trainer_overall['wins_overall'] / trainer_overall['runs_overall']
        
        trainer_course = trainer_course.merge(trainer_overall, on='Trainer', how='left')
        
        # Bayesian smoothing with moderate capping
        trainer_course['win_rate_smooth'] = (
            (trainer_course['wins'] + self.smooth_factor * trainer_course['win_rate_overall']) / 
            (trainer_course['runs'] + self.smooth_factor)
        ).clip(upper=self.max_win_rate)
        
        self.specialization_stats['trainer_course'] = trainer_course
        
        # 2. Jockey-Course combinations
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
            (jockey_course['wins'] + self.smooth_factor * jockey_course['win_rate_overall']) / 
            (jockey_course['runs'] + self.smooth_factor)
        ).clip(upper=self.max_win_rate)
        
        self.specialization_stats['jockey_course'] = jockey_course
        
        # 3. Trainer-Distance
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
        
        # 4. Trainer-Going preferences
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
        
        # 5. Overall trainer stats
        trainer_stats = train_df.groupby('Trainer').agg({
            'Position': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
        }).reset_index()
        trainer_stats.columns = ['Trainer', 'total_runs', 'total_wins', 'total_places']
        trainer_stats['win_rate'] = (trainer_stats['total_wins'] / trainer_stats['total_runs']).clip(upper=self.max_win_rate)
        trainer_stats['place_rate'] = (trainer_stats['total_places'] / trainer_stats['total_runs']).clip(upper=0.50)
        
        self.specialization_stats['trainer_stats'] = trainer_stats
        
    def create_specialization_features(self, df, is_train=True):
        """Add specialization features"""
        df = df.copy()
        
        df['distance_band'] = pd.cut(df['distanceYards'], 
                                    bins=[0, 1320, 1760, 2200, 5000],
                                    labels=['sprint', 'mile', 'middle', 'long'])
        
        # Trainer-Course features
        df = df.merge(
            self.specialization_stats['trainer_course'][['Trainer', 'Course', 'win_rate_smooth', 'runs']],
            on=['Trainer', 'Course'],
            how='left',
            suffixes=('', '_tc')
        )
        df.rename(columns={
            'win_rate_smooth': 'trainer_course_win_rate',
            'runs': 'trainer_course_experience'
        }, inplace=True)
        
        # Jockey-Course features
        df = df.merge(
            self.specialization_stats['jockey_course'][['Jockey', 'Course', 'win_rate_smooth', 'runs']],
            on=['Jockey', 'Course'],
            how='left',
            suffixes=('', '_jc')
        )
        df.rename(columns={
            'win_rate_smooth': 'jockey_course_win_rate',
            'runs': 'jockey_course_experience'
        }, inplace=True)
        
        # Trainer-Distance features
        df = df.merge(
            self.specialization_stats['trainer_distance'][['Trainer', 'distance_band', 'win_rate']],
            on=['Trainer', 'distance_band'],
            how='left'
        )
        df.rename(columns={'win_rate': 'trainer_distance_win_rate'}, inplace=True)
        
        # Trainer-Going features
        df = df.merge(
            self.specialization_stats['trainer_going'][['Trainer', 'Going', 'win_rate']],
            on=['Trainer', 'Going'],
            how='left'
        )
        df.rename(columns={'win_rate': 'trainer_going_win_rate'}, inplace=True)
        
        # Trainer overall stats
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
        
        # Fill NaN with conservative defaults
        df['trainer_course_win_rate'] = df['trainer_course_win_rate'].fillna(0.10)
        df['trainer_course_experience'] = df['trainer_course_experience'].fillna(0)
        df['jockey_course_win_rate'] = df['jockey_course_win_rate'].fillna(0.10)
        df['jockey_course_experience'] = df['jockey_course_experience'].fillna(0)
        df['trainer_distance_win_rate'] = df['trainer_distance_win_rate'].fillna(0.10)
        df['trainer_going_win_rate'] = df['trainer_going_win_rate'].fillna(0.10)
        df['trainer_overall_win_rate'] = df['trainer_overall_win_rate'].fillna(0.10)
        df['trainer_place_rate'] = df['trainer_place_rate'].fillna(0.25)
        df['trainer_experience'] = df['trainer_experience'].fillna(0)
        
        # Interaction features
        df['trainer_jockey_synergy'] = df['trainer_course_win_rate'] * df['jockey_course_win_rate']
        df['total_experience'] = np.log1p(df['trainer_course_experience'] + df['jockey_course_experience'])
        
        df = df.drop('distance_band', axis=1)
        
        return df
            
    def create_compliant_features(self, df):
        """Create effective features - REMOVED age categorical variables"""
        df = df.copy()
        
        # Speed features
        df['speed_consistency'] = df['Speed_PreviousRun'] - df['Speed_2ndPreviousRun']
        df['speed_improving'] = (df['speed_consistency'] > 0).astype(int)
        df['speed_avg'] = (df['Speed_PreviousRun'] + df['Speed_2ndPreviousRun']) / 2
        df['speed_trend'] = df['Speed_PreviousRun'] / (df['Speed_2ndPreviousRun'] + 1e-6)
        df['speed_max'] = df[['Speed_PreviousRun', 'Speed_2ndPreviousRun']].max(axis=1)
        
        # Ratings features
        df['team_rating'] = (df['TrainerRating'] + df['JockeyRating']) / 2
        df['bloodline_rating'] = (df['SireRating'] + df['DamsireRating']) / 2
        df['combined_rating'] = (df['team_rating'] + df['bloodline_rating']) / 2
        df['rating_advantage'] = df['TrainerRating'] - df['JockeyRating']
        
        # Class and prize indicators
        df['class_indicator'] = np.log1p(df['Prize'])
        df['prize_per_runner'] = df['Prize'] / (df['Runners'] + 1)
        
        # Field size features
        df['field_pressure'] = df['Runners'] / df['meanRunners']
        df['large_field'] = (df['Runners'] > 12).astype(int)
        df['small_field'] = (df['Runners'] < 8).astype(int)
        
        # Distance features
        df['distance_miles'] = df['distanceYards'] / 1760
        df['distance_log'] = np.log1p(df['distanceYards'])
        df['sprint_distance'] = (df['distanceYards'] < 1320).astype(int)
        df['middle_distance'] = (df['distanceYards'].between(1540, 2200)).astype(int)
        df['long_distance'] = (df['distanceYards'] > 2200).astype(int)
        
        # Rest patterns
        df['rest_days_log'] = np.log1p(df['daysSinceLastRun'])
        df['optimal_rest'] = (df['daysSinceLastRun'].between(14, 28)).astype(int)
        df['fresh'] = (df['daysSinceLastRun'] > 60).astype(int)
        df['quick_return'] = (df['daysSinceLastRun'] < 14).astype(int)
        
        # Market-based features (historical)
        df['prev_market_prob'] = 1 / (df['MarketOdds_PreviousRun'] + 1)
        df['prev_market_prob_2nd'] = 1 / (df['MarketOdds_2ndPreviousRun'] + 1)
        df['market_consistency'] = abs(df['prev_market_prob'] - df['prev_market_prob_2nd'])
        df['market_improving'] = (df['prev_market_prob'] > df['prev_market_prob_2nd']).astype(int)
        df['avg_market_prob'] = (df['prev_market_prob'] + df['prev_market_prob_2nd']) / 2
        
        # Age as continuous (NOT categorical as per feedback)
        df['age_squared'] = df['Age'] ** 2
        df['age_normalized'] = (df['Age'] - 5) / 3  # Center around typical prime age
        
        # Race-relative features (critical for calibration)
        for col in ['Speed_PreviousRun', 'TrainerRating', 'JockeyRating', 
                    'Prize', 'daysSinceLastRun', 'prev_market_prob', 'team_rating', 
                    'bloodline_rating', 'combined_rating']:
            if col in df.columns:
                df[f'{col}_percentile'] = df.groupby('Race_ID')[col].rank(pct=True)
                df[f'{col}_zscore'] = df.groupby('Race_ID')[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-6)
                )
        
        # Interaction features
        df['speed_x_rest'] = df['Speed_PreviousRun'] * df['optimal_rest']
        df['rating_x_field'] = df['team_rating'] * df['field_pressure']
        df['age_x_distance'] = df['Age'] * df['distance_miles']
        df['speed_x_rating'] = df['speed_avg'] * df['combined_rating']
        df['market_x_speed'] = df['prev_market_prob'] * df['Speed_PreviousRun']
        
        return df
        
    def train(self, train_df, calibrate=True):
        """Train model with optional calibration"""
        self.validate_no_leakage(train_df, "training")
        
        self.calculate_specialization_stats(train_df)
        
        train_df = self.create_compliant_features(train_df)
        train_df = self.create_specialization_features(train_df, is_train=True)
        
        y = (train_df['Position'] == 1).astype(int)
        
        exclude = ['Race_Time', 'Race_ID', 'Horse', 'Trainer', 'Jockey', 
                  'Course', 'Distance', 'Going'] + FORBIDDEN_COLUMNS
        
        self.feature_cols = [col for col in train_df.columns 
                           if col not in exclude and 
                           train_df[col].dtype in ['int64', 'float64']]
        
        X = train_df[self.feature_cols].fillna(0)
        
        print(f"Model trained on {len(self.feature_cols)} features")
        
        # Optimized hyperparameters for better calibration
        self.model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=500,
            learning_rate=0.02,
            num_leaves=31,
            max_depth=6,
            min_child_samples=25,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
            min_split_gain=0.01,
            min_child_weight=0.001
        )
        self.model.fit(X, y)
        
        # Calibration using cross-validation
        if calibrate:
            print("Training isotonic calibrator...")
            race_ids = train_df['Race_ID'].unique()
            race_labels = train_df.groupby('Race_ID')['Position'].apply(
                lambda x: (x == 1).any()
            ).astype(int)
            
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            cal_preds = []
            cal_true = []
            
            for train_idx, cal_idx in skf.split(race_ids, race_labels):
                cal_races = race_ids[cal_idx]
                cal_data = train_df[train_df['Race_ID'].isin(cal_races)].copy()
                
                X_cal = cal_data[self.feature_cols].fillna(0)
                y_cal = (cal_data['Position'] == 1).astype(int)
                
                preds = self.model.predict_proba(X_cal)[:, 1]
                cal_preds.extend(preds)
                cal_true.extend(y_cal)
            
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(cal_preds, cal_true)
            print("Calibration complete")
        
        # Show feature importance
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 Features:")
        print(importance.head(20))
        
    def predict(self, test_df):
        """Generate calibrated predictions"""
        self.validate_no_leakage(test_df, "prediction")
        
        test_df = self.create_compliant_features(test_df)
        test_df = self.create_specialization_features(test_df, is_train=False)
        
        X_test = test_df[self.feature_cols].fillna(0)
        raw_probs = self.model.predict_proba(X_test)[:, 1]
        
        # Apply calibration
        if self.calibrator is not None:
            calibrated_probs = self.calibrator.predict(raw_probs)
        else:
            calibrated_probs = raw_probs
        
        predictions = pd.DataFrame({
            'Race_ID': test_df['Race_ID'],
            'Horse': test_df['Horse'],
            'raw_prob': calibrated_probs
        })
        
        # Normalize by race
        final_predictions = []
        for race_id, race_data in predictions.groupby('Race_ID'):
            race_probs = race_data['raw_prob'].values
            # Ensure all probabilities are positive
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
    """Main execution"""
    print("V5 CALIBRATED Horse Racing Model")
    print("="*50)
    print("Improvements:")
    print("- Removed ineffective age categorical features")
    print("- Added isotonic regression calibration")
    print("- Optimized hyperparameters for calibration")
    print("- Enhanced market-based features")
    print("- Better interaction features")
    print()
    
    # Load data
    train_df = pd.read_csv('data/trainData.csv')
    test_df = pd.read_csv('data/testData.csv')
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Remove forbidden columns
    forbidden_to_remove = [col for col in FORBIDDEN_COLUMNS if col != 'Position']
    
    for col in forbidden_to_remove:
        if col in train_df.columns:
            train_df = train_df.drop(col, axis=1)
            print(f"Removed forbidden column '{col}' from training data")
    
    for col in FORBIDDEN_COLUMNS:
        if col in test_df.columns:
            test_df = test_df.drop(col, axis=1)
            print(f"Removed forbidden column '{col}' from test data")
    
    # Train model
    model = CalibratedModel()
    model.train(train_df, calibrate=True)
    
    # Predict
    predictions = model.predict(test_df)
    
    # Save
    predictions.to_csv('predictions_v5.csv', index=False)
    print("\nPredictions saved to predictions_v5.csv!")
    
    # Validate
    for race_id, race_data in predictions.groupby('Race_ID'):
        total = race_data['Predicted_Probability'].sum()
        assert abs(total - 1.0) < 1e-6, f"Race {race_id} probabilities sum to {total}"
    print("All probabilities sum to 1.0")
    
    # Statistics
    max_prob = predictions['Predicted_Probability'].max()
    min_prob = predictions['Predicted_Probability'].min()
    print(f"\nMax prediction: {max_prob:.3f}")
    print(f"Min prediction: {min_prob:.6f}")
    
    print("\nSample predictions:")
    print(predictions.head(10))

if __name__ == "__main__":
    main()
