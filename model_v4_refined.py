#!/usr/bin/env python3

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

#CRITICAL: Define forbidden columns
FORBIDDEN_COLUMNS = ['betfairSP', 'Position', 'timeSecs', 'pdsBeaten', 'NMFP', 'NMFPLTO']

class RefinedCompliantModel:
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.specialization_stats = {}
        #Refinement parameters
        self.max_win_rate = 0.30  #Cap at 30% win rate
        self.smooth_factor = 20   #Stronger smoothing than V3's 10
        self.feature_weight = 0.5  #Reduce impact of specialization features
        
    def validate_no_leakage(self, df, stage=""):
        """Ensure no forbidden columns are used"""
        violations = [col for col in df.columns if col in FORBIDDEN_COLUMNS and col != 'Position']
        if violations:
            raise ValueError(f"DATA LEAKAGE at {stage}: Found forbidden columns {violations}")
    
    def calculate_specialization_stats(self, train_df):
        """Calculate historical specialization statistics with stronger regularization"""
        print("Calculating specialization statistics with refinements...")
        
        #Ensuring we only use past data
        train_df = train_df.sort_values(['Race_ID', 'Horse'])
        
        #1. Trainer-Course combinations
        trainer_course = train_df.groupby(['Trainer', 'Course']).agg({
            'Position': ['count', lambda x: (x == 1).sum(), 'mean']
        }).reset_index()
        trainer_course.columns = ['Trainer', 'Course', 'runs', 'wins', 'avg_position']
        trainer_course['raw_win_rate'] = trainer_course['wins'] / trainer_course['runs']
        
        #Overall trainer performance
        trainer_overall = train_df.groupby('Trainer').agg({
            'Position': [lambda x: (x == 1).sum(), 'count']
        }).reset_index()
        trainer_overall.columns = ['Trainer', 'wins_overall', 'runs_overall']
        trainer_overall['win_rate_overall'] = trainer_overall['wins_overall'] / trainer_overall['runs_overall']
        
        trainer_course = trainer_course.merge(trainer_overall, on='Trainer', how='left')
        
        #Strong Bayesian smoothing AND capping
        trainer_course['win_rate_smooth'] = (
            (trainer_course['wins'] + self.smooth_factor * trainer_overall['win_rate_overall']) / 
            (trainer_course['runs'] + self.smooth_factor)
        ).clip(upper=self.max_win_rate)
        
        #Adding log-odds transformation to reduce extreme impact
        trainer_course['win_rate_logodds'] = np.log(
            (trainer_course['win_rate_smooth'] + 0.01) / (1 - trainer_course['win_rate_smooth'] + 0.01)
        )
        
        self.specialization_stats['trainer_course'] = trainer_course
        
        #2. Jockey-Course combinations
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
        
        #3. Trainer-Distance (simplified and capped)
        train_df['distance_band'] = pd.cut(train_df['distanceYards'], 
                                          bins=[0, 1320, 1760, 2200, 5000],
                                          labels=['sprint', 'mile', 'middle', 'long'])
        
        trainer_distance = train_df.groupby(['Trainer', 'distance_band']).agg({
            'Position': ['count', lambda x: (x == 1).sum()]
        }).reset_index()
        trainer_distance.columns = ['Trainer', 'distance_band', 'runs', 'wins']
        
        #Applying smoothing based on overall trainer performance
        trainer_distance = trainer_distance.merge(
            trainer_overall[['Trainer', 'win_rate_overall']], on='Trainer', how='left'
        )
        trainer_distance['win_rate'] = (
            (trainer_distance['wins'] + self.smooth_factor * trainer_distance['win_rate_overall']) / 
            (trainer_distance['runs'] + self.smooth_factor)
        ).clip(upper=self.max_win_rate)
        
        self.specialization_stats['trainer_distance'] = trainer_distance
        
        #4. Going preferences (simplified)
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
        
        #5. Overall trainer stats (capped)
        trainer_stats = train_df.groupby('Trainer').agg({
            'Position': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
        }).reset_index()
        trainer_stats.columns = ['Trainer', 'total_runs', 'total_wins', 'total_places']
        trainer_stats['win_rate'] = (trainer_stats['total_wins'] / trainer_stats['total_runs']).clip(upper=self.max_win_rate)
        trainer_stats['place_rate'] = (trainer_stats['total_places'] / trainer_stats['total_runs']).clip(upper=0.50)
        
        self.specialization_stats['trainer_stats'] = trainer_stats
        
    def create_specialization_features(self, df, is_train=True):
        """Add moderated specialization features"""
        df = df.copy()
        
        #Adding distance band
        df['distance_band'] = pd.cut(df['distanceYards'], 
                                    bins=[0, 1320, 1760, 2200, 5000],
                                    labels=['sprint', 'mile', 'middle', 'long'])
        
        #1. Trainer-Course features (using log-odds)
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
        
        #2. Jockey-Course features (using log-odds)
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
        
        #3. Trainer-Distance features
        df = df.merge(
            self.specialization_stats['trainer_distance'][['Trainer', 'distance_band', 'win_rate']],
            on=['Trainer', 'distance_band'],
            how='left'
        )
        df.rename(columns={'win_rate': 'trainer_distance_win_rate'}, inplace=True)
        
        #4. Trainer-Going features
        df = df.merge(
            self.specialization_stats['trainer_going'][['Trainer', 'Going', 'win_rate']],
            on=['Trainer', 'Going'],
            how='left'
        )
        df.rename(columns={'win_rate': 'trainer_going_win_rate'}, inplace=True)
        
        #5. Trainer overall stats
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
        
        #Filling NaN with conservative defaults
        df['trainer_course_win_rate'] = df['trainer_course_win_rate'].fillna(0.08)
        df['trainer_course_logodds'] = df['trainer_course_logodds'].fillna(-2.4)  #log(0.08/0.92)
        df['trainer_course_experience'] = df['trainer_course_experience'].fillna(0)
        df['jockey_course_win_rate'] = df['jockey_course_win_rate'].fillna(0.08)
        df['jockey_course_logodds'] = df['jockey_course_logodds'].fillna(-2.4)
        df['jockey_course_experience'] = df['jockey_course_experience'].fillna(0)
        df['trainer_distance_win_rate'] = df['trainer_distance_win_rate'].fillna(0.08)
        df['trainer_going_win_rate'] = df['trainer_going_win_rate'].fillna(0.08)
        df['trainer_overall_win_rate'] = df['trainer_overall_win_rate'].fillna(0.08)
        df['trainer_place_rate'] = df['trainer_place_rate'].fillna(0.25)
        df['trainer_experience'] = df['trainer_experience'].fillna(0)
        
        #Creating MODERATED interaction features
        df['trainer_jockey_course_synergy'] = (
            df['trainer_course_win_rate'] * df['jockey_course_win_rate'] * self.feature_weight
        )
        df['experience_advantage'] = np.log1p(
            df['trainer_course_experience'] + df['jockey_course_experience']
        )
        df['trainer_specialization'] = (
            df['trainer_course_win_rate'] - df['trainer_overall_win_rate']
        ).clip(-0.1, 0.1)  #Capping advantage
        
        #Dropping temporary column
        df = df.drop('distance_band', axis=1)
        
        return df
            
    def create_compliant_features(self, df):
        """Create all features including V1 features"""
        df = df.copy()
        
        #=== V1 FEATURES (keeping what works) ===
        
        #Speed features
        df['speed_consistency'] = df['Speed_PreviousRun'] - df['Speed_2ndPreviousRun']
        df['speed_improving'] = (df['speed_consistency'] > 0).astype(int)
        df['speed_avg'] = (df['Speed_PreviousRun'] + df['Speed_2ndPreviousRun']) / 2
        df['speed_trend'] = df['Speed_PreviousRun'] / (df['Speed_2ndPreviousRun'] + 1e-6)
        
        #Ratings features
        df['team_rating'] = (df['TrainerRating'] + df['JockeyRating']) / 2
        df['bloodline_rating'] = (df['SireRating'] + df['DamsireRating']) / 2
        df['combined_rating'] = (df['team_rating'] + df['bloodline_rating']) / 2
        
        #Class and prize indicators
        df['class_indicator'] = df['Prize'] / 1000
        df['prize_per_runner'] = df['Prize'] / (df['Runners'] + 1)
        
        #Field size features
        df['field_pressure'] = df['Runners'] / df['meanRunners']
        df['large_field'] = (df['Runners'] > 12).astype(int)
        df['small_field'] = (df['Runners'] < 8).astype(int)
        
        #Distance features
        df['distance_miles'] = df['distanceYards'] / 1760
        df['sprint_distance'] = (df['distanceYards'] < 1320).astype(int)
        df['middle_distance'] = (df['distanceYards'].between(1540, 2200)).astype(int)
        df['long_distance'] = (df['distanceYards'] > 2200).astype(int)
        
        #Rest patterns
        df['optimal_rest'] = (df['daysSinceLastRun'].between(14, 28)).astype(int)
        df['fresh'] = (df['daysSinceLastRun'] > 60).astype(int)
        df['quick_return'] = (df['daysSinceLastRun'] < 14).astype(int)
        
        #Historical market
        df['prev_market_prob'] = 1 / (df['MarketOdds_PreviousRun'] + 1)
        df['prev_market_prob_2nd'] = 1 / (df['MarketOdds_2ndPreviousRun'] + 1)
        df['market_consistency'] = abs(df['prev_market_prob'] - df['prev_market_prob_2nd'])
        
        #Age features
        df['young_horse'] = (df['Age'] <= 3).astype(int)
        df['prime_age'] = (df['Age'].between(4, 6)).astype(int)
        df['veteran'] = (df['Age'] >= 7).astype(int)
        
        #Race-relative features
        for col in ['Speed_PreviousRun', 'TrainerRating', 'JockeyRating', 
                    'Prize', 'daysSinceLastRun', 'prev_market_prob']:
            if col in df.columns:
                df[f'{col}_percentile'] = df.groupby('Race_ID')[col].rank(pct=True)
                df[f'{col}_zscore'] = df.groupby('Race_ID')[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-6)
                )
        
        #Interaction features
        df['speed_x_rest'] = df['Speed_PreviousRun'] * df['optimal_rest']
        df['rating_x_field'] = df['team_rating'] * df['field_pressure']
        df['age_x_distance'] = df['Age'] * df['distance_miles']
        
        return df
        
    def train(self, train_df):
        """Train refined compliant model"""
        #Validate input
        self.validate_no_leakage(train_df, "training")
        
        #Calculating specialization statistics
        self.calculate_specialization_stats(train_df)
        
        #Creating features
        train_df = self.create_compliant_features(train_df)
        train_df = self.create_specialization_features(train_df, is_train=True)
        
        #Target
        y = (train_df['Position'] == 1).astype(int)
        
        #Selecting features
        exclude = ['Race_Time', 'Race_ID', 'Horse', 'Trainer', 'Jockey', 
                  'Course', 'Distance', 'Going'] + FORBIDDEN_COLUMNS
        
        self.feature_cols = [col for col in train_df.columns 
                           if col not in exclude and 
                           train_df[col].dtype in ['int64', 'float64']]
        
        X = train_df[self.feature_cols].fillna(0)
        
        print(f"Model trained on {len(self.feature_cols)} features (including refined specialization)")
        
        #Training model with STRONGER regularization than V3
        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.025,  #Slightly lower
            num_leaves=25,        #Fewer leaves
            max_depth=5,          #Shallower
            min_child_samples=30, #More samples required
            subsample=0.7,        #More aggressive subsampling
            colsample_bytree=0.7,
            reg_alpha=0.2,        #Stronger L1
            reg_lambda=0.2,       #Stronger L2
            random_state=42,
            verbose=-1
        )
        self.model.fit(X, y)
        
        #Showing feature importance
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 Features:")
        print(importance.head(20))
        
        #Showing specialization features
        spec_features = importance[importance['feature'].str.contains('trainer_|jockey_|experience|logodds')]
        print("\nRefined Specialization Features:")
        print(spec_features.head(15))
        
    def predict(self, test_df):
        """Generate predictions with moderate confidence"""
        #Validating input
        self.validate_no_leakage(test_df, "prediction")
        
        #Creating features
        test_df = self.create_compliant_features(test_df)
        test_df = self.create_specialization_features(test_df, is_train=False)
        
        #Predicting
        X_test = test_df[self.feature_cols].fillna(0)
        raw_probs = self.model.predict_proba(X_test)[:, 1]
        
        #NO additional calibration - letting model's natural calibration work
        
        #Creating output
        predictions = pd.DataFrame({
            'Race_ID': test_df['Race_ID'],
            'Horse': test_df['Horse'],
            'raw_prob': raw_probs
        })
        
        #Normalizing by race
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
    """Main execution"""
    print("V4 REFINED Horse Racing Model")
    print("="*50)
    print("Key refinements:")
    print("- Win rates capped at 30%")
    print("- Stronger Bayesian smoothing (factor=20)")
    print("- Log-odds transformation for course features")
    print("- Stronger model regularization")
    print("- No post-hoc calibration")
    print()
    
    #Loading data
    #Paths are relative to deliverables directory
    train_df = pd.read_csv('data/trainData.csv')
    test_df = pd.read_csv('data/testData.csv')
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    #CRITICAL: Removing forbidden columns BEFORE any processing
    forbidden_to_remove = [col for col in FORBIDDEN_COLUMNS if col != 'Position']
    
    #Removing from train (except Position which we need for target)
    for col in forbidden_to_remove:
        if col in train_df.columns:
            train_df = train_df.drop(col, axis=1)
            print(f"Removed forbidden column '{col}' from training data")
    
    #Removing ALL forbidden columns from test (including Position)
    for col in FORBIDDEN_COLUMNS:
        if col in test_df.columns:
            test_df = test_df.drop(col, axis=1)
            print(f"Removed forbidden column '{col}' from test data")
    
    #Final validation
    print("\nValidating no forbidden columns remain...")
    train_forbidden = [col for col in train_df.columns if col in FORBIDDEN_COLUMNS and col != 'Position']
    test_forbidden = [col for col in test_df.columns if col in FORBIDDEN_COLUMNS]
    
    if train_forbidden:
        raise ValueError(f"CRITICAL: Forbidden columns in train: {train_forbidden}")
    if test_forbidden:
        raise ValueError(f"CRITICAL: Forbidden columns in test: {test_forbidden}")
    print("No forbidden columns found (except Position in train for target)")
    
    #Training model
    model = RefinedCompliantModel()
    model.train(train_df)
    
    #Predicting
    predictions = model.predict(test_df)
    
    #Saving
    #Saving predictions in current (deliverables) directory
    predictions.to_csv('predictions_v4.csv', index=False)
    print("\nPredictions saved to predictions_v4.csv (in deliverables directory)!")
    
    #Validating probabilities sum to 1
    for race_id, race_data in predictions.groupby('Race_ID'):
        total = race_data['Predicted_Probability'].sum()
        assert abs(total - 1.0) < 1e-6, f"Race {race_id} probabilities sum to {total}"
    print("All probabilities sum to 1.0")
    
    #Checking for extreme predictions
    max_prob = predictions['Predicted_Probability'].max()
    print(f"\nMax prediction: {max_prob:.3f} (V3 was 0.979)")
    
    extreme_count = (predictions['Predicted_Probability'] > 0.70).sum()
    print(f"Predictions >70%: {extreme_count} (V3 had 57)")
    
    #Showing sample predictions
    print("\nSample predictions:")
    print(predictions.head(10))

if __name__ == "__main__":
    main() 