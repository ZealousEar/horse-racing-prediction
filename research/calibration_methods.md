# Probability Calibration Research

## Key Papers and Methods

### 1. Platt Scaling & Temperature Scaling
**Source**: "On Calibration of Modern Neural Networks" (Guo et al., 2017)

**Key Findings**:
- Modern neural networks tend to be overconfident
- Temperature scaling: `P_calibrated = softmax(logits / T)` where T > 1 reduces confidence
- Platt scaling: Fits a logistic regression on validation set
- **Best Practice**: Use held-out calibration set (not training data)

**Implementation**: 
- Split training into train + calibration set
- Fit model on train
- Optimize temperature T on calibration set to minimize NLL or Brier

### 2. Beta Calibration
**Source**: "Beta calibration: a well-founded and easily implemented improvement on logistic calibration" (Kull et al., 2017)

**Key Insight**: 
- Extends Platt scaling with 3 parameters instead of 2
- Better for datasets where miscalibration is not monotonic
- Formula: Uses beta distribution for calibration map

**For Horse Racing**:
- Useful when extreme probabilities (close to 0 or 1) are poorly calibrated
- Our max prediction is 0.746, so this could help

### 3. Focal Loss for Calibration
**Source**: "Focal Loss for Dense Object Detection" (Lin et al., 2017)

**Formula**: `FL(p_t) = -α(1-p_t)^γ log(p_t)`
- γ = 2 is common (down-weights easy examples)
- α balances positive/negative samples

**For Imbalanced Classification**:
- Horse racing is highly imbalanced (~10% win rate per race)
- Focal loss helps model focus on hard examples
- Better calibration for rare positive class

### 4. Ensemble Calibration
**Source**: "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" (Lakshminarayanan et al., 2017)

**Method**:
- Train multiple models with different initializations
- Average predictions: `P_ensemble = (1/M) Σ P_i`
- Ensembles are naturally better calibrated

**Implementation**:
- Train 5-10 models with different random seeds
- Average their probabilities
- Apply temperature scaling on ensemble

### 5. Histogram Binning
**Source**: Classic calibration method

**How it works**:
- Bin predictions into intervals [0-0.1, 0.1-0.2, ...]
- For each bin, replace predictions with empirical frequency
- Good for visualizing calibration

### 6. Isotonic Regression (Refined)
**Source**: Zadrozny & Elkan (2002)

**Why it failed before**:
- Applied on cross-validation predictions (data leakage in folds)
- Should use completely separate calibration set
- Need more samples for stable fit

**Correct Implementation**:
- Hold out 20% of data for calibration only
- Train on 80%, calibrate on 20%
- More stable than CV approach

## Horse Racing Specific Research

### 1. Market-Based Calibration
**Source**: "Searching for Positive Returns at the Track: A Multinomial Logit Model for Handicapping Horse Races" (Bolton & Chapman, 1986)

**Key Insight**:
- Market odds are generally well-calibrated (favorite-longshot bias aside)
- Use market probabilities as features AND calibration targets
- Blend model predictions with market probabilities

**Favorite-Longshot Bias**:
- Favorites are slightly underbet (better value)
- Longshots are overbet (poor value)
- Our model should correct this

### 2. Bradley-Terry Model for Pairwise Comparisons
**Source**: Bradley-Terry model adapted for racing

**Idea**:
- Model pairwise win probabilities
- Convert to race win probabilities
- Can be more calibrated than direct classification

### 3. Conditional Logit Model
**Source**: "Efficiency of Racetrack Betting Markets" (Hausch et al., 1994)

**For Racing**:
- Model probability of horse i winning race j
- Condition on race-specific features
- Forces probabilities to sum to 1 naturally

## Implementation Strategy

### Approach 1: Proper Calibration Pipeline
1. Split data: 70% train, 15% calibration, 15% validation
2. Train model on 70%
3. Optimize temperature T on 15% calibration
4. Evaluate on 15% validation

### Approach 2: Ensemble with Diversity
1. Train 5 models with:
   - Different random seeds
   - Different feature subsets
   - Different hyperparameters
2. Average predictions
3. Apply temperature scaling

### Approach 3: Focal Loss Training
1. Implement custom focal loss objective
2. Train LightGBM with focal loss
3. Tune γ parameter (try 1.0, 2.0, 3.0)

### Approach 4: Market-Augmented Features
1. Create "market deviation" features
2. Model how actual results deviate from market
3. Blend model with market odds: `P_final = 0.7*P_model + 0.3*P_market`

### Approach 5: Better Feature Engineering
1. **Race pace features**: Early speed indicators
2. **Jockey-Trainer combinations**: Not just course-specific
3. **Class changes**: Moving up/down in class
4. **Weight carried**: If available in allowed columns
5. **Post position bias**: Inside/outside draw advantage

## Expected Improvements

Based on literature:
- Proper calibration: -5-10% Brier score improvement
- Ensembling: -10-15% Brier score improvement  
- Focal loss: -5-8% Brier score improvement
- Market blending: -15-20% Brier score improvement (if market odds available)

**Target**: Reduce Brier from 0.0908 to < 0.080 (12% improvement)
