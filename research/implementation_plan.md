# Implementation Plan for Score Improvement

## Current Baseline
- **Brier Score**: 0.0908
- **Log Loss**: 0.3351

## Target
- **Brier Score**: < 0.080 (12% improvement)
- **Log Loss**: < 0.300 (10% improvement)

## Implementation Sequence

### Phase 1: Proper Hold-out Calibration (Quick Win)
**Expected Improvement**: -5% Brier

1. Create proper train/calibration/validation split
2. Implement temperature scaling with grid search
3. Implement Platt scaling
4. Compare both methods

**Time**: 15 minutes
**Risk**: Low

### Phase 2: Ensemble Methods (High Impact)
**Expected Improvement**: -10-15% Brier

1. Train 5-7 diverse models:
   - Different random seeds
   - Different learning rates (0.01, 0.03, 0.05)
   - Different max_depth (4, 5, 6)
   - Different num_leaves (15, 25, 35)
   
2. Average predictions
3. Apply temperature scaling on ensemble

**Time**: 30 minutes
**Risk**: Low

### Phase 3: Focal Loss Training (Medium Impact)
**Expected Improvement**: -5-8% Brier

1. Implement focal loss for LightGBM
2. Use custom objective function
3. Tune gamma parameter (1.0, 2.0, 3.0)

**Time**: 20 minutes
**Risk**: Medium (need to implement custom loss)

### Phase 4: Advanced Feature Engineering (High Impact)
**Expected Improvement**: -10% Brier

1. **Interaction Features**:
   - Trainer-Jockey-Course 3-way interactions
   - Speed * Distance interactions
   - Recent form momentum

2. **Temporal Features**:
   - Win streak indicators
   - Consistency metrics (std of positions)
   - Last 3 race average position

3. **Competitive Context**:
   - Strength of field (avg competitor rating)
   - Position in betting market relative to ability

**Time**: 25 minutes
**Risk**: Low

### Phase 5: Market-Based Calibration (Highest Impact)
**Expected Improvement**: -15-20% Brier

**Problem**: We don't have real-time market odds (betfairSP is forbidden)

**Solution**: Use HISTORICAL market odds as calibration target
1. Historical odds show what well-calibrated probabilities look like
2. Train model to match historical market patterns
3. Create "market expectation" features from historical data

**Implementation**:
```python
# Use past market odds to calibrate future predictions
hist_market_prob = 1 / (historical_odds + 1)
calibration_target = 0.7 * actual_win + 0.3 * hist_market_prob
```

**Time**: 20 minutes
**Risk**: Medium

### Phase 6: Stacking Meta-Model (Advanced)
**Expected Improvement**: -5-10% additional

1. Use base model predictions as features
2. Train meta-model on calibration set
3. Meta-model learns to correct base model errors

**Time**: 20 minutes
**Risk**: Medium

## Total Expected Improvement

Conservative estimate:
- Phase 1 (Calibration): -5%
- Phase 2 (Ensemble): -10%
- Phase 4 (Features): -8%

**Expected Final Brier**: 0.0908 * (1 - 0.23) = **0.070** ✅

Optimistic estimate (if all work):
- All phases combined: -30-35%
- **Expected Final Brier**: 0.0908 * 0.65 = **0.059** 🎯

## Execution Order

1. ✅ Phase 4 (Features) - Build better base model
2. ✅ Phase 2 (Ensemble) - Combine diverse models  
3. ✅ Phase 1 (Calibration) - Proper temperature scaling
4. ✅ Phase 3 (Focal Loss) - If time permits
5. ✅ Phase 5 (Market-based) - If time permits

## Success Metrics

- Brier Score < 0.080: ✅ Success
- Brier Score < 0.075: ✅✅ Great success
- Brier Score < 0.070: ✅✅✅ Excellent
- Brier Score < 0.065: 🏆 Outstanding

Let's implement these in order and track improvements!
