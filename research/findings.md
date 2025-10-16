# Research Findings and Results

## Experiments Conducted

### 1. Advanced Feature Engineering
**Approach**: Added 89 advanced features including:
- Trainer-Jockey combinations
- 3-way interactions
- Market volatility features
- Extended race-relative features

**Result**: ❌ **FAILED**
- Brier: 0.0908 → 0.1105 (+21.7% worse)
- Cause: Overfitting with too many features

### 2. Ensemble Methods (5 diverse models)
**Approach**: Averaged predictions from 5 models with different hyperparameters

**Result**: ❌ **FAILED**
- Brier: 0.0908 → 0.0993 (+9.3% worse)
- Cause: Individual models overfitted, average didn't help

### 3. Temperature Scaling with Hold-out Calibration
**Approach**: Proper hold-out calibration set (70/30 split)
- Optimized temperature on separate calibration set
- Applied scaled predictions

**Result**: ❌ **FAILED**
- Brier: 0.0908 → 0.0942 (+3.7% worse)
- Optimal temperature: 1.834 (model too confident)
- Even with scaling, still worse than baseline

## Key Insights

### Why Everything Failed

1. **V4 Model is Already Well-Calibrated**
   - Strong regularization (L1/L2 = 0.2)
   - Bayesian smoothing (factor = 20)
   - Natural calibration from these techniques

2. **Overfitting Problem**
   - More features → overfitting
   - More models → overfitting
   - More calibration → overfitting

3. **The Calibration Paradox**
   - Trying to calibrate an already calibrated model makes it worse
   - Adding complexity destroys existing calibration

### What the Research Says vs. Reality

**Research says**:
- Ensemble helps calibration ✅ (in general)
- Temperature scaling helps ✅ (in general)
- More features help ✅ (in general)

**Our reality**:
- V4 baseline already implements best practices
- Additional techniques add noise, not signal
- Simpler is better for this problem

## Next Steps

### Approach: Market-Based Blending
Instead of trying to calibrate the model, blend it with historical market odds which are inherently calibrated.

**Formula**:
```python
P_final = alpha * P_model + (1 - alpha) * P_market_historical
```

Where:
- P_model = our model predictions
- P_market_historical = 1 / (historical_odds + 1)
- alpha = blending weight (optimize on validation)

**Rationale**:
- Market odds reflect collective wisdom
- Markets are generally well-calibrated (except favorite-longshot bias)
- Blending leverages both model insights and market calibration

This is the last approach to try.
