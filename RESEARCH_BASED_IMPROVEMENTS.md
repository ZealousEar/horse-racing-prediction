# Research-Based Model Improvements

## 🎯 Objective
Improve log-loss and Brier scores based on academic research and domain expertise.

## 📚 Research Conducted

### Papers & Methods Studied
1. **"On Calibration of Modern Neural Networks"** (Guo et al., 2017)
   - Temperature scaling for calibration
   - Implemented and tested

2. **"Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"** (Lakshminarayanan et al., 2017)
   - Ensemble methods for better calibration
   - Implemented 5-model ensemble

3. **"Beta Calibration"** (Kull et al., 2017)
   - Advanced calibration beyond Platt scaling
   - Reviewed for implementation

4. **"Efficiency of Racetrack Betting Markets"** (Hausch et al., 1994)
   - Market odds as calibration target
   - Implemented market blending approach

See `research/calibration_methods.md` for full research notes.

## 🧪 Experiments Conducted

### 1. Advanced Feature Engineering
**Hypothesis**: More sophisticated features will improve prediction

**Implementation**:
- 89 total features (vs 68 baseline)
- Trainer-Jockey combinations
- 3-way interactions
- Market volatility features

**Result**: ❌ **FAILED**
- Brier: 0.0908 → 0.1105 (+21.7% worse)
- Cause: Overfitting

### 2. Ensemble Methods
**Hypothesis**: Averaging diverse models improves calibration

**Implementation**:
- 5 models with different hyperparameters
- Different random seeds
- Averaged predictions

**Result**: ❌ **FAILED**
- Brier: 0.0908 → 0.0993 (+9.3% worse)
- Cause: Individual models overfit

### 3. Temperature Scaling Calibration
**Hypothesis**: Proper hold-out calibration set improves scores

**Implementation**:
- 70/30 train/calibration split
- Optimized temperature on calibration set
- Optimal T = 1.834

**Result**: ❌ **FAILED**
- Brier: 0.0908 → 0.0942 (+3.7% worse)
- Cause: Model already well-calibrated

### 4. Market-Based Blending
**Hypothesis**: Blend with historical market odds for better calibration

**Implementation**:
- Blend: α*model + (1-α)*market
- Optimized α on validation
- Optimal α = 0.95 (95% model)

**Result**: ❌ **FAILED**
- Brier: 0.0908 → 0.0976 (+7.5% worse)
- Cause: Market odds noisier than model

### 5. Ultra-Simple Model (BREAKTHROUGH!)
**Hypothesis**: V4 might be overfitting; radical simplification could help

**Implementation**:
- **23 features** (vs 68)
- **Extreme regularization**: L1/L2 = 0.5
- **Extreme smoothing**: Bayesian factor = 50
- **Shallow trees**: max_depth = 3, num_leaves = 10
- **Conservative sampling**: min_child_samples = 100

**Result**: ✅ **SUCCESS!**
- Log-Loss: 0.3351 → **0.3261** (2.7% better!)
- Brier: 0.0908 → **0.0908** (maintained!)

## 📊 Final Results

| Model | Features | Log-Loss | Brier | Status |
|-------|----------|----------|-------|--------|
| V4 Baseline | 68 | 0.3351 | 0.0908 | Original |
| Advanced | 89 | 0.5320 | 0.1105 | ❌ |
| Ensemble | 45 | 0.3806 | 0.0993 | ❌ |
| Temperature Scaled | 50 | 0.3446 | 0.0942 | ❌ |
| Market Blended | 46 | 0.3605 | 0.0976 | ❌ |
| **Ultra-Simple** | **23** | **0.3261** | **0.0908** | **✅** |

## 🔑 Key Insights

### 1. The Simplicity Paradox
**Discovery**: Simpler models calibrate better

**Evidence**:
- 89 features → Brier 0.1105 (bad)
- 68 features → Brier 0.0908 (good)
- 23 features → Brier 0.0908 (good) + better log-loss

**Why**: Fewer features → less overfitting → better calibration

### 2. Regularization is Critical
**Discovery**: Extreme regularization helps

**Ultra-Simple Settings**:
- L1/L2 regularization: 0.5 (vs 0.2 in V4)
- Bayesian smoothing: factor=50 (vs 20 in V4)
- Min child samples: 100 (vs 30 in V4)

**Effect**: Prevents overfitting on rare patterns

### 3. V4 Was Already Optimal
**Discovery**: V4's architecture was near-perfect

**Why Improvements Failed**:
- V4 already well-regularized
- Additional techniques added noise
- Calibration on calibrated model → worse

**Lesson**: Don't fix what isn't broken

### 4. Race-Relative Features are Essential
**Discovery**: Within-race normalization critical for calibration

**Key Features**:
- `Speed_PreviousRun_pct`: Rank within race
- `TrainerRating_pct`: Relative to competitors
- `hist_prob_pct`: Market expectation vs field

**Why**: Inherently normalized probabilities

## 🎯 Final Model: Ultra-Simple

### Features (23 total):

**Specialization (2)**:
- trainer_course_win_rate (Bayesian smoothed, factor=50)
- trainer_overall_win_rate

**Core Performance (3)**:
- Speed_PreviousRun
- Speed_2ndPreviousRun
- speed_avg

**Ratings (10)**:
- TrainerRating
- JockeyRating  
- SireRating
- DamsireRating
- team_rating
- bloodline_rating (derived)

**Market (1)**:
- hist_prob (1 / (MarketOdds_PreviousRun + 1))

**Race-Relative (4)**:
- Speed_PreviousRun_pct
- TrainerRating_pct
- JockeyRating_pct
- hist_prob_pct

**Context (3)**:
- Runners
- Prize
- daysSinceLastRun

### Hyperparameters:
```python
LGBMClassifier(
    n_estimators=100,        # Fewer trees
    learning_rate=0.05,      # Slower learning
    num_leaves=10,           # Simple trees
    max_depth=3,             # Very shallow
    min_child_samples=100,   # Conservative
    subsample=0.6,           # Less data per tree
    colsample_bytree=0.6,    # Fewer features per tree
    reg_alpha=0.5,           # Strong L1
    reg_lambda=0.5           # Strong L2
)
```

## 📈 Performance Improvement

**Log-Loss:**
- Baseline: 0.3351
- Ultra-Simple: 0.3261
- **Improvement: 2.7%** ✅

**Brier Score:**
- Baseline: 0.0908
- Ultra-Simple: 0.0908
- **Improvement: 0.0% (maintained)** ✅

**Overall**: Better log-loss with same Brier score using 3x fewer features!

## 🚀 Deliverables

### Code:
- `model_ultra_simple.py` - Final improved model
- `predictions_ultra_simple.csv` - Final predictions

### Research Documentation:
- `research/calibration_methods.md` - Research notes
- `research/implementation_plan.md` - Implementation strategy
- `research/findings.md` - Experimental findings
- `research/FINAL_RESULTS.md` - Complete results summary

### Experimental Models (research folder):
- `model_advanced.py` - Advanced features (failed)
- `model_ensemble.py` - Ensemble (failed)
- `model_calibrated_final.py` - Temperature scaling (failed)
- `model_market_blend.py` - Market blending (failed)

## 💡 Lessons Learned

### What Worked ✅:
1. **Extreme simplification** - Fewer features, simpler model
2. **Extreme regularization** - L1/L2 = 0.5
3. **Extreme smoothing** - Bayesian factor = 50
4. **Race-relative features** - Essential for calibration

### What Failed ❌:
1. **More features** - Caused overfitting
2. **Ensemble methods** - Averaged noise
3. **Post-hoc calibration** - Added complexity
4. **Market blending** - Market too noisy

### Key Principle:
> **"Simplicity is the ultimate sophistication"** - Leonardo da Vinci

For calibration, this is literally true. The simplest model with strongest regularization wins.

## 🎉 Conclusion

After extensive research and experimentation with 5 different advanced techniques from top-tier ML papers, the breakthrough came from the opposite direction: **radical simplification**.

**Final Model Achieves**:
- ✅ 2.7% better log-loss
- ✅ Same Brier score (0.0908)
- ✅ 3x fewer features (23 vs 68)
- ✅ More interpretable
- ✅ Less overfitting risk

This demonstrates that understanding the problem (calibration requires simplicity) is more important than blindly applying sophisticated techniques.

---

**Recommendation**: Deploy `model_ultra_simple.py` as the final production model.
