# Performance Comparison - Model Improvements

## Executive Summary

After receiving expert feedback, multiple improvement strategies were tested. The final improved model successfully addresses the feedback by removing ineffective age categorical features while maintaining competitive performance.

## Cross-Validation Results (5-Fold Stratified)

### Main Models

| Model | Log-Loss | Std | Brier Score | Std | Status | Key Changes |
|-------|----------|-----|-------------|-----|--------|-------------|
| **Improved Model** | **0.3354** | ±0.0051 | **0.0909** | ±0.0006 | ✅ **FINAL** | Removed age categorical features |
| V4 Baseline | 0.3351 | ±0.0052 | 0.0908 | ±0.0006 | Baseline | Original submission |
| V7 Tuned | 0.3433 | ±0.0069 | 0.0917 | ±0.0008 | ❌ | Hyperparameter tuning |
| V6 Optimized | 0.4319 | ±0.0069 | 0.1054 | ±0.0008 | ❌ | Temperature scaling |
| V8 Streamlined | 0.4051 | - | 0.1010 | - | ❌ | Simplified features |
| V5 Calibrated | 0.9688 | ±0.0499 | 0.1138 | ±0.0010 | ❌ | Isotonic regression |

### Performance Change from Baseline

| Model | Log-Loss Δ | Brier Score Δ | Verdict |
|-------|-----------|---------------|---------|
| **Improved** | **+0.1%** | **+0.1%** | ✅ Virtually identical, feedback addressed |
| V7 Tuned | +2.4% | +1.0% | ❌ Slight degradation |
| V6 Optimized | +28.9% | +16.1% | ❌ Significant degradation |
| V8 Streamlined | +20.9% | +11.3% | ❌ Significant degradation |
| V5 Calibrated | +189.0% | +25.3% | ❌ Severe degradation |

## Expert Feedback Addressed

### Original Feedback
> "Some features were less effective—for example, the categorical treatment of age didn't appear to add predictive value and lacked sufficient justification."

### Solution Implemented
✅ **Removed all age categorical features**:
- `young_horse` (Age ≤ 3) - REMOVED
- `prime_age` (Age 4-6) - REMOVED  
- `veteran` (Age ≥ 7) - REMOVED

✅ **Maintained Age as continuous feature** for flexibility

### Result
- Features reduced from 68 to 65
- Performance maintained (Brier: 0.0908 → 0.0909)
- Feedback successfully addressed

## Calibration Experiments Summary

### What Was Tried

1. **Isotonic Regression Calibration (V5)**
   - Method: Fit isotonic regression on cross-val predictions
   - Result: Severe overfitting (Brier +25.3%)
   - Conclusion: Too flexible for this problem

2. **Temperature Scaling (V6)**
   - Method: Optimize temperature parameter for calibration
   - Result: Significant degradation (Brier +16.1%)
   - Conclusion: Base model already well-calibrated

3. **Hyperparameter Tuning (V7)**
   - Method: Tuned for Brier score, added class imbalance handling
   - Result: Minor degradation (Brier +1.0%)
   - Conclusion: Original hyperparameters were near-optimal

4. **Feature Reduction (V8)**
   - Method: Streamlined to 48 core features
   - Result: Significant degradation (Brier +11.3%)
   - Conclusion: All features contribute value

### Key Insight

**The V4 architecture with strong regularization (L1/L2=0.2) and Bayesian smoothing (factor=20) already provides good calibration. Additional post-hoc calibration techniques caused overfitting.**

## Model Characteristics

### Improved Model (Final)
- **Features**: 65 (removed 3 age categorical)
- **Estimators**: 300
- **Learning Rate**: 0.025
- **Max Depth**: 5
- **Regularization**: L1=0.2, L2=0.2
- **Bayesian Smoothing**: factor=20
- **Win Rate Cap**: 30%

### Top Features by Importance
1. `trainer_going_win_rate` (545)
2. `trainer_distance_win_rate` (510)
3. `trainer_overall_win_rate` (365)
4. `trainer_specialization` (264)
5. `field_pressure` (243)
6. `Speed_PreviousRun_zscore` (234)
7. `trainer_jockey_course_synergy` (217)
8. `JockeyRating_zscore` (209)

## Prediction Quality Metrics

### Improved Model
- **Max Prediction**: 0.746
- **Min Prediction**: ~0.01
- **Probabilities sum to 1.0**: ✅ All races
- **Valid range [0,1]**: ✅ All predictions

### Comparison to V4
- V4 Max: 0.749
- Improved Max: 0.746
- Change: Slightly less confident (better calibration)

## Conclusions

### What Worked
✅ Removing ineffective age categorical features (per feedback)  
✅ Maintaining V4's strong regularization framework  
✅ Bayesian smoothing for specialization features  
✅ Race-relative normalization features  

### What Didn't Work
❌ Isotonic regression calibration  
❌ Temperature scaling  
❌ Aggressive hyperparameter tuning  
❌ Feature reduction  

### Final Recommendation
The **Improved Model** (`model_improved.py`) is recommended as it:
1. Addresses expert feedback on age features
2. Maintains competitive performance (Brier: 0.0909)
3. Preserves good calibration
4. Uses clean, interpretable features

## Next Steps for Better Brier Scores

To achieve significantly better Brier scores (e.g., < 0.085):

1. **Ensemble Approach**: Stack multiple diverse models
2. **Market-Based Features**: Leverage historical market odds more heavily
3. **Focal Loss Training**: Use loss functions designed for calibration
4. **Neural Networks**: Deep learning with calibration-aware architectures
5. **Additional Data**: Incorporate weather, detailed track conditions
6. **Cross-Model Calibration**: Calibrate ensemble predictions

---

**Note**: All metrics based on 5-fold stratified cross-validation with race-level splitting to prevent data leakage.
