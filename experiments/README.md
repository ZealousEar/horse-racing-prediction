# Experimental Models

This folder contains experimental model versions tested during the improvement process.

## Files

### Models Tested:
1. **model_v5_calibrated.py** - Isotonic regression calibration attempt (Brier: 0.1138) ❌
2. **model_v6_optimized.py** - Temperature scaling approach (Brier: 0.1054) ❌
3. **model_v7_tuned.py** - Hyperparameter tuning attempt (Brier: 0.0917) ❌
4. **model_v8_final.py** - Streamlined features attempt (Brier: 0.1010) ❌

### Evaluation Scripts:
- **evaluate_v5.py** - Evaluation for V5 model
- **evaluate_v6.py** - Evaluation for V6 model

### Old Predictions:
- **predictions_v4.csv** - Original V4 predictions

## Key Finding

All calibration attempts **degraded performance** compared to the base V4 model:
- V4 Baseline: Brier = 0.0908
- Best attempt (V7): Brier = 0.0917 (worse)
- Worst attempt (V5): Brier = 0.1138 (much worse)

**Conclusion**: The original V4 architecture with strong regularization already provides good calibration. Additional post-hoc calibration caused overfitting.

## Lessons Learned

1. **Isotonic Regression**: Too flexible, overfits on validation folds
2. **Temperature Scaling**: Helps some models but not this one
3. **Over-tuning**: Aggressive hyperparameter changes hurt generalization
4. **Feature Reduction**: Removing features didn't help calibration

The final improved model (`model_improved.py`) uses the V4 architecture with only the age categorical features removed per expert feedback.
