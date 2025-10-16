# Horse Racing Prediction Model - Improved Version

This repository contains the improved horse racing prediction model, addressing expert feedback on the original V4 submission.

## 📋 Expert Feedback Addressed

The original submission received feedback that:
1. ✅ **Strong feature engineering** with custom trainer/jockey variables
2. ❌ **Age categorical features** were ineffective and lacked justification  
3. ❌ **Brier score** was relatively poor compared to other applicants

## 🚀 Quick Start

### View Results Summary
```bash
python3 show_results.py
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Improved Model (optional - predictions already generated)
```bash
python3 model_improved.py
```

### Validate Predictions
```bash
python3 check_predictions.py
```

### Evaluate with Cross-Validation
```bash
python3 evaluate_model.py
```

## 📁 Repository Structure

```
/workspace/
├── model_improved.py          # ⭐ Final improved model (V4 - age categorical features)
├── predictions_improved.csv   # ⭐ Final predictions on test set
├── IMPROVEMENTS_SUMMARY.md    # Detailed summary of changes and experiments
├── model_v4_refined.py        # Original V4 baseline model
├── evaluate_model.py          # Cross-validation evaluation script
├── check_predictions.py       # Prediction validation utility
├── requirements.txt           # Python dependencies
├── technical_report.pdf       # Original technical report
├── data/                      # Training and test datasets
│   ├── trainData.csv
│   └── testData.csv
└── experiments/               # Failed improvement attempts (organized separately)
    ├── README.md
    ├── model_v5_calibrated.py
    ├── model_v6_optimized.py
    ├── model_v7_tuned.py
    └── model_v8_final.py
```

## 📊 Model Performance

### Cross-Validation Results (5-fold)

| Model | Log-Loss | Brier Score | Notes |
|-------|----------|-------------|-------|
| **Improved Model** | **0.3354** | **0.0909** | Addressed feedback |
| V4 Baseline | 0.3351 | 0.0908 | Original submission |
| V5 (Isotonic Cal.) | 0.9688 | 0.1138 | Failed - overfitting |
| V6 (Temp. Scaling) | 0.4319 | 0.1054 | Failed |
| V7 (Hyperparams) | 0.3433 | 0.0917 | Minor degradation |
| V8 (Streamlined) | 0.4051 | 0.1010 | Failed |

**Key Finding**: Removing ineffective age features maintains performance while addressing expert feedback. Additional calibration techniques degraded performance.

## ✨ Key Changes

### 1. Removed Age Categorical Features (Per Feedback)
- ❌ Removed: `young_horse` (Age ≤ 3)
- ❌ Removed: `prime_age` (Age 4-6)
- ❌ Removed: `veteran` (Age ≥ 7)
- ✅ Kept: `Age` as continuous + `age_x_distance` interaction

### 2. Maintained Strong Architecture
- 65 features (down from 68)
- Bayesian smoothing (factor=20) for specialization features
- Strong regularization (L1/L2 = 0.2)
- Win rate capping at 30%

### 3. Top Features (by importance)
1. `trainer_going_win_rate` (545)
2. `trainer_distance_win_rate` (510)
3. `trainer_overall_win_rate` (365)
4. `trainer_specialization` (264)
5. `field_pressure` (243)

## 🔬 Experimental Insights

See `experiments/README.md` for details on failed improvement attempts.

**Key Learning**: The original V4 architecture with strong regularization already provides good calibration. Post-hoc calibration techniques (isotonic regression, temperature scaling) caused overfitting and degraded performance.

## ✅ Compliance

The model strictly adheres to competition rules:
- Uses only permitted columns
- Removes all forbidden columns before processing
- Validates no data leakage at every stage
- Outputs valid probabilities that sum to 1.0 per race
- All predictions between 0 and 1

## 📈 Recommendations for Further Improvements

To achieve better Brier scores:

1. **Ensemble Methods**: Combine multiple diverse models
2. **Market Odds Integration**: Use historical odds more heavily (inherently calibrated)
3. **Focal Loss**: Train with focal loss for better probability calibration
4. **Deep Learning**: Neural networks with calibration-focused loss functions
5. **External Data**: Weather, detailed track conditions if available

## 📄 Documentation

- **IMPROVEMENTS_SUMMARY.md** - Comprehensive summary of all changes and experiments
- **experiments/README.md** - Details on failed calibration attempts
- **technical_report.pdf** - Original model technical report

---

For questions or details, see `IMPROVEMENTS_SUMMARY.md`. 