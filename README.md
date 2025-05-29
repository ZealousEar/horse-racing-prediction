# Horse Racing Prediction Model V4 - Deliverables

This folder contains all deliverables for the horse racing prediction assignment and is fully self-sufficient for evaluation.

## Contents

- **model_v4_refined.py** - The final V4 model implementation with refined specialization features
- **predictions_v4.csv** - Predictions for all test races (11,276 predictions across 1,216 races)
- **technical_report.pdf** - 2-page technical report describing the model
- **requirements.txt** - Python package dependencies
- **check_predictions.py** - Utility script to validate predictions format and constraints
- **data/** - Directory containing the train and test datasets
  - trainData.csv - Training data used by the model
  - testData.csv - Test data used to generate predictions
- **README.md** - This file

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the model (optional - predictions already provided):
```bash
python model_v4_refined.py
```

3. Validate predictions (optional):
```bash
python check_predictions.py
```

## Model Overview

The V4 Refined model achieves an 8.7% performance gap relative to Betfair through:
- 68 engineered features from 15 permitted columns
- Sophisticated trainer/course specialization features with proper regularization
- LightGBM with enhanced regularization parameters
- No post-hoc calibration (natural calibration preserved)

Key improvements over previous versions:
- Win rates capped at 30% to prevent overconfidence
- Stronger Bayesian smoothing (factor=20)
- Log-odds transformation for extreme value compression
- Maximum prediction of 74.9% (vs 97.9% in overfitted V3)

## Compliance

The model strictly adheres to competition rules:
- Uses only permitted columns
- Removes all forbidden columns before processing
- Validates no data leakage at every stage
- Outputs valid probabilities that sum to 1.0 per race 