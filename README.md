# Irrigation Need Prediction

A Kaggle competition project that predicts a farm field's irrigation need (**Low / Medium / High**) from soil, climate, and crop-management data using gradient-boosted tree models.

## Problem

Given per-field measurements (soil chemistry, weather, crop stage, irrigation setup), classify how much irrigation the field needs. This is a 3-class classification task evaluated on accuracy.

## Data

- `data/train.csv` — labeled training set (target: `Irrigation_Need`)
- `data/test.csv` — unlabeled test set to generate predictions for

**Features:**

| Type | Columns |
|---|---|
| Numeric | `Soil_pH`, `Soil_Moisture`, `Organic_Carbon`, `Electrical_Conductivity`, `Temperature_C`, `Humidity`, `Rainfall_mm`, `Sunlight_Hours`, `Wind_Speed_kmh`, `Field_Area_hectare`, `Previous_Irrigation_mm` |
| Categorical (one-hot) | `Soil_Type`, `Crop_Type`, `Season`, `Irrigation_Type`, `Water_Source`, `Region` |
| Ordinal | `Crop_Growth_Stage` |
| Binary | `Mulching_Used` |

## Approach

Implemented in [`Irrigation_Need.ipynb`](Irrigation_Need.ipynb):

1. **EDA** — inspect shape, dtypes, missing values, and unique categorical values.
2. **Preprocessing**
   - Drop `id` from training features.
   - One-hot encode nominal categorical columns (`pd.get_dummies`).
   - Ordinal-encode `Crop_Growth_Stage` with `OrdinalEncoder`.
   - Map `Mulching_Used` (Yes/No) to 1/0.
   - Map the target `Irrigation_Need` to 0/1/2 (Low/Medium/High).
3. **Modeling** — `GridSearchCV` (5-fold CV, accuracy scoring) over three gradient-boosting classifiers:
   - `XGBClassifier`
   - `CatBoostClassifier`
   - `LGBMClassifier`
4. **Evaluation** — compare each model's best CV accuracy with a bar chart.
5. **Inference** — predict on `data/test.csv` with the best estimator, map predictions back to Low/Medium/High, and write [`submission.csv`](submission.csv).

## Setup

```bash
pip install pandas scikit-learn xgboost lightgbm catboost matplotlib jupyter
```

## Usage

```bash
jupyter notebook Irrigation_Need.ipynb
```

Run all cells top to bottom; this regenerates `submission.csv` in the project root.

## Project structure

```
.
├── Irrigation_Need.ipynb   # EDA, preprocessing, model search, inference
├── submission.csv          # Kaggle submission output
└── data/
    ├── train.csv
    └── test.csv
```
