# Addiction Population Data — Predictive Modelling

A machine-learning project that uses the `addiction_population_data.csv` dataset (3 000 individuals, 25 features) to:

1. **Classify** whether an individual has health issues (`has_health_issues`)
2. **Regress** the number of cigarettes smoked per day (`smokes_per_day`)

---

## Project Structure

```
.
├── addiction_population_data.csv   # Raw dataset
├── addiction_model.ipynb           # Full ML pipeline notebook
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── Project_Report.docx             # Detailed project report
├── best_classifier.pkl             # Saved best classification pipeline (generated on run)
└── best_regressor.pkl              # Saved best regression pipeline (generated on run)
```

---

## Dataset Overview

| Column | Type | Description |
|---|---|---|
| id | int | Unique identifier |
| name | str | Individual's name |
| age | int | Age in years |
| gender | str | Gender (Male / Female / Other) |
| country | str | Country of residence |
| city | str | City of residence |
| education_level | str | Highest education attained |
| employment_status | str | Employment type |
| annual_income_usd | float | Annual income in USD |
| marital_status | str | Marital status |
| children_count | int | Number of children |
| smokes_per_day | int | Cigarettes per day |
| drinks_per_week | int | Alcoholic drinks per week |
| age_started_smoking | int | Age when smoking began |
| age_started_drinking | int | Age when drinking began |
| attempts_to_quit_smoking | int | Number of quit attempts (smoking) |
| attempts_to_quit_drinking | int | Number of quit attempts (drinking) |
| **has_health_issues** | bool | **Classification target** |
| mental_health_status | str | Self-reported mental health |
| exercise_frequency | str | How often the person exercises |
| diet_quality | str | Quality of diet |
| sleep_hours | float | Average nightly sleep |
| bmi | float | Body Mass Index |
| social_support | str | Level of social support |
| therapy_history | str | Therapy engagement history |

**Total rows:** 3 000 · **Total columns:** 25

---

## Notebook Pipeline

### 1. Imports & Configuration
Sets up all libraries and global constants.

### 2. Data Loading & Overview
Loads the CSV, prints shape, data types, and missing-value summary.

### 3. Exploratory Data Analysis (EDA)
- Target distributions (classification & regression)
- Numerical feature histograms
- Correlation heatmap
- Categorical feature count plots
- Health-issues breakdown by group

### 4. Pre-processing & Feature Engineering
- Drops non-predictive columns (`id`, `name`, `city`, `country`)
- Engineers four new features:
  - `smoking_duration` = age − age_started_smoking
  - `drinking_duration` = age − age_started_drinking
  - `total_substance` = smokes_per_day + drinks_per_week
  - `quit_attempts_total` = sum of both quit-attempt columns
- `ColumnTransformer` with median imputation + standard scaling (numerical) and most-frequent imputation + one-hot encoding (categorical)
- 80 / 20 stratified train-test split

### 5. Classification — `has_health_issues`
Five models benchmarked with 5-fold stratified cross-validation:

| Model | Metric |
|---|---|
| Logistic Regression | Accuracy, AUC |
| Random Forest | Accuracy, AUC |
| Gradient Boosting | Accuracy, AUC |
| SVM (RBF kernel) | Accuracy, AUC |
| K-Nearest Neighbours | Accuracy, AUC |

Outputs: comparison bar chart, confusion matrix, ROC curves.

### 6. Regression — `smokes_per_day`
Four models benchmarked:

| Model | Metric |
|---|---|
| Linear Regression | MAE, RMSE, R² |
| Ridge Regression | MAE, RMSE, R² |
| Random Forest Regressor | MAE, RMSE, R² |
| Gradient Boosting Regressor | MAE, RMSE, R² |

Outputs: error bar charts, actual-vs-predicted scatter.

### 7. Feature Importance
Top-20 feature importances from the Random Forest models for both tasks.

### 8. Model Persistence
Best pipelines saved via `joblib` as `.pkl` files.

### 9. Quick Inference
Demonstrates loading a saved pipeline and running predictions on a new record.

---

## Quickstart

### Prerequisites
- Python 3.10+

### Installation

```bash
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook addiction_model.ipynb
```

Run all cells from top to bottom (`Kernel → Restart & Run All`).

---

## Output Files (generated after running)

| File | Description |
|---|---|
| `best_classifier.pkl` | Serialised sklearn Pipeline for health-issue classification |
| `best_regressor.pkl` | Serialised sklearn Pipeline for smoking-intensity regression |

---

## Dependencies

| Package | Min Version |
|---|---|
| numpy | 1.24.0 |
| pandas | 2.0.0 |
| matplotlib | 3.7.0 |
| seaborn | 0.12.0 |
| scikit-learn | 1.3.0 |
| joblib | 1.3.0 |
| jupyter / notebook | 1.0.0 / 7.0.0 |

---

## License
This project is intended for educational and research purposes.
