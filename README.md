# 🏠 Kaggle House Prices — End-to-End ML Project

> A clean, reproducible, portfolio-ready workflow for the **House Prices: Advanced Regression Techniques** competition (Ames, Iowa).  
> This README doubles as documentation: dataset overview, project workflow, course snippets, results, insights, and a link to a Medium write-up.

---

## Badges (optional)

<!-- Replace with real badges or delete this block -->
![status](https://img.shields.io/badge/status-active-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Table of Contents

- [Project Overview (TL;DR)](#project-overview-tldr)
- [Dataset](#dataset)
- [Project Workflow (CRISP-DM)](#project-workflow-crisp-dm)
- [Repository Structure](#repository-structure)
- [Quickstart](#quickstart)
- [Experiments & Modeling](#experiments--modeling)
- [Results](#results)
- [Key Insights](#key-insights)
- [Selected Course Snippets](#selected-course-snippets)
- [Extended Documentation (Medium)](#extended-documentation-medium)
- [Reproducibility Notes](#reproducibility-notes)
- [Limitations & Considerations](#limitations--considerations)
- [Acknowledgments & Citation](#acknowledgments--citation)
- [License](#license)
- [Contact](#contact)

---

## Project Overview (TL;DR)

- **Goal:** Predict `SalePrice` for Ames, Iowa homes using structural and neighborhood features.  
- **Type:** Supervised Regression (tabular).  
- **Primary Metric:** RMSE (on original price; log1p used internally during training).  
- **What’s included:**  
  - End-to-end pipeline: cleaning → feature engineering → outlier handling → modeling → evaluation → submission.  
  - Reproducible scripts and assets.  
  - Clear visuals and professional conclusions.

---

## Dataset

**Source:** Kaggle competition *House Prices: Advanced Regression Techniques*.  
**Files (typical):**
- `train.csv` — labeled training data (includes `SalePrice`)  
- `test.csv` — unlabeled test data (predict `SalePrice`)  
- `data_description.txt` — feature dictionary (levels/semantics)  
- `sample_submission.csv` — submission template

**Target:** `SalePrice` (continuous).  
**Common feature families:**
- **Size/Area:** `GrLivArea`, `TotalBsmtSF`, `LotArea`, floor areas, porches/decks  
- **Quality/Condition:** `OverallQual`, `OverallCond`, `ExterQual`, `KitchenQual`, `HeatingQC`  
- **Time:** `YearBuilt`, `YearRemodAdd`, `GarageYrBlt`, `YrSold`  
- **Location:** `Neighborhood`, `MSSubClass`, `MSZoning`  
- **Garage/Basement:** presence/quality/area — **NAs sometimes mean “None”** (crucial!)

> 📌 **Note:** In this project, many *NAs mean “feature absent”* (e.g., `BsmtQual`, `FireplaceQu`, `PoolQC`), not “missing at random”. This is central to correct preprocessing.

---

## Project Workflow (CRISP-DM)

1) **Business Understanding**  
   - Estimate fair prices; identify drivers; ensure robust generalization.

2) **Data Understanding**  
   - Explore distributions, missingness, right-skew in area/price; numeric vs categorical balance; target log transform rationale.

3) **Data Preparation**  
   - Domain-aware imputations (e.g., *None* vs *Unknown*), ordinal encodings (quality scales), engineered features (`TotalSF`, `TotalBath`, `AgeSinceRemodel`, porch sums), and alignment.

4) **Modeling**  
   - Baselines: Linear/Ridge/Lasso/ElasticNet.  
   - Tree-based: RandomForest (+ optional XGBoost/LightGBM).  
   - `TransformedTargetRegressor(func=log1p)` for stability.

5) **Evaluation**  
   - K-fold CV, holdout metrics (RMSE/MAE/R²), learning curves, error slices (by neighborhood, by price quartiles), feature influence.

6) **Deployment**  
   - Save pipeline (`.joblib`), submission CSV, and a tiny CLI for scoring new data.  
   - Optional: one-click script to regenerate all artifacts.

---

## Repository Structure

```text
.
├── README.md
├── house_prices_end_to_end.py        # One-file script: full pipeline & submission
├── assets/                           # Images & charts for the README
│   ├── eda/
│   │   ├── saleprice_hist.png
│   │   ├── saleprice_log_hist.png
│   │   └── grlivarea_vs_price.png
│   └── results/
│       ├── model_comparison.png
│       ├── learning_curve.png
│       ├── feature_influence.png
│       └── error_by_quartile.png
├── data/                             # Place Kaggle files here (not versioned)
│   ├── train.csv
│   ├── test.csv
│   ├── data_description.txt
│   └── sample_submission.csv
└── artifacts/                        # Auto-generated outputs (gitignored)
    ├── clean_train.csv
    ├── clean_test.csv
    ├── outlier_report.csv
    ├── model_comparison.csv
    ├── best_model.joblib
    └── submission.csv
```

> ⚠️ Add `data/` and `artifacts/` to `.gitignore`.

---

## Quickstart

### 1) Environment
```bash
# conda or mamba recommended
conda create -n house-prices python=3.10 -y
conda activate house-prices
pip install -r requirements.txt
# If you want GBMs:
# pip install xgboost lightgbm
```

Minimal `requirements.txt`:
```
pandas
numpy
scikit-learn>=1.3
matplotlib
joblib
```

### 2) Data
Download the competition files and drop them in `./data/`:
```
data/
  train.csv
  test.csv
  data_description.txt
  sample_submission.csv
```

### 3) Run the end-to-end pipeline
```bash
python house_prices_end_to_end.py   --data_dir ./data   --artifacts_dir ./artifacts   --remove_outliers
```

Outputs:
- Cleaned datasets, outlier report, model comparison CSV
- Saved model `best_model.joblib`
- `submission.csv` ready for Kaggle

### 4) Score any cleaned file (CLI)
```bash
python artifacts/score.py --model artifacts/best_model.joblib   --input artifacts/clean_test.csv   --output artifacts/scored.csv
```

---

## Experiments & Modeling

**Validation strategy**
- **Holdout**: 80/20 split (random state fixed).  
- **Cross-validation**: typically 3–5 folds for quick iteration and robustness.  

**Models**
- Linear family: **Linear**, **Ridge**, **Lasso**, **ElasticNet**  
- Trees: **RandomForest** (fast), **XGBoost/LightGBM** (optional; often top performers with tuning).  

**Preprocessing**
- **Numeric**: standardized for linear models; passthrough for trees.  
- **Categorical**: One-Hot (`handle_unknown='ignore'` to safely handle unseen labels).  
- **Target**: `log1p(SalePrice)` via `TransformedTargetRegressor` for stability/normality.

**Hyper-parameter tuning**
- Lightweight **GridSearchCV**/**RandomizedSearchCV** on core knobs (e.g., `alpha` for Ridge/Lasso; `depth/leaves` for trees/GBMs).

---

## Results

> Replace placeholders with your actual numbers/paths.

### Metrics (Validation)
| Model         | CV RMSE (mean±std) | Holdout RMSE | Holdout MAE | Holdout R² |
|---------------|--------------------|--------------|-------------|------------|
| Ridge         |  **$XX,XXX ± X**   | **$XX,XXX**  | $X,XXX      | 0.90+      |
| Lasso         |  $XX,XXX ± X       | $XX,XXX      | $X,XXX      | 0.90+      |
| ElasticNet    |  $XX,XXX ± X       | $XX,XXX      | $X,XXX      | 0.89+      |
| RandomForest  |  $XX,XXX ± X       | $XX,XXX      | $X,XXX      | 0.88+      |
| XGBoost*      |  $XX,XXX ± X       | $XX,XXX      | $X,XXX      | 0.91+      |
| LightGBM*     |  $XX,XXX ± X       | $XX,XXX      | $X,XXX      | 0.91+      |

*\* if enabled*

**Charts (embed from `assets/`):**

- **Model comparison**  
  <img src="assets/results/model_comparison.png" width="520" />

- **Learning curve (best model)**  
  <img src="assets/results/learning_curve.png" width="520" />

- **Feature influence (coefficients/importances)**  
  <img src="assets/results/feature_influence.png" width="520" />

- **Error by price quartile**  
  <img src="assets/results/error_by_quartile.png" width="520" />

- **EDA samples**  
  <img src="assets/eda/saleprice_hist.png" width="360" />  
  <img src="assets/eda/grlivarea_vs_price.png" width="360" />

> 📎 Save your plots to these paths or update the image sources above.

**Leaderboard (optional)**  
- Public/Private scores if you submitted to Kaggle:  
  - Public LB: **—**  
  - Private LB: **—**

---

## Key Insights

- **Size × Quality dominate:** `GrLivArea`, `TotalSF`, and **`OverallQual`** consistently rank at the top; their interaction is powerful.  
- **Location matters:** `Neighborhood` signals are strong; target/mean encoding (with CV!) can help.  
- **Ages & renovations:** `YearBuilt`, `YearRemodAdd`, and `AgeSinceRemodel` carry meaningful value effects.  
- **Right-skew handling is essential:** log transform of `SalePrice` stabilizes training.  
- **NAs ≠ missing at random:** many NAs mean **absence** (e.g., no basement/garage/pool) and should be encoded explicitly.

---

## Selected Course Snippets

> Curate brief, high-signal excerpts from your coursework (or blog notes). Keep each to ~3–6 lines and link to the source if public.

- **Bias–Variance & Learning Curves**
  ```text
  Learning curves flattened with small train–CV gap ⇒ model underfits mildly; regularization is appropriate;
  adding data helps a bit; feature engineering / non-linear models may yield larger gains.
  ```

- **Encoding Strategy**
  ```text
  One-Hot for nominal features with handle_unknown='ignore'.
  Ordinal codes for quality scales (Po < Fa < TA < Gd < Ex).
  ```

- **Validation Hygiene**
  ```text
  CV splits inside training only; no peeking at the holdout.
  Hyper-params selected on CV; final metrics reported on untouched validation.
  ```

*(Replace/add with your own notes, links, or short code blocks.)*

---

## Extended Documentation (Medium)

For a long-form narrative (technical deep-dive, design decisions, and lessons learned), see:  
**Medium article:** _Predicting Home Prices_ — **[Medium]([https://medium.com/](https://medium.com/@shilpa.yelkurramakrishnaiah/predicting-home-prices-411df8c9b543))**

---

## Reproducibility Notes

- **Random seeds** fixed (`42`) for splits and models where applicable.  
- **Environment** pinned in `requirements.txt`.  
- **Data version**: specify the competition version/date you downloaded.  
- **Command log**: include exact CLI calls used to generate artifacts.

---

## Limitations & Considerations

- Price prediction is sensitive to **market regime shifts** and **data leakage** (beware target leakage from time/location encodings).  
- **Ethical use:** Do not over-interpret model outputs as fair market value without proper appraisal context and domain review.  
- **Generalization:** Ames features may not transfer cleanly to other locales.

---

## Acknowledgments & Citation

- **Kaggle** & competition hosts for the dataset.  
- **Ames Housing** dataset contributors and documentation authors.  
- Libraries: **pandas**, **NumPy**, **scikit-learn**, (optionally **XGBoost**/**LightGBM**).

---

## License

This project is released under the **MIT License**.

---

## Contact

- **Author:** _Shilpa Yelkur Ramakrishnaiah_  
- **Email:** _shilpa.yelkurramakrishnaiah@sjsu.edu_

---

### Appendix: How to Regenerate Everything

```bash
# From project root
conda activate house-prices  # or your env
python house_prices_end_to_end.py   --data_dir ./data   --artifacts_dir ./artifacts   --remove_outliers
```

This will:
- produce cleaned data,  
- fit/tune multiple models with CV,  
- export metrics & plots under `artifacts/`,  
- save `best_model.joblib`, and  
- create `submission.csv` ready for Kaggle.
