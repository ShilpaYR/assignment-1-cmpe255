
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
House Prices (Kaggle) — End-to-End CRISP‑DM Pipeline
====================================================
A single, runnable script that:
  1) Loads raw Kaggle files (train.csv, test.csv, data_description.txt if present)
  2) Produces lightweight EDA artifacts
  3) Cleans & engineers features (domain-aware imputations + ordinals)
  4) Detects & filters outliers (influence-based on log target)
  5) Builds modeling matrices with proper encoders
  6) Trains & tunes multiple models with CV (Linear, Ridge, Lasso, ElasticNet, RandomForest,
     and optionally XGBoost/LightGBM if installed)
  7) Evaluates with RMSE/MAE/R2 (CV + holdout), selects the best
  8) Fits on full train, scores test, and writes a Kaggle submission
  9) Saves a deployable .joblib pipeline + scoring CLI

**Concise explanation is integrated throughout the code via comments/docstrings.**
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Headless plots saved to files
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, KFold, GridSearchCV, RandomizedSearchCV, cross_validate, learning_curve
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor

# Optional GBMs — seamlessly skipped if not installed
try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    from lightgbm import LGBMRegressor
    LGBM_OK = True
except Exception:
    LGBM_OK = False

RNG_SEED = 42
np.random.seed(RNG_SEED)
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------- Utilities -----------------------------

def safe_expm1(z: np.ndarray) -> np.ndarray:
    """Stable inverse for log1p to protect against extreme predictions."""
    return np.expm1(np.clip(z, -50, 20))

def rmse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred, squared=False))

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def find_file(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None

# ----------------------------- Data Loading -----------------------------

def locate_kaggle_files(data_dir: Path) -> Tuple[Path, Path, Optional[Path]]:
    """Find train/test (and optional data_description) in common locations."""
    train_candidates = [
        data_dir / "train.csv", Path("train.csv"), Path.cwd()/ "train.csv"
    ]
    test_candidates = [
        data_dir / "test.csv", Path("test.csv"), Path.cwd() / "test.csv"
    ]
    desc_candidates = [
        data_dir / "data_description.txt", Path("data_description.txt")
    ]
    train_path = find_file(train_candidates)
    test_path = find_file(test_candidates)
    desc_path = find_file(desc_candidates)
    if not train_path or not test_path:
        raise FileNotFoundError(
            "Could not find train.csv and/or test.csv. Place them in the data directory."
        )
    return train_path, test_path, desc_path

# ----------------------------- EDA (light) -----------------------------

def eda_snapshot(train: pd.DataFrame, outdir: Path) -> None:
    """Save a compact EDA snapshot (tables + a few plots) without heavy runtime."""
    ensure_dir(outdir)
    meta = {
        "n_rows": int(train.shape[0]),
        "n_cols": int(train.shape[1]),
        "n_numeric": int(train.select_dtypes(include=[np.number]).shape[1]),
        "n_categorical": int(train.select_dtypes(exclude=[np.number]).shape[1]),
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Missingness table
    miss = train.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0].to_frame("missing_count")
    miss["missing_pct"] = miss["missing_count"] / len(train) * 100.0
    miss.to_csv(outdir / "missingness.csv")

    # Target distributions (linear + log1p)
    if "SalePrice" in train.columns:
        y = train["SalePrice"].dropna().values
        plt.figure(figsize=(7,4))
        plt.hist(y, bins=50)
        plt.title("SalePrice — distribution")
        plt.xlabel("SalePrice"); plt.ylabel("Count"); plt.tight_layout()
        plt.savefig(outdir / "saleprice_hist.png"); plt.close()

        plt.figure(figsize=(7,4))
        plt.hist(np.log1p(y), bins=50)
        plt.title("log1p(SalePrice) — distribution")
        plt.xlabel("log1p(SalePrice)"); plt.ylabel("Count"); plt.tight_layout()
        plt.savefig(outdir / "saleprice_log_hist.png"); plt.close()

        # Quick scatter vs. GrLivArea
        if "GrLivArea" in train.columns:
            plt.figure(figsize=(7,4))
            plt.scatter(train["GrLivArea"], train["SalePrice"], s=10, alpha=0.6)
            plt.title("GrLivArea vs SalePrice")
            plt.xlabel("GrLivArea"); plt.ylabel("SalePrice"); plt.tight_layout()
            plt.savefig(outdir / "grlivarea_vs_price.png"); plt.close()

# ----------------------------- Cleaning & Feature Engineering -----------------------------

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-aware imputations and engineered features.
    - Respect 'NA = none' semantics for many basement/garage/fireplace/pool fields.
    - Median/mode fills otherwise; LotFrontage by Neighborhood median; GarageYrBlt=0 if missing.
    - Create engineered totals, ages, and simple interaction terms.
    - Add ordinal encodings as *_ord columns for quality/condition scales.
    """
    out = df.copy()

    none_cats = [
        "Alley","PoolQC","Fence","MiscFeature","FireplaceQu",
        "GarageType","GarageFinish","GarageQual","GarageCond",
        "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
        "MasVnrType"
    ]
    zero_nums = [
        "MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF",
        "BsmtFullBath","BsmtHalfBath","GarageCars","GarageArea"
    ]
    mode_fill_cats = ["MSZoning","Exterior1st","Exterior2nd","KitchenQual","Functional","SaleType","Utilities","Electrical"]

    for c in none_cats:
        if c in out.columns:
            out[c] = out[c].fillna("None")
    for c in zero_nums:
        if c in out.columns:
            out[c] = out[c].fillna(0)

    if "Neighborhood" in out.columns and "LotFrontage" in out.columns:
        out["LotFrontage"] = out.groupby("Neighborhood")["LotFrontage"].transform(
            lambda s: s.fillna(s.median())
        )
    if "GarageYrBlt" in out.columns:
        out["GarageYrBlt"] = out["GarageYrBlt"].fillna(0)

    # Remaining fills
    num_cols = out.select_dtypes(include=[np.number]).columns
    cat_cols = out.select_dtypes(exclude=[np.number]).columns

    for c in num_cols:
        if out[c].isna().any():
            out[c] = out[c].fillna(out[c].median())
    for c in mode_fill_cats:
        if c in out.columns and out[c].isna().any():
            mode_val = out[c].mode(dropna=True)
            out[c] = out[c].fillna(mode_val.iloc[0] if len(mode_val) else "Unknown")
    for c in out.columns:
        if c in cat_cols and out[c].isna().any():
            out[c] = out[c].fillna("Unknown")

    # Ordinals
    def ord_map(series, order, none_label="None"):
        order = [none_label] + order if none_label not in order else order
        cat = pd.Categorical(series.fillna(none_label).astype("object"), categories=order, ordered=True)
        return cat.codes.astype("int16")

    qual_order = ["Po","Fa","TA","Gd","Ex"]
    exposure_order = ["No","Mn","Av","Gd"]
    finish_order = ["Unf","RFn","Fin"]
    bsmtfin_order = ["Unf","LwQ","Rec","BLQ","ALQ","GLQ"]
    paved_order = ["N","P","Y"]
    lotslope_order = ["Sev","Mod","Gtl"]
    lotshape_order = ["IR3","IR2","IR1","Reg"]
    landcontour_order = ["Low","HLS","Bnk","Lvl"]
    utilities_order = ["ELO","NoSeWa","NoSewr","AllPub"]
    functional_order = ["Sal","Sev","Maj2","Maj1","Mod","Min2","Min1","Typ"]

    ordinal_specs = {
        "ExterQual": qual_order, "ExterCond": qual_order, "BsmtQual": qual_order,
        "BsmtCond": qual_order, "HeatingQC": qual_order, "KitchenQual": qual_order,
        "FireplaceQu": qual_order, "GarageQual": qual_order, "GarageCond": qual_order,
        "PoolQC": qual_order, "BsmtExposure": exposure_order, "GarageFinish": finish_order,
        "BsmtFinType1": bsmtfin_order, "BsmtFinType2": bsmtfin_order, "PavedDrive": paved_order,
        "LandSlope": lotslope_order, "LotShape": lotshape_order, "LandContour": landcontour_order,
        "Utilities": utilities_order, "Functional": functional_order
    }
    for col, order in ordinal_specs.items():
        if col in out.columns:
            out[col + "_ord"] = ord_map(out[col], order)

    # Engineered features
    if {"TotalBsmtSF","1stFlrSF","2ndFlrSF"}.issubset(out.columns):
        out["TotalSF"] = out["TotalBsmtSF"] + out["1stFlrSF"] + out["2ndFlrSF"]

    for b in ["FullBath","HalfBath","BsmtFullBath","BsmtHalfBath"]:
        if b not in out.columns:
            out[b] = 0
    out["TotalBath"] = out["FullBath"] + 0.5*out["HalfBath"] + out["BsmtFullBath"] + 0.5*out["BsmtHalfBath"]

    for p in ["OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","WoodDeckSF"]:
        if p not in out.columns:
            out[p] = 0
    out["TotalPorchSF"] = out["OpenPorchSF"] + out["EnclosedPorch"] + out["3SsnPorch"] + out["ScreenPorch"] + out["WoodDeckSF"]

    for y in ["YearBuilt","YearRemodAdd","GarageYrBlt"]:
        if y not in out.columns:
            out[y] = 0
    if "YrSold" in out.columns:
        out["AgeHouse"] = out["YrSold"] - out["YearBuilt"]
        out["AgeSinceRemodel"] = out["YrSold"] - out["YearRemodAdd"]
        gyb = out["GarageYrBlt"].replace(0, out["YearBuilt"])
        out["AgeGarage"] = out["YrSold"] - gyb

    if {"OverallQual","GrLivArea"}.issubset(out.columns):
        out["Qual_x_GrLiv"] = out["OverallQual"] * out["GrLivArea"]

    out["HasPool"] = (out.get("PoolArea", 0) > 0).astype(int)
    out["Has2ndFlr"] = (out.get("2ndFlrSF", 0) > 0).astype(int)
    out["HasBsmt"] = (out.get("TotalBsmtSF", 0) > 0).astype(int)
    out["HasFireplace"] = (out.get("Fireplaces", 0) > 0).astype(int)

    return out

# ----------------------------- Outlier Detection -----------------------------

def influence_outliers(train_clean: pd.DataFrame, y_name="SalePrice") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect outliers via linear OLS on log target with compact predictors.
    Returns: (report_df, train_no_outliers)
    """
    # Compact numeric set
    feats = [c for c in [
        "GrLivArea","TotalSF","OverallQual","GarageCars","TotalBath",
        "AgeHouse","AgeSinceRemodel","AgeGarage","HasBsmt","HasFireplace","Has2ndFlr"
    ] if c in train_clean.columns]
    X = train_clean[feats].astype(float).copy()
    y = np.log1p(train_clean[y_name].astype(float).values)

    # Standardize + add intercept
    Xz = (X - X.mean()) / X.std(ddof=1).replace(0, 1.0)
    Xmat = np.column_stack([np.ones(len(Xz)), Xz.values])
    yv = y

    # Closed-form OLS
    XtX = Xmat.T @ Xmat
    beta = np.linalg.inv(XtX) @ (Xmat.T @ yv)
    yhat = Xmat @ beta
    resid = yv - yhat
    n, p = Xmat.shape
    H = Xmat @ (np.linalg.inv(XtX)) @ Xmat.T
    h = np.clip(np.diag(H), 0, 1)
    mse = (resid @ resid) / max(1, (n - p))
    std_resid = resid / np.sqrt(np.maximum(1e-12, mse * (1 - h)))
    cooks = (resid**2 / (p * np.maximum(1e-12, mse))) * (h / np.maximum(1e-12, (1 - h)**2))

    thr_std = 3.0
    thr_cook = 4.0 / max(n, 1)
    is_out = (np.abs(std_resid) > thr_std) | (cooks > thr_cook)

    report = pd.DataFrame({
        "Id": train_clean["Id"].values if "Id" in train_clean.columns else np.arange(len(train_clean)),
        "SalePrice": train_clean[y_name].values,
        "std_resid": std_resid, "hat": h, "cooks_d": cooks, "is_outlier": is_out
    }).sort_values("cooks_d", ascending=False)

    train_no_out = train_clean.loc[~is_out].copy()
    return report, train_no_out

# ----------------------------- Modeling -----------------------------

@dataclass
class ModelSpec:
    name: str
    estimator: object
    preprocessor: ColumnTransformer
    grid: Optional[dict]
    search: str = "grid"
    n_iter: int = 10

def build_preprocessors(X: pd.DataFrame) -> Tuple[ColumnTransformer, ColumnTransformer, List[str], List[str]]:
    """Create preprocessors for linear vs tree models."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    pre_lin = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), obj_cols)
    ], remainder="drop")
    pre_tree = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), obj_cols)
    ], remainder="drop")
    return pre_lin, pre_tree, num_cols, obj_cols

def model_specs(X: pd.DataFrame) -> List[ModelSpec]:
    """Define candidate models and modest parameter grids to keep runtime reasonable."""
    pre_lin, pre_tree, *_ = build_preprocessors(X)
    specs: List[ModelSpec] = [
        ModelSpec("Linear", LinearRegression(), pre_lin, grid=None, search="none"),
        ModelSpec("Ridge", Ridge(random_state=RNG_SEED), pre_lin, grid={"regressor__reg__alpha": [0.1, 1.0, 10.0, 50.0]}),
        ModelSpec("Lasso", Lasso(random_state=RNG_SEED, max_iter=5000), pre_lin, grid={"regressor__reg__alpha": [0.001, 0.01, 0.1]}),
        ModelSpec("ElasticNet", ElasticNet(random_state=RNG_SEED, max_iter=5000), pre_lin, grid={"regressor__reg__alpha":[0.01,0.1], "regressor__reg__l1_ratio":[0.4,0.8]}),
        ModelSpec("RandomForest", RandomForestRegressor(n_estimators=300, max_depth=20, max_features="sqrt", random_state=RNG_SEED, n_jobs=-1), pre_tree, grid=None, search="none"),
    ]
    if XGB_OK:
        specs.append(ModelSpec("XGBoost",
            XGBRegressor(tree_method="hist", random_state=RNG_SEED, n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, n_jobs=-1),
            pre_tree, grid=None, search="none"))
    if LGBM_OK:
        specs.append(ModelSpec("LightGBM",
            LGBMRegressor(random_state=RNG_SEED, n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.9, colsample_bytree=0.9),
            pre_tree, grid=None, search="none"))
    return specs

def fit_and_compare(X: pd.DataFrame, y: pd.Series, artifacts: Path, test_size: float = 0.2) -> Tuple[pd.DataFrame, str, object]:
    """
    Train/validation split; CV inside training; metrics on validation.
    Returns (results_table, best_name, best_fitted_estimator)
    """
    ensure_dir(artifacts)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=test_size, random_state=RNG_SEED)

    rmse_scorer = make_scorer(lambda yt, yp: mean_squared_error(yt, yp, squared=False), greater_is_better=False)
    mae_scorer  = make_scorer(mean_absolute_error, greater_is_better=False)
    r2_scorer   = make_scorer(r2_score)

    cv = KFold(n_splits=3, shuffle=True, random_state=RNG_SEED)

    rows = []
    best_name = None
    best_rmse = math.inf
    best_model = None

    for spec in model_specs(X):
        pipe = Pipeline([("pre", spec.preprocessor), ("reg", spec.estimator)])
        model = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=safe_expm1)

        if spec.search == "none" or not spec.grid:
            cv_res = cross_validate(model, X_tr, y_tr, cv=cv,
                                    scoring={"rmse": rmse_scorer, "mae": mae_scorer, "r2": r2_scorer},
                                    n_jobs=-1, return_estimator=False)
            fitted = model.fit(X_tr, y_tr)
            params = {}
            cv_rmse_mean = -np.mean(cv_res["test_rmse"])
            cv_rmse_std  = np.std(-cv_res["test_rmse"])
        else:
            # You can switch to RandomizedSearchCV by setting spec.search to "random"
            SearchCV = GridSearchCV if spec.search == "grid" else RandomizedSearchCV
            search = SearchCV(
                model,
                param_grid=spec.grid if spec.search == "grid" else spec.grid,
                n_iter=spec.n_iter if spec.search == "random" else None,
                scoring={"rmse": rmse_scorer, "mae": mae_scorer, "r2": r2_scorer},
                refit="rmse", cv=cv, n_jobs=-1, verbose=0, random_state=RNG_SEED if spec.search == "random" else None
            )
            search.fit(X_tr, y_tr)
            fitted = search.best_estimator_
            params = search.best_params_
            cv_rmse_mean = -search.best_score_
            cv_rmse_std  = float("nan")

        # Validation metrics
        y_pred = fitted.predict(X_va)
        hold_rmse = rmse(y_va, y_pred)
        hold_mae  = mean_absolute_error(y_va, y_pred)
        hold_r2   = r2_score(y_va, y_pred)

        rows.append({
            "model": spec.name,
            "cv_rmse_mean": cv_rmse_mean, "cv_rmse_std": cv_rmse_std,
            "holdout_RMSE": hold_rmse, "holdout_MAE": hold_mae, "holdout_R2": hold_r2,
            "params": params
        })

        if hold_rmse < best_rmse:
            best_rmse = hold_rmse
            best_name = spec.name
            best_model = fitted

    results_df = pd.DataFrame(rows).sort_values("holdout_RMSE").reset_index(drop=True)
    results_df.to_csv(artifacts / "model_comparison.csv", index=False)
    return results_df, best_name, best_model

# ----------------------------- Deployment Helpers -----------------------------

def save_learning_curve(model, X, y, outpath: Path) -> None:
    """Learning curve (RMSE) to visualize bias/variance behavior."""
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X, y, cv=3, scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1
    )
    train_rmse = np.sqrt(-train_scores.mean(axis=1))
    valid_rmse = np.sqrt(-valid_scores.mean(axis=1))
    plt.figure(figsize=(7,4))
    plt.plot(train_sizes, train_rmse, marker="o", label="Train RMSE")
    plt.plot(train_sizes, valid_rmse, marker="o", label="CV RMSE")
    plt.title("Learning Curve (best model)")
    plt.xlabel("Training examples"); plt.ylabel("RMSE"); plt.legend(); plt.tight_layout()
    plt.savefig(outpath); plt.close()

def save_feature_influence(best_model, X: pd.DataFrame, outpath: Path) -> None:
    """Top-20 coefficients/importances for the best model."""
    reg_pipe = best_model.regressor_
    pre = reg_pipe.named_steps["pre"]
    reg = reg_pipe.named_steps["reg"]

    # Infer feature names
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    try:
        num_names = pre.named_transformers_["num"].get_feature_names_out()
    except Exception:
        num_names = np.array(num_cols)
    if hasattr(pre.named_transformers_["cat"], "get_feature_names_out"):
        cat_names = pre.named_transformers_["cat"].get_feature_names_out(obj_cols)
    else:
        cat_names = np.array(obj_cols)
    feat_names = np.concatenate([num_names, cat_names])

    if hasattr(reg, "coef_"):
        vals = np.abs(np.ravel(reg.coef_))
    elif hasattr(reg, "feature_importances_"):
        vals = reg.feature_importances_
    else:
        return

    k = min(20, len(vals))
    idx = np.argsort(vals)[-k:]
    top_names = feat_names[idx]
    top_vals  = vals[idx]

    plt.figure(figsize=(8,5))
    plt.barh(top_names, top_vals)
    plt.title("Top Feature Influences (best model)")
    plt.xlabel("Abs(coef) / importance")
    plt.tight_layout()
    plt.savefig(outpath); plt.close()

# ----------------------------- Main Entry -----------------------------

def main():
    ap = argparse.ArgumentParser(description="End-to-end CRISP-DM pipeline for Kaggle House Prices.")
    ap.add_argument("--data_dir", type=str, default=".", help="Directory containing train.csv/test.csv")
    ap.add_argument("--artifacts_dir", type=str, default="./artifacts", help="Where to save outputs")
    ap.add_argument("--remove_outliers", action="store_true", help="If set, remove influence-flagged outliers before modeling")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    artifacts = Path(args.artifacts_dir)
    ensure_dir(artifacts)

    # 1) Load
    train_path, test_path, desc_path = locate_kaggle_files(data_dir)
    train_raw = pd.read_csv(train_path)
    test_raw  = pd.read_csv(test_path)

    # 2) EDA (lightweight)
    eda_dir = artifacts / "eda"
    eda_snapshot(train_raw, eda_dir)

    # 3) Cleaning + feature engineering
    train_clean = clean_and_engineer(train_raw)
    test_clean  = clean_and_engineer(test_raw)
    train_clean.to_csv(artifacts / "clean_train.csv", index=False)
    test_clean.to_csv(artifacts / "clean_test.csv", index=False)

    # 4) Outliers (optional removal)
    out_report, train_no_out = influence_outliers(train_clean, y_name="SalePrice")
    out_report.to_csv(artifacts / "outlier_report.csv", index=False)
    train_used = train_no_out if args.remove_outliers else train_clean

    # 5) Modeling matrices
    X = train_used.drop(columns=["SalePrice"])
    y = train_used["SalePrice"].astype(float)

    # 6) Fit & compare candidate models with CV; evaluate on a holdout split
    results_df, best_name, best_model = fit_and_compare(X, y, artifacts)
    print("\n=== Model Comparison (sorted by holdout RMSE) ===")
    print(results_df.to_string(index=False))
    print(f"\nSelected best model: {best_name}")

    results_df.to_csv(artifacts / "model_comparison.csv", index=False)

    # 7) Learning curve + feature influences for best model (saved)
    save_learning_curve(best_model, X, y, artifacts / "best_learning_curve.png")
    save_feature_influence(best_model, X, artifacts / "best_feature_influence.png")

    # 8) Train on FULL training set and score Kaggle test; write submission
    #    (The model already includes preprocessing inside the pipeline.)
    best_model.fit(train_clean.drop(columns=["SalePrice"]), train_clean["SalePrice"].astype(float))
    preds = best_model.predict(test_clean)

    # If sample_submission.csv exists, use its order; else infer from test Id
    sample_path = data_dir / "sample_submission.csv"
    if sample_path.exists():
        sub = pd.read_csv(sample_path)
        if "Id" in test_clean.columns:
            pred_df = pd.DataFrame({"Id": test_clean["Id"], "SalePrice": preds})
            sub = sub[["Id","SalePrice"]].merge(pred_df, on="Id", how="left", suffixes=("_template",""))
            sub["SalePrice"] = sub["SalePrice"].fillna(sub["SalePrice_template"])
            sub = sub.drop(columns=["SalePrice_template"])
        else:
            sub["SalePrice"] = preds[:len(sub)]
    else:
        if "Id" in test_clean.columns:
            sub = pd.DataFrame({"Id": test_clean["Id"], "SalePrice": preds})
        else:
            sub = pd.DataFrame({"SalePrice": preds})

    sub_path = artifacts / "submission.csv"
    sub.to_csv(sub_path, index=False)

    # 9) Persist deployable model
    import joblib
    joblib.dump(best_model, artifacts / "best_model.joblib")

    # 10) Write a tiny scoring CLI
    scorer = f"""#!/usr/bin/env python3
import argparse, pandas as pd, joblib
from pathlib import Path
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='best_model.joblib')
    ap.add_argument('--input', required=True, help='CSV to score (expects cleaned schema)')
    ap.add_argument('--output', default='scored.csv')
    args = ap.parse_args()
    model = joblib.load(args.model)
    X = pd.read_csv(args.input)
    preds = model.predict(X)
    if 'Id' in X.columns:
        out = pd.DataFrame({{'Id': X['Id'], 'SalePrice': preds}})
    else:
        out = pd.DataFrame({{'SalePrice': preds}})
    out.to_csv(args.output, index=False)
    print(f'Wrote: {{args.output}}')
if __name__ == '__main__':
    main()
"""
    (artifacts / "score.py").write_text(scorer)

    print(f"\nArtifacts saved to: {artifacts.resolve()}")
    print(" - Cleaned train/test CSVs")
    print(" - EDA figures & tables")
    print(" - Outlier report")
    print(" - Model comparison CSV + plots")
    print(" - best_model.joblib")
    print(" - submission.csv")
    print(" - score.py (CLI)")

if __name__ == "__main__":
    main()
