"""
FraudShield ML Pipeline  v4
============================
Fixes and improvements over v3:

  BUG FIXES
  ---------
  - _unwrap_clf now handles CalibratedClassifierCV: was returning the wrapper
    instead of the base estimator, causing feature importance to fall back to
    all-ones and SHAP to try LinearExplainer on a tree model (silently failing).
  - LabelEncoder now includes "Unknown" as an explicit class at fit time.
    Previously unseen categories fell back to le.classes_[0] (alphabetically
    first, e.g. "ATM" for an unseen transaction type) — now they map to "Unknown"
    which is a meaningful, consistent fallback.
  - SHAP: TreeExplainer is now correctly used for RF (after unwrapping calibration).

  NEW FEATURES
  ------------
  - training_metadata stored in artifact: timestamp, sklearn/lgbm versions,
    row count, fraud rate, feature count, best model name.  Exposed via
    GET /api/version so the frontend can display it.
  - Encoder fit is deterministic (sorted class list) and idempotent.
"""

import os
import sys
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.utils.class_weight import compute_class_weight
import sklearn

warnings.filterwarnings("ignore")

# ── Optional dependencies ────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
    LGB_VERSION = lgb.__version__
except ImportError:
    LGB_AVAILABLE = False
    LGB_VERSION = None
    from sklearn.ensemble import GradientBoostingClassifier
    print("⚠️  lightgbm not installed — falling back to GradientBoostingClassifier")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  shap not installed — SHAP explainability disabled")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

TARGET = "Fraud_Label"

_BASE_NUMERIC = [
    "Transaction_Amount", "Distance_From_Home",
    "Account_Balance", "Daily_Transaction_Count",
    "Weekly_Transaction_Count", "Avg_Transaction_Amount",
    "Max_Transaction_Last_24h", "Failed_Transaction_Count",
    "Previous_Fraud_Count", "Hour", "DayOfWeek", "Month",
    "IsWeekend", "IsNight", "Amount_vs_Avg", "Amount_vs_Max24h",
    "Balance_vs_Amount", "Spend_Ratio", "Location_Mismatch",
    "Tx_Velocity_Ratio", "Risk_Flag_Count",
]

_NEW_NUMERIC = [
    "Hour_Sin", "Hour_Cos",
    "Amount_ZScore", "Amount_Spike_Flag", "Max24h_Utilization",
    "Daily_vs_Expected", "Failed_Rate", "Spend_Velocity_Today",
    "Dist_100_500", "Dist_500_2000", "Dist_Over_2000",
    "Balance_Very_Low", "Balance_Low",
    "Intl_x_NewMerchant",
    "Intl_x_Distance",
    "Intl_x_Night",
    "NewMerchant_x_HighAmt",
    "PrevFraud_x_Intl",
    "PrevFraud_x_NewMerchant",
    "PrevFraud_x_AmtRatio",
    "Failed_x_Intl",
    "Risk_x_AmtRatio",
    "Distance_x_Night",
]

NUMERIC_FEATURES     = _BASE_NUMERIC + _NEW_NUMERIC
CATEGORICAL_FEATURES = [
    "Transaction_Type", "Merchant_Category", "Card_Type",
    "Is_International_Transaction", "Is_New_Merchant",
    "Unusual_Time_Transaction",
]


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Temporal ─────────────────────────────────────────────────────────
    df["Hour"] = (
        pd.to_datetime(df["Transaction_Time"], format="%H:%M", errors="coerce")
        .dt.hour.fillna(12).astype(int)
    )
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
    df["DayOfWeek"] = df["Transaction_Date"].dt.dayofweek
    df["Month"]     = df["Transaction_Date"].dt.month
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["IsNight"]   = ((df["Hour"] >= 22) | (df["Hour"] <= 5)).astype(int)

    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    # ── Convenience aliases ───────────────────────────────────────────────
    amt  = df["Transaction_Amount"]
    avg  = df["Avg_Transaction_Amount"]
    mx   = df["Max_Transaction_Last_24h"]
    bal  = df["Account_Balance"]
    wk   = df["Weekly_Transaction_Count"]
    dy   = df["Daily_Transaction_Count"]
    fail = df["Failed_Transaction_Count"]
    pf   = df["Previous_Fraud_Count"]
    dist = df["Distance_From_Home"]

    is_intl = (df["Is_International_Transaction"] == "Yes").astype(int)
    is_new  = (df["Is_New_Merchant"]              == "Yes").astype(int)
    is_unu  = (df["Unusual_Time_Transaction"]     == "Yes").astype(int)

    # ── Ratio features ────────────────────────────────────────────────────
    df["Amount_vs_Avg"]     = amt / (avg + 1e-9)
    df["Amount_vs_Max24h"]  = amt / (mx + 1e-9)
    df["Balance_vs_Amount"] = bal / (amt + 1e-9)
    df["Spend_Ratio"]       = (amt / (bal + 1e-9)).clip(0, 10)
    df["Location_Mismatch"] = (
        df["Transaction_Location"] != df["Customer_Home_Location"]
    ).astype(int)
    df["Tx_Velocity_Ratio"] = (dy / (wk / 7 + 1e-9)).clip(0, 10)
    df["Risk_Flag_Count"]   = (
        is_intl + is_new + is_unu
        + df["Location_Mismatch"]
        + (fail > 0).astype(int)
        + (pf   > 0).astype(int)
    )

    # ── Amount deviation signals ──────────────────────────────────────────
    df["Amount_ZScore"]      = ((amt - avg) / (avg + 1)).clip(-5, 20)
    df["Amount_Spike_Flag"]  = (amt > avg * 2).astype(int)
    df["Max24h_Utilization"] = (mx / (avg + 1e-9)).clip(0, 20)

    # ── Velocity & behavioral anomaly ────────────────────────────────────
    expected_daily             = wk / 7
    df["Daily_vs_Expected"]    = (dy - expected_daily).clip(-5, 20)
    df["Failed_Rate"]          = (fail / (dy + 1)).clip(0, 1)
    df["Spend_Velocity_Today"] = (amt * dy / (avg + 1e-9)).clip(0, 200)

    # ── Distance risk tier binary flags ──────────────────────────────────
    df["Dist_100_500"]   = ((dist >= 100)  & (dist < 500)).astype(int)
    df["Dist_500_2000"]  = ((dist >= 500)  & (dist < 2000)).astype(int)
    df["Dist_Over_2000"] = (dist >= 2000).astype(int)

    # ── Balance tier flags ────────────────────────────────────────────────
    df["Balance_Very_Low"] = (bal < amt * 1.5).astype(int)
    df["Balance_Low"]      = (bal < amt * 5).astype(int)

    # ── Pairwise interaction features ─────────────────────────────────────
    df["Intl_x_NewMerchant"]      = is_intl * is_new
    df["Intl_x_Distance"]         = is_intl * (dist / 1000).clip(0, 20)
    df["Intl_x_Night"]            = is_intl * df["IsNight"]
    df["NewMerchant_x_HighAmt"]   = is_new * df["Amount_Spike_Flag"]
    df["PrevFraud_x_Intl"]        = (pf > 0).astype(int) * is_intl
    df["PrevFraud_x_NewMerchant"] = (pf > 0).astype(int) * is_new
    df["PrevFraud_x_AmtRatio"]    = pf * df["Amount_vs_Avg"].clip(0, 20)
    df["Failed_x_Intl"]           = fail * is_intl
    df["Risk_x_AmtRatio"]         = df["Risk_Flag_Count"] * df["Amount_vs_Avg"].clip(0, 20)
    df["Distance_x_Night"]        = (dist / 1000).clip(0, 20) * df["IsNight"]

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESS  (used by pipeline AND by the API at predict-time)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame, encoders=None, fit: bool = True):
    """
    Engineer features and label-encode categorical columns.

    At fit time  (fit=True):  builds and returns encoders.
    At predict time (fit=False): applies stored encoders, mapping any unseen
    category value to the explicit "Unknown" class that was included at fit.
    """
    df = engineer_features(df)

    if fit:
        encoders = {}

    cat_cols = []
    for col in CATEGORICAL_FEATURES:
        if fit:
            le = LabelEncoder()
            # Include "Unknown" explicitly so unseen values at inference time
            # have a consistent, meaningful fallback instead of landing on
            # whatever happens to be first alphabetically (old bug: "ATM").
            raw_vals = df[col].fillna("Unknown").astype(str).unique().tolist()
            if "Unknown" not in raw_vals:
                raw_vals = raw_vals + ["Unknown"]
            le.fit(sorted(raw_vals))   # sorted for determinism across runs
            encoders[col] = le

        le = encoders[col]
        # Safe map: any value not seen at train time → "Unknown"
        # "Unknown" is guaranteed to be in le.classes_ because we added it above.
        unknown_fallback = "Unknown" if "Unknown" in le.classes_ else le.classes_[0]
        encoded = le.transform(
            df[col].fillna("Unknown").astype(str).map(
                lambda x, _le=le, _unk=unknown_fallback: x if x in _le.classes_ else _unk
            )
        )
        cat_cols.append(pd.Series(encoded, name=col, index=df.index))

    num_df = df[NUMERIC_FEATURES].copy()
    num_df = num_df.fillna(num_df.median())

    X = pd.concat([num_df] + cat_cols, axis=1).reset_index(drop=True)

    if fit:
        y = (df[TARGET] == "Fraud").astype(int).reset_index(drop=True)
        return X, y, encoders
    return X


# ══════════════════════════════════════════════════════════════════════════════
# EDA
# ══════════════════════════════════════════════════════════════════════════════

def _fraud_rate_by(df: pd.DataFrame, col: str) -> list:
    grp = (
        df.groupby(col, observed=True)
        .agg(total=(TARGET, "count"), fraud=(TARGET, lambda x: (x == "Fraud").sum()))
        .reset_index()
    )
    grp["fraud"]      = grp["fraud"].astype(float)
    grp["total"]      = grp["total"].astype(float)
    grp["fraud_rate"] = (grp["fraud"] / grp["total"]).round(6)
    return grp.to_dict("records")


def compute_eda(df: pd.DataFrame) -> dict:
    print("📊 Computing EDA stats…")

    categorical_cols = [
        "Transaction_Type", "Merchant_Category", "Transaction_Location",
        "Customer_Home_Location", "Card_Type", "Is_International_Transaction",
        "Is_New_Merchant", "Unusual_Time_Transaction",
    ]
    for col in categorical_cols:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])

    is_fraud = df[TARGET] == "Fraud"
    df2      = df.copy()
    df2["Hour"] = (
        pd.to_datetime(df2["Transaction_Time"], format="%H:%M", errors="coerce").dt.hour
    )

    hr = (
        df2.groupby("Hour", observed=True)
        .agg(total=(TARGET, "count"), fraud=(TARGET, lambda x: (x == "Fraud").sum()))
        .reset_index()
    )
    hr["fraud_rate"] = (hr["fraud"] / hr["total"]).round(6)

    df2["Combo"] = (
        df2["Is_International_Transaction"].astype(str)
        + " Intl / "
        + df2["Is_New_Merchant"].astype(str)
        + " New"
    )
    combo = (
        df2.groupby("Combo", observed=True)
        .agg(total=(TARGET, "count"), fraud=(TARGET, lambda x: (x == "Fraud").sum()))
        .reset_index()
    )
    combo["fraud_rate"] = (combo["fraud"] / combo["total"]).round(6)

    df2["Previous_Fraud_Count"] = df2["Previous_Fraud_Count"].fillna(0).astype(int)

    num_subset = [
        "Transaction_Amount", "Distance_From_Home",
        "Account_Balance", "Daily_Transaction_Count",
        "Weekly_Transaction_Count", "Failed_Transaction_Count",
        "Previous_Fraud_Count",
    ]
    available = [c for c in num_subset if c in df.columns]
    corr      = df[available].corr().round(3)

    return {
        "total_transactions": int(len(df)),
        "total_fraud":        int(is_fraud.sum()),
        "fraud_rate":         float(is_fraud.mean()),
        "total_amount":       float(df["Transaction_Amount"].sum()),
        "avg_fraud_amount":   float(df.loc[is_fraud, "Transaction_Amount"].mean()),
        "fraud_by_type":          _fraud_rate_by(df, "Transaction_Type"),
        "fraud_by_merchant":      _fraud_rate_by(df, "Merchant_Category"),
        "fraud_by_location":      _fraud_rate_by(df, "Transaction_Location"),
        "fraud_by_international": _fraud_rate_by(df, "Is_International_Transaction"),
        "fraud_by_new_merchant":  _fraud_rate_by(df, "Is_New_Merchant"),
        "fraud_by_prev_fraud":    _fraud_rate_by(df2, "Previous_Fraud_Count"),
        "fraud_by_hour":          hr[["Hour", "fraud_rate"]].to_dict("records"),
        "fraud_by_combo":         combo[["Combo", "fraud_rate"]].to_dict("records"),
        "amount_normal":   df.loc[~is_fraud, "Transaction_Amount"].clip(0, 100000).tolist(),
        "amount_fraud":    df.loc[ is_fraud, "Transaction_Amount"].clip(0, 100000).tolist(),
        "distance_normal": df.loc[~is_fraud, "Distance_From_Home"].tolist(),
        "distance_fraud":  df.loc[ is_fraud, "Distance_From_Home"].tolist(),
        "correlation_matrix": {c: corr[c].to_dict() for c in corr.columns},
    }


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def compute_threshold_analysis(y_test: np.ndarray, y_prob: np.ndarray) -> dict:
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    data = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm     = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        data.append({
            "threshold": float(t),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1":        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "fp": int(fp), "fn": int(fn), "tp": int(tp), "tn": int(tn),
        })

    best_f1   = max(data, key=lambda r: r["f1"])
    best_cost = min(data, key=lambda r: r["fn"] * 5.0 + r["fp"] * 0.1)
    return {
        "optimal_f1_threshold":   best_f1["threshold"],
        "optimal_cost_threshold": best_cost["threshold"],
        "data": data,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SHAP HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _unwrap_clf(model):
    """
    Unwrap nested model wrappers to reach the base estimator.

    Handles:
      - sklearn Pipeline  (has .named_steps) → returns last step
      - CalibratedClassifierCV (has .calibrated_classifiers_) → returns the
        base estimator from the first fold
      - Plain estimator → returned as-is

    FIX (v4): previously returned CalibratedClassifierCV itself, which has no
    feature_importances_ or coef_, causing feature importance to silently fall
    back to all-ones and SHAP to pick the wrong explainer type.
    """
    # Unwrap Pipeline first (e.g. StandardScaler → LogisticRegression)
    if hasattr(model, "named_steps"):
        model = list(model.named_steps.values())[-1]
    # Unwrap CalibratedClassifierCV → get the base estimator from fold 0
    if hasattr(model, "calibrated_classifiers_"):
        model = model.calibrated_classifiers_[0].estimator
    return model


def _shap_values_for_class1(sv):
    """Normalise SHAP output to always return the class-1 (fraud) array."""
    if isinstance(sv, list):
        return sv[1]
    arr = np.asarray(sv)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train(data_path: str, output_dir: str = "models") -> dict:
    print(f"📂 Loading data from {data_path}…")
    df = pd.read_csv(data_path)
    fraud_rate = (df[TARGET] == "Fraud").mean()
    print(f"   {len(df):,} rows  |  Fraud rate: {fraud_rate:.2%}")

    eda = compute_eda(df)

    print("\n🔧 Preprocessing…")
    X, y, encoders = preprocess(df, fit=True)
    feature_names  = list(X.columns)
    print(f"   Feature count: {len(feature_names)}  ({len(_BASE_NUMERIC)} base + {len(_NEW_NUMERIC)} new)")
    print(f"   Encoder classes (Transaction_Type): {list(encoders['Transaction_Type'].classes_)}")

    if len(X) > 200_000:
        idx = X.sample(200_000, random_state=42).index
        X   = X.loc[idx].reset_index(drop=True)
        y   = y.loc[idx].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    weights          = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    cw               = {0: float(weights[0]), 1: float(weights[1])}
    sw_train         = np.where(y_train == 1, weights[1], weights[0])
    pos_weight_ratio = weights[1] / weights[0]

    # ── Model definitions ─────────────────────────────────────────────────
    _rf_base = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight=cw,
        random_state=42,
        n_jobs=-1,
    )
    rf_model = CalibratedClassifierCV(_rf_base, method="isotonic", cv=5)

    if LGB_AVAILABLE:
        boost_name  = "LightGBM"
        boost_model = lgb.LGBMClassifier(
            n_estimators=700,
            learning_rate=0.02,
            max_depth=8,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=pos_weight_ratio,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        boost_needs_sw = False
    else:
        boost_name  = "Gradient Boosting"
        boost_model = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.02, max_depth=6, random_state=42,
        )
        boost_needs_sw = True

    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000, C=0.1,
            class_weight=cw, random_state=42, solver="lbfgs",
        )),
    ])

    model_defs = [
        ("Random Forest",       rf_model,     False),
        (boost_name,            boost_model,  boost_needs_sw),
        ("Logistic Regression", lr_pipeline,  False),
    ]

    if XGB_AVAILABLE:
        xgb_model = xgb.XGBClassifier(
            n_estimators=700, learning_rate=0.02, max_depth=7,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=pos_weight_ratio, eval_metric="aucpr",
            random_state=42, verbosity=0, n_jobs=-1,
        )
        model_defs.append(("XGBoost", xgb_model, False))

    # ── Train & evaluate ──────────────────────────────────────────────────
    print(f"\n🏋️  Training {len(model_defs)} models…")
    model_results = {}

    for name, model, needs_sw in tqdm(model_defs, desc="Models"):
        print(f"\n  → {name}")
        if needs_sw:
            model.fit(X_train, y_train, sample_weight=sw_train)
        else:
            model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        cm_raw = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm_raw.ravel()
        print(f"   Test: {tn+fp+fn+tp} samples | TP={tp}  FP={fp}  FN={fn}  TN={tn}")

        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            Xtr, Xva = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            ytr, yva = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            m2 = clone(model)
            if needs_sw:
                sw2 = np.where(ytr == 1, weights[1], weights[0])
                m2.fit(Xtr, ytr, sample_weight=sw2)
            else:
                m2.fit(Xtr, ytr)
            cv_scores.append(average_precision_score(yva, m2.predict_proba(Xva)[:, 1]))

        roc = float(roc_auc_score(y_test, y_prob))
        pr  = float(average_precision_score(y_test, y_prob))
        print(f"     ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  CV-PR={np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f}")

        opt_t      = max(np.arange(0.05, 0.96, 0.05),
                         key=lambda t: f1_score(y_test, (y_prob >= t).astype(int), zero_division=0))
        cm_opt_raw = confusion_matrix(y_test, (y_prob >= opt_t).astype(int))

        model_results[name] = {
            "model":      model,
            "roc_auc":    roc,
            "pr_auc":     pr,
            "cv_mean":    round(float(np.mean(cv_scores)), 4),
            "cv_std":     round(float(np.std(cv_scores)), 4),
            "brier":      round(float(brier_score_loss(y_test, y_prob)), 4),
            "report":     rep,
            "y_test":     y_test.tolist(),
            "y_prob":     y_prob.tolist(),
            "cm":         cm_raw.tolist(),
            "cm_opt":     cm_opt_raw.tolist(),
            "opt_thresh": round(float(opt_t), 2),
        }

    best_name  = max(model_results, key=lambda k: model_results[k]["pr_auc"])
    best_model = model_results[best_name]["model"]
    print(f"\n🏆 Best model: {best_name}  (ROC-AUC={model_results[best_name]['roc_auc']:.4f})")

    # ── Feature importance ────────────────────────────────────────────────
    # _unwrap_clf now correctly handles CalibratedClassifierCV (v4 fix)
    try:
        inner_clf = _unwrap_clf(best_model)
        imp = (
            inner_clf.feature_importances_
            if hasattr(inner_clf, "feature_importances_")
            else np.abs(inner_clf.coef_[0])
        )
        fi = (
            pd.DataFrame({"feature": feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception as e:
        print(f"   ⚠️  Feature importance extraction failed: {e}")
        fi = pd.DataFrame({"feature": feature_names, "importance": np.ones(len(feature_names))})

    # ── SHAP ─────────────────────────────────────────────────────────────
    # TreeExplainer is now correctly used for RF because _unwrap_clf returns the
    # base RandomForestClassifier (not CalibratedClassifierCV). (v4 fix)
    shap_data      = None
    shap_explainer = None
    if SHAP_AVAILABLE:
        try:
            print("\n🔬 Computing SHAP values…")
            sample    = X_test.sample(min(500, len(X_test)), random_state=42)
            inner_clf = _unwrap_clf(best_model)

            if hasattr(best_model, "named_steps"):
                # LR Pipeline: transform features first, then use LinearExplainer
                pre      = best_model[:-1]
                sample_t = pd.DataFrame(pre.transform(sample),  columns=feature_names)
                bg_t     = pd.DataFrame(pre.transform(X_train.sample(100, random_state=42)), columns=feature_names)
                explainer = shap.LinearExplainer(inner_clf, bg_t)
                sv        = explainer.shap_values(sample_t)
            else:
                # Tree models (LightGBM, RF, XGBoost): use TreeExplainer directly
                explainer = (
                    shap.TreeExplainer(inner_clf)
                    if hasattr(inner_clf, "feature_importances_")
                    else shap.LinearExplainer(inner_clf, shap.sample(X_train, 100))
                )
                sv = explainer.shap_values(sample)

            sv_arr   = _shap_values_for_class1(sv)
            mean_abs = (
                pd.Series(np.abs(sv_arr).mean(axis=0), index=feature_names)
                .sort_values(ascending=False)
            )
            shap_data      = {"mean_abs": mean_abs}
            shap_explainer = explainer
            print("   ✅ SHAP done.")
        except Exception as ex:
            print(f"   ⚠️  SHAP failed: {ex}")

    # ── Calibration ──────────────────────────────────────────────────────
    y_prob_best = np.array(model_results[best_name]["y_prob"])
    y_test_arr  = np.array(model_results[best_name]["y_test"])
    prob_true, prob_pred = calibration_curve(
        y_test_arr, y_prob_best, n_bins=10, strategy="uniform"
    )

    # ── Threshold analysis ────────────────────────────────────────────────
    threshold_analysis = compute_threshold_analysis(y_test_arr, y_prob_best)

    # ── Training metadata ─────────────────────────────────────────────────
    # Stored in the artifact so /api/version can expose it without re-training.
    training_metadata = {
        "trained_at":      datetime.now().isoformat(),
        "pipeline_version": "4",
        "sklearn_version":  sklearn.__version__,
        "lgbm_version":     LGB_VERSION,
        "n_rows":           int(len(df)),
        "n_features":       len(feature_names),
        "fraud_rate":       round(float(fraud_rate), 6),
        "best_model":       best_name,
        "test_set_size":    len(y_test_arr),
        "optimal_f1_threshold": threshold_analysis["optimal_f1_threshold"],
    }
    print(f"\n📋 Training metadata: {training_metadata}")

    # ── Bundle & save ─────────────────────────────────────────────────────
    arts = {
        "best_name":          best_name,
        "best_model":         best_model,
        "model_results":      model_results,
        "feature_names":      feature_names,
        "encoders":           encoders,
        "feature_importance": fi,
        "shap_data":          shap_data,
        "shap_explainer":     shap_explainer,
        "eda":                eda,
        "calibration":        {"prob_pred": prob_pred.tolist(), "prob_true": prob_true.tolist()},
        "threshold_analysis": threshold_analysis,
        "test_set_size":      len(y_test_arr),
        "training_metadata":  training_metadata,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "model.pkl")
    with open(out_path, "wb") as fh:
        pickle.dump(arts, fh)
    print(f"\n✅ Artifacts saved → {out_path}")
    return arts


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/FraudShield_Banking_Data.csv"
    train(data_path)