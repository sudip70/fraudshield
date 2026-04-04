"""
FraudShield ML Pipeline  v3
============================
Changes in this version:
  - Feature count: 21 → 46 (+25 new features)
    · Cyclical hour encoding (sin/cos) — preserves circular continuity 23:00 ≈ 00:00
    · Amount z-score & spike flag — captures deviation from personal baseline
    · Velocity anomaly — daily vs expected daily from weekly cadence
    · Failed transaction rate — normalised by today's activity
    · Spend velocity — dollar throughput today vs baseline
    · Distance risk tier binary flags (100-500, 500-2000, 2000+)
    · Balance tier flags (very_low, low) for spend-context signals
    · Max24h utilisation — how high is today's max vs rolling average
    · 10 pairwise interaction features — intl×new, prevfraud×distance, etc.
  - LightGBM: n_estimators 300→700, lr 0.05→0.02, num_leaves=63, min_child=20,
              subsample/colsample=0.8, L1/L2 regularisation
  - Random Forest: n_estimators 150→300, max_depth 10→None, min_samples_leaf=4
  - Logistic Regression: C=0.1 (stronger regularisation), max_iter=2000
  - XGBoost added as 4th model (skipped gracefully if not installed)
  - All previous fixes retained (Pipeline LR, clone CV, SHAP compat, etc.)
"""

import os
import sys
import pickle
import warnings

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

warnings.filterwarnings("ignore")

# ── Optional dependencies ────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    from sklearn.ensemble import GradientBoostingClassifier
    print("⚠️  lightgbm not installed — falling back to GradientBoostingClassifier")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  xgboost not installed — XGBoost model skipped")

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
    # Cyclical time encoding
    "Hour_Sin", "Hour_Cos",
    # Amount deviation signals
    "Amount_ZScore", "Amount_Spike_Flag", "Max24h_Utilization",
    # Velocity & behavioral anomaly
    "Daily_vs_Expected", "Failed_Rate", "Spend_Velocity_Today",
    # Distance risk tiers
    "Dist_100_500", "Dist_500_2000", "Dist_Over_2000",
    # Balance tiers
    "Balance_Very_Low", "Balance_Low",
    # 10 pairwise interaction features
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

    # Cyclical hour encoding — 23:00 and 00:00 are adjacent, not 23 apart
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

    # ── Original ratio features (unchanged) ──────────────────────────────
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

    # ── NEW: Amount deviation signals ─────────────────────────────────────
    # Z-score style deviation from personal baseline — more robust than raw ratio
    df["Amount_ZScore"]      = ((amt - avg) / (avg + 1)).clip(-5, 20)
    # Hard binary flag: this tx is more than 2× the personal average
    df["Amount_Spike_Flag"]  = (amt > avg * 2).astype(int)
    # How high is today's rolling max relative to average?
    df["Max24h_Utilization"] = (mx / (avg + 1e-9)).clip(0, 20)

    # ── NEW: Velocity & behavioral anomaly ───────────────────────────────
    expected_daily             = wk / 7
    # Absolute gap between today's actual tx count and expected daily pace
    df["Daily_vs_Expected"]    = (dy - expected_daily).clip(-5, 20)
    # Fraction of today's transactions that failed
    df["Failed_Rate"]          = (fail / (dy + 1)).clip(0, 1)
    # Total dollar throughput today, relative to personal average spend
    df["Spend_Velocity_Today"] = (amt * dy / (avg + 1e-9)).clip(0, 200)

    # ── NEW: Distance risk tier binary flags ─────────────────────────────
    # Separate bins allow the model to weight each range independently
    df["Dist_100_500"]   = ((dist >= 100)  & (dist < 500)).astype(int)
    df["Dist_500_2000"]  = ((dist >= 500)  & (dist < 2000)).astype(int)
    df["Dist_Over_2000"] = (dist >= 2000).astype(int)

    # ── NEW: Balance tier flags ───────────────────────────────────────────
    df["Balance_Very_Low"] = (bal < amt * 1.5).astype(int)  # nearly depleting account
    df["Balance_Low"]      = (bal < amt * 5).astype(int)    # low headroom

    # ── NEW: Pairwise interaction features ────────────────────────────────
    # Encode the most predictive two-way combinations directly —
    # tree models can find these via splits but explicit features reduce
    # the depth required and make them available to linear models too.

    # International × New Merchant — highest fraud-rate combo in EDA
    df["Intl_x_NewMerchant"]      = is_intl * is_new

    # International × normalised distance — intl AND far from home
    df["Intl_x_Distance"]         = is_intl * (dist / 1000).clip(0, 20)

    # International × night — overseas charge during unusual hours
    df["Intl_x_Night"]            = is_intl * df["IsNight"]

    # New merchant × amount spike — first-time merchant + large amount
    df["NewMerchant_x_HighAmt"]   = is_new * df["Amount_Spike_Flag"]

    # Prior fraud × international — known fraudster travelling
    df["PrevFraud_x_Intl"]        = (pf > 0).astype(int) * is_intl

    # Prior fraud × new merchant — repeat offender at new merchants
    df["PrevFraud_x_NewMerchant"] = (pf > 0).astype(int) * is_new

    # Prior fraud × amount ratio — known fraudster making large tx
    df["PrevFraud_x_AmtRatio"]    = pf * df["Amount_vs_Avg"].clip(0, 20)

    # Failed transactions × international — card testing abroad
    df["Failed_x_Intl"]           = fail * is_intl

    # Total risk flag count × amount deviation — high-risk profile + big spend
    df["Risk_x_AmtRatio"]         = df["Risk_Flag_Count"] * df["Amount_vs_Avg"].clip(0, 20)

    # Far from home × night — physically displaced at unusual hour
    df["Distance_x_Night"]        = (dist / 1000).clip(0, 20) * df["IsNight"]

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESS  (used by pipeline AND by the API at predict-time)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame, encoders=None, fit: bool = True):
    df = engineer_features(df)

    if fit:
        encoders = {}

    cat_cols = []
    for col in CATEGORICAL_FEATURES:
        if fit:
            le = LabelEncoder()
            le.fit(df[col].fillna("Unknown").astype(str))
            encoders[col] = le

        le       = encoders[col]
        safe_map = lambda x, _le=le: x if x in _le.classes_ else _le.classes_[0]
        encoded  = le.transform(df[col].fillna("Unknown").astype(str).map(safe_map))
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
    if hasattr(model, "named_steps"):
        return list(model.named_steps.values())[-1]
    return model


def _shap_values_for_class1(sv):
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
    print(f"   {len(df):,} rows  |  Fraud rate: {(df[TARGET]=='Fraud').mean():.2%}")

    eda = compute_eda(df)

    print("\n🔧 Preprocessing…")
    X, y, encoders = preprocess(df, fit=True)
    feature_names  = list(X.columns)
    print(f"   Feature count: {len(feature_names)}  ({len(_BASE_NUMERIC)} base + {len(_NEW_NUMERIC)} new)")

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

    # FIX: max_depth=None produced badly compressed probabilities — almost every
    # transaction scored near 0, giving excellent AUC rankings but 0.003 recall
    # at any usable threshold. Capping depth + isotonic calibration corrects this.
    _rf_base = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,            # restored depth cap — uncapped RF compresses P(fraud)→0
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
            num_leaves=63,           # 2^(max_depth-1) — controls model complexity
            min_child_samples=20,    # prevents tiny leaf overfitting
            subsample=0.8,           # row subsampling
            colsample_bytree=0.8,    # feature subsampling
            reg_alpha=0.1,           # L1
            reg_lambda=1.0,          # L2
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
            max_iter=2000, C=0.1,    # stronger regularisation: C=0.1 vs default C=1
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
            n_estimators=700,
            learning_rate=0.02,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=pos_weight_ratio,
            eval_metric="aucpr",
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
        model_defs.append(("XGBoost", xgb_model, False))

    # ── Train & evaluate ─────────────────────────────────────────────────
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
            # FIX: CV now scores on PR-AUC — the right metric for imbalanced classes.
            # ROC-AUC inflates apparent performance when negatives dominate.
            cv_scores.append(average_precision_score(yva, m2.predict_proba(Xva)[:, 1]))

        roc = float(roc_auc_score(y_test, y_prob))
        pr  = float(average_precision_score(y_test, y_prob))
        print(f"     ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  CV-PR={np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f}")

        # Store CM at 0.5 (for reference) and also at the best-F1 threshold
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
            "cm":         cm_raw.tolist(),       # at 0.5 threshold
            "cm_opt":     cm_opt_raw.tolist(),   # at best-F1 threshold
            "opt_thresh": round(float(opt_t), 2),
        }

    # FIX: select best model by PR-AUC — not ROC-AUC.
    # ROC-AUC is inflated for imbalanced data and selected RF which had 0.003 recall.
    # PR-AUC directly measures precision/recall trade-off where it matters.
    best_name  = max(model_results, key=lambda k: model_results[k]["pr_auc"])
    best_model = model_results[best_name]["model"]
    print(f"\n🏆 Best model: {best_name}  (ROC-AUC={model_results[best_name]['roc_auc']:.4f})")

    # ── Feature importance ────────────────────────────────────────────────
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
    except Exception:
        fi = pd.DataFrame({"feature": feature_names, "importance": np.ones(len(feature_names))})

    # ── SHAP ─────────────────────────────────────────────────────────────
    shap_data      = None
    shap_explainer = None
    if SHAP_AVAILABLE:
        try:
            print("\n🔬 Computing SHAP values…")
            sample    = X_test.sample(min(300, len(X_test)), random_state=42)
            inner_clf = _unwrap_clf(best_model)

            if hasattr(best_model, "named_steps"):
                pre      = best_model[:-1]
                sample_t = pd.DataFrame(pre.transform(sample),  columns=feature_names)
                bg_t     = pd.DataFrame(pre.transform(X_train.sample(100, random_state=42)), columns=feature_names)
                explainer = shap.LinearExplainer(inner_clf, bg_t)
                sv        = explainer.shap_values(sample_t)
            else:
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

    # ── Bundle & save ────────────────────────────────────────────────────
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