"""
FraudShield — FastAPI Backend  v4
==================================
Endpoints:
  GET  /api/health
  GET  /api/version
  GET  /api/eda
  GET  /api/model
  POST /api/predict

Environment variables (set in Render dashboard or render.yaml):
  ARTIFACT_PATH   Path to model.pkl  (default: models/model.pkl relative to repo root)
  CORS_ORIGINS    Comma-separated allowed origins  (default: * — lock down in production)
  LOG_LEVEL       Python logging level  (default: INFO)
"""

import logging
import os
import sys
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pipeline import preprocess, _shap_values_for_class1

# ── Environment variables ──────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# CORS_ORIGINS: in production set to your GitHub Pages URL, e.g.
#   "https://yourusername.github.io"
# Multiple origins: comma-separated, e.g.
#   "https://yourusername.github.io,https://myapp.com"
_cors_raw    = os.environ.get("CORS_ORIGINS", "*")
CORS_ORIGINS = [o.strip() for o in _cors_raw.split(",")]

# ARTIFACT_PATH: absolute or relative to the repo root.
# Render's working directory is the repo root, so the default works out of the box.
ARTIFACT_PATH = os.environ.get(
    "ARTIFACT_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl"),
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("fraudshield")

log.info("CORS_ORIGINS  = %s", CORS_ORIGINS)
log.info("ARTIFACT_PATH = %s", ARTIFACT_PATH)
log.info("LOG_LEVEL     = %s", LOG_LEVEL)

# ── Load artifacts once at startup ────────────────────────────────────────────
def load_artifacts():
    path = os.path.abspath(ARTIFACT_PATH)
    if not os.path.exists(path):
        raise RuntimeError(
            f"Model artifact not found at {path}.\n"
            "Run locally:  python src/pipeline.py data/FraudShield_Banking_Data.csv\n"
            "Then commit:  git add models/model.pkl && git commit -m 'add model artifact'"
        )
    with open(path, "rb") as f:
        arts = pickle.load(f)
    log.info(
        "Artifacts loaded — best model: %s  ROC-AUC: %.4f  test_set_size: %d",
        arts["best_name"],
        arts["model_results"][arts["best_name"]]["roc_auc"],
        arts.get("test_set_size", 0),
    )
    return arts


arts = load_artifacts()

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FraudShield API",
    description="Fraud detection ML backend — EDA stats, model metrics, live scoring",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Utility ────────────────────────────────────────────────────────────────────
def clean_nan(val):
    """Recursively replace NaN / Inf / numpy scalars with JSON-safe equivalents."""
    if isinstance(val, dict):
        return {k: clean_nan(v) for k, v in val.items()}
    if isinstance(val, list):
        return [clean_nan(v) for v in val]
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return None if (np.isnan(val) or np.isinf(val)) else float(val)
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    return val


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/health")
def health():
    """
    Lightweight liveness check — safe to ping every 5 minutes from UptimeRobot
    to prevent Render free-tier spin-down.
    """
    best = arts["best_name"]
    return {
        "status":  "ok",
        "model":   best,
        "roc_auc": arts["model_results"][best]["roc_auc"],
        "shap":    arts.get("shap_data") is not None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# VERSION
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/version")
def version():
    """
    Artifact provenance — when the model was trained, on what data, with what
    software versions. Supports model governance and UI freshness indicators.
    """
    meta = arts.get("training_metadata", {})
    return {
        "api_version":          "4.0.0",
        "pipeline_version":     meta.get("pipeline_version", "unknown"),
        "trained_at":           meta.get("trained_at"),
        "sklearn_version":      meta.get("sklearn_version"),
        "lgbm_version":         meta.get("lgbm_version"),
        "n_training_rows":      meta.get("n_rows"),
        "n_features":           meta.get("n_features"),
        "fraud_rate":           meta.get("fraud_rate"),
        "best_model":           meta.get("best_model", arts["best_name"]),
        "test_set_size":        arts.get("test_set_size"),
        "optimal_f1_threshold": meta.get(
            "optimal_f1_threshold",
            arts["threshold_analysis"]["optimal_f1_threshold"],
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# EDA
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/eda")
def eda():
    e    = arts["eda"]
    bins = np.linspace(0, 100000, 22).tolist()
    bcs  = ((np.array(bins[:-1]) + np.array(bins[1:])) / 2).tolist()

    def _hist(data):
        counts, _ = np.histogram(data, bins=bins, density=True)
        return {"x": bcs, "y": [clean_nan(float(c)) for c in counts.tolist()]}

    response = {
        "overview": {
            "total_transactions": e["total_transactions"],
            "total_fraud":        e["total_fraud"],
            "fraud_rate":         round(e["fraud_rate"], 6),
            "total_amount":       round(e["total_amount"], 2),
            "avg_fraud_amount":   round(e["avg_fraud_amount"], 4),
        },
        "fraud_by_type":          e["fraud_by_type"],
        "fraud_by_merchant":      e["fraud_by_merchant"],
        "fraud_by_location":      e["fraud_by_location"],
        "fraud_by_international": e["fraud_by_international"],
        "fraud_by_new_merchant":  e["fraud_by_new_merchant"],
        "fraud_by_prev_fraud":    e["fraud_by_prev_fraud"],
        "fraud_by_hour":          e["fraud_by_hour"],
        "fraud_by_combo":         e["fraud_by_combo"],
        "amount_dist": {
            "normal": _hist(e["amount_normal"]),
            "fraud":  _hist(e["amount_fraud"]),
        },
        "distance_dist": {
            "normal_median": clean_nan(float(np.median(e["distance_normal"]))),
            "fraud_median":  clean_nan(float(np.median(e["distance_fraud"]))),
            "normal_p75":    clean_nan(float(np.percentile(e["distance_normal"], 75))),
            "fraud_p75":     clean_nan(float(np.percentile(e["distance_fraud"], 75))),
        },
        "correlation": e.get("correlation_matrix", {}),
    }
    return clean_nan(response)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/model")
def model_info():
    from sklearn.metrics import roc_curve, precision_recall_curve

    results = arts["model_results"]
    best    = arts["best_name"]

    def _ds(arr, n=200):
        arr = np.asarray(arr)
        idx = np.round(np.linspace(0, len(arr) - 1, min(n, len(arr)))).astype(int)
        return arr[idx].tolist()

    curves = {}
    for name, r in results.items():
        y_t = np.array(r["y_test"])
        y_p = np.array(r["y_prob"])
        fpr, tpr, _ = roc_curve(y_t, y_p)
        pre, rec, _ = precision_recall_curve(y_t, y_p)
        curves[name] = {
            "roc": {"fpr": _ds(fpr), "tpr": _ds(tpr)},
            "pr":  {"precision": _ds(pre), "recall": _ds(rec)},
        }

    comparison = []
    for name, r in results.items():
        rep        = r["report"]
        fraud_rep  = rep.get("1", rep.get(1, {}))
        normal_rep = rep.get("0", rep.get(0, {}))
        comparison.append({
            "name":             name,
            "roc_auc":          round(r["roc_auc"],  4),
            "pr_auc":           round(r["pr_auc"],   4),
            "cv_mean":          round(r["cv_mean"],  4),
            "cv_std":           round(r["cv_std"],   4),
            "brier":            round(r["brier"],    4),
            "precision":        round(fraud_rep.get("precision", 0),  4),
            "recall":           round(fraud_rep.get("recall", 0),     4),
            "f1":               round(fraud_rep.get("f1-score", 0),   4),
            "precision_normal": round(normal_rep.get("precision", 0), 4),
            "recall_normal":    round(normal_rep.get("recall", 0),    4),
            "f1_normal":        round(normal_rep.get("f1-score", 0),  4),
            "is_best":          name == best,
        })

    fi_rows = [
        {"feature": row["feature"], "importance": round(float(row["importance"]), 6)}
        for _, row in arts["feature_importance"].head(15).iterrows()
    ]

    shap_global = []
    sd = arts.get("shap_data")
    if sd is not None:
        top_shap = sd["mean_abs"].head(15)
        shap_global = [
            {"feature": k, "value": round(float(v), 6)}
            for k, v in top_shap.items()
        ]

    cal    = arts.get("calibration", {})
    thresh = arts["threshold_analysis"]
    best_r = results[best]

    return {
        "best_name":                   best,
        "comparison":                  comparison,
        "curves":                      curves,
        "confusion_matrix":            best_r["cm"],
        "confusion_matrix_opt":        best_r.get("cm_opt", best_r["cm"]),
        "confusion_matrix_opt_thresh": best_r.get("opt_thresh", 0.5),
        "feature_importance":          fi_rows,
        "shap_global":                 shap_global,
        "calibration":                 cal,
        "threshold_analysis": {
            "optimal_f1_threshold":   thresh["optimal_f1_threshold"],
            "optimal_cost_threshold": thresh["optimal_cost_threshold"],
            "data": [
                {k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()}
                for row in thresh["data"]
            ],
        },
        "fraud_rate":    arts["eda"]["fraud_rate"],
        "test_set_size": arts.get("test_set_size", len(best_r["y_test"])),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
class TransactionInput(BaseModel):
    amount:       float
    balance:      float
    distance:     float
    tx_time:      str
    tx_type:      str
    merchant_cat: str
    card_type:    str
    tx_location:  str
    home_loc:     str
    daily_tx:     int
    weekly_tx:    int
    avg_amount:   float
    max_24h:      float
    failed:       int
    prev_fraud:   int
    is_intl:      str
    is_new:       str
    unusual:      str

    @validator("tx_time")
    def validate_time_format(cls, v):
        try:
            datetime.strptime(v, "%H:%M")
        except ValueError:
            raise ValueError("tx_time must be HH:MM format, e.g. '14:30'")
        return v

    @validator("is_intl", "is_new", "unusual")
    def validate_yes_no(cls, v):
        if v not in ("Yes", "No"):
            raise ValueError("Must be 'Yes' or 'No'")
        return v

    @validator("amount", "balance", "avg_amount", "max_24h")
    def validate_non_negative_float(cls, v):
        if v < 0:
            raise ValueError("Must be >= 0")
        return v

    @validator("daily_tx", "weekly_tx")
    def validate_tx_counts(cls, v):
        if v < 1:
            raise ValueError("Transaction counts must be >= 1")
        return v

    @validator("failed", "prev_fraud")
    def validate_non_negative_int(cls, v):
        if v < 0:
            raise ValueError("Must be >= 0")
        return v


@app.post("/api/predict")
def predict(tx: TransactionInput):
    today = datetime.now().strftime("%Y-%m-%d")
    log.info(
        "predict  amount=%.2f  location=%s  intl=%s  new=%s  prev_fraud=%d  failed=%d",
        tx.amount, tx.tx_location, tx.is_intl, tx.is_new, tx.prev_fraud, tx.failed,
    )

    row = pd.DataFrame([{
        "Transaction_Amount":           tx.amount,
        "Transaction_Time":             tx.tx_time,
        "Transaction_Date":             today,
        "Transaction_Type":             tx.tx_type,
        "Merchant_Category":            tx.merchant_cat,
        "Transaction_Location":         tx.tx_location,
        "Customer_Home_Location":       tx.home_loc,
        "Distance_From_Home":           tx.distance,
        "Card_Type":                    tx.card_type,
        "Account_Balance":              tx.balance,
        "Daily_Transaction_Count":      tx.daily_tx,
        "Weekly_Transaction_Count":     tx.weekly_tx,
        "Avg_Transaction_Amount":       tx.avg_amount,
        "Max_Transaction_Last_24h":     tx.max_24h,
        "Is_International_Transaction": tx.is_intl,
        "Is_New_Merchant":              tx.is_new,
        "Failed_Transaction_Count":     tx.failed,
        "Unusual_Time_Transaction":     tx.unusual,
        "Previous_Fraud_Count":         tx.prev_fraud,
        # Required by engineer_features but not used as model features
        "Transaction_ID": 0, "Customer_ID": 0, "Merchant_ID": 0,
        "Device_ID": 0, "IP_Address": "0.0.0.0", "Fraud_Label": "Normal",
    }])

    try:
        X = preprocess(row, encoders=arts["encoders"], fit=False)
        X = X.reindex(columns=arts["feature_names"], fill_value=0)
        prob = float(arts["best_model"].predict_proba(X)[0][1])
    except Exception as e:
        log.error("Feature engineering / inference error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # ── ML-tier thresholds ────────────────────────────────────────────────
    # opt_t: probability threshold that maximises F1 on the held-out test set.
    # HIGH   >= opt_t         (model's best operating point)
    # MEDIUM >= opt_t * 0.5   (elevated but below critical)
    # LOW    <  opt_t * 0.5
    opt_t    = arts["threshold_analysis"]["optimal_f1_threshold"]
    high_t   = opt_t
    medium_t = round(opt_t * 0.5, 4)

    if prob >= high_t:
        ml_tier = "HIGH"
    elif prob >= medium_t:
        ml_tier = "MEDIUM"
    else:
        ml_tier = "LOW"

    tier = ml_tier  # may be upgraded by rule engine below

    # ── Rule engine ───────────────────────────────────────────────────────
    # Hybrid ML + rules is standard in production fraud systems.
    # Rules catch hard patterns the model can underweight due to data sparsity
    # in rare but critical combinations. Rules never change ml_probability —
    # only tier. Every fired rule is recorded in decision_trace.
    rule_fired = False
    rule_name  = None
    rule_tier  = None

    critical_flags = sum([
        tx.prev_fraud > 0,
        tx.failed >= 2,
        tx.is_intl == "Yes",
        tx.is_new  == "Yes",
        tx.distance > 2000,
        tx.unusual == "Yes",
    ])

    if tx.prev_fraud > 0 and tx.failed >= 2 and tx.is_intl == "Yes":
        # Known fraudster + card-testing pattern + cross-border
        tier       = "HIGH"
        rule_fired = True
        rule_name  = "RULE_01: prior fraud + card-testing (≥2 failed) + international"
        rule_tier  = "HIGH"

    elif tx.distance > 5000 and tx.is_intl == "Yes" and tx.is_new == "Yes":
        # Extreme geographic displacement + first-time merchant abroad
        tier       = "HIGH"
        rule_fired = True
        rule_name  = f"RULE_02: extreme displacement ({int(tx.distance):,} km) + international + new merchant"
        rule_tier  = "HIGH"

    elif critical_flags >= 5:
        # 5+ of 6 critical signals simultaneously active
        tier       = "HIGH"
        rule_fired = True
        rule_name  = f"RULE_03: {critical_flags}/6 critical risk signals simultaneously active"
        rule_tier  = "HIGH"

    elif critical_flags >= 4 and tier == "LOW":
        # Many signals but model scored low — elevate for human review
        tier       = "MEDIUM"
        rule_fired = True
        rule_name  = f"RULE_04: {critical_flags}/6 critical risk signals — elevated for review"
        rule_tier  = "MEDIUM"

    # ── Composite risk score (0–100) ──────────────────────────────────────
    # ML component  (0–60 pts): maps prob → 0–60 using opt_t as the 60-pt anchor.
    # Rule component (0–40 pts): weighted active risk signals + combination bonuses.
    ml_component = min(60.0, (prob / max(opt_t, 1e-9)) * 60.0)

    rule_points = 0.0
    if tx.prev_fraud > 0:             rule_points += 12.0
    if tx.failed >= 2:                rule_points += 8.0
    if tx.is_intl == "Yes":           rule_points += 5.0
    if tx.is_new  == "Yes":           rule_points += 5.0
    if tx.distance > 2000:            rule_points += 5.0
    elif tx.distance > 500:           rule_points += 2.0
    if tx.unusual == "Yes":           rule_points += 3.0
    if tx.tx_location != tx.home_loc: rule_points += 2.0
    # Combination bonuses — non-additive risk
    if tx.is_intl == "Yes" and tx.is_new == "Yes":  rule_points += 5.0
    if tx.prev_fraud > 0 and tx.is_intl == "Yes":   rule_points += 5.0
    if tx.prev_fraud > 0 and tx.failed >= 2:         rule_points += 5.0
    rule_component = min(40.0, rule_points)

    risk_score = round(ml_component + rule_component, 1)

    # ── Composite tier ────────────────────────────────────────────────────
    if risk_score >= 70 or tier == "HIGH":
        composite_tier = "HIGH"
    elif risk_score >= 35 or tier == "MEDIUM":
        composite_tier = "MEDIUM"
    else:
        composite_tier = "LOW"

    # risk_score_pct: composite index as "XX.X%" — NOT a probability.
    risk_score_pct = f"{risk_score:.1f}%"

    log.info(
        "scored  ml_prob=%.4f  ml_tier=%s  rule=%s  risk_score=%.1f  tier=%s",
        prob, ml_tier, rule_name or "none", risk_score, composite_tier,
    )

    # ── Decision trace ────────────────────────────────────────────────────
    decision_trace = {
        "ml_probability":       round(prob, 6),
        "ml_tier":              ml_tier,
        "optimal_f1_threshold": opt_t,
        "medium_threshold":     medium_t,
        "rule_engine": {
            "fired":                 rule_fired,
            "rule_id":               rule_name,
            "tier_override":         rule_tier,
            "critical_flags_active": critical_flags,
        },
        "composite": {
            "ml_component":    round(ml_component, 2),
            "rule_component":  round(rule_component, 2),
            "total":           risk_score,
            "tier_thresholds": {"HIGH": 70, "MEDIUM": 35},
        },
        "final_tier": composite_tier,
    }

    # ── Display flags ─────────────────────────────────────────────────────
    flags = []
    if tx.is_intl == "Yes":
        flags.append({"icon": "🌍", "text": "International transaction"})
    if tx.is_new == "Yes":
        flags.append({"icon": "🏪", "text": "New merchant — no prior history"})
    if tx.unusual == "Yes":
        flags.append({"icon": "🕐", "text": "Unusual transaction time"})
    if tx.tx_location != tx.home_loc:
        flags.append({"icon": "📍", "text": f"Location mismatch: {tx.tx_location} vs home {tx.home_loc}"})
    if tx.distance > 400:
        flags.append({"icon": "📏", "text": f"{int(tx.distance):,} km from home"})
    if tx.failed > 0:
        flags.append({"icon": "❌", "text": f"{tx.failed} failed transaction(s) in session"})
    if tx.prev_fraud > 0:
        flags.append({"icon": "⚠️", "text": f"Prior fraud history: {tx.prev_fraud} incident(s)"})
    if tx.avg_amount > 0 and tx.amount > tx.avg_amount * 2:
        flags.append({
            "icon": "💰",
            "text": f"Amount spike: ${tx.amount:,.0f} vs avg ${tx.avg_amount:,.0f} ({tx.amount / tx.avg_amount:.1f}×)",
        })

    # ── SHAP waterfall ────────────────────────────────────────────────────
    shap_waterfall = []
    expl = arts.get("shap_explainer")
    if expl is not None:
        try:
            source_model = arts["best_model"]
            # LR Pipeline: transform through scaler before LinearExplainer.
            # Tree models (LightGBM, RF, XGBoost): raw features → TreeExplainer.
            if hasattr(source_model, "named_steps"):
                X_shap = pd.DataFrame(
                    source_model[:-1].transform(X),
                    columns=arts["feature_names"],
                )
            else:
                X_shap = X

            sv      = expl.shap_values(X_shap)
            sv_arr  = _shap_values_for_class1(sv)
            sv_flat = sv_arr[0]

            series = pd.Series(sv_flat, index=arts["feature_names"])
            top    = pd.concat([series.nlargest(6), series.nsmallest(6)]).sort_values()
            shap_waterfall = [
                {"feature": k, "value": round(float(v), 5)}
                for k, v in top.items()
            ]
        except Exception as e:
            log.warning("SHAP waterfall failed (non-fatal): %s", e)

    # ── Response ──────────────────────────────────────────────────────────
    return {
        # Primary display fields
        "risk_score":         risk_score,        # composite 0-100 index
        "risk_score_pct":     risk_score_pct,     # "XX.X%" for UI display
        "tier":               composite_tier,     # HIGH / MEDIUM / LOW

        # Raw ML output — kept for full transparency
        "ml_probability":     round(prob, 6),
        "ml_probability_pct": f"{prob:.1%}",
        "ml_tier":            ml_tier,

        # Explainability
        "decision_trace":     decision_trace,
        "rule_override":      rule_name,
        "flags":              flags,
        "shap_waterfall":     shap_waterfall,

        # Metadata
        "model":              arts["best_name"],
        "roc_auc":            round(arts["model_results"][arts["best_name"]]["roc_auc"], 4),
        "optimal_threshold":  opt_t,
    }