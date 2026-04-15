# 🛡️ FraudShield 
Fraud Monitoring dashboard with Live fraud detection and business impact calculator.

**Demo:** 
<p align="center">
  <img src="fraudshield_demo.gif" width="100%"/>
</p>
---

HTML/CSS/JS frontend on **GitHub Pages** + FastAPI backend on **Render**.

```
fraudshield/
├── backend/
│   ├── main.py            #FastAPI - /api/health, /api/version, /api/eda, /api/model, /api/predict
│   └── requirements.txt
├── docs/
│   ├── index.html         #GitHub Pages frontend (pure HTML/CSS/JS + Chart.js 4)
│   ├── app.js             #All chart rendering, scoring logic, business impact
│   └── style.css          #Design system
├── src/
│   └── pipeline.py        #ML pipeline (feature engineering, training, SHAP)
├── models/
│   └── model.pkl          #Trained artifact — commit this after training
├── data/
│   └── FraudShield_Banking_Data.csv
├── render.yaml            #Render deployment config
└── README.md
```

---

## Step 1 — Train the model

```bash
pip install -r backend/requirements.txt
python src/pipeline.py data/FraudShield_Banking_Data.csv
```

This creates `models/model.pkl` (~60 seconds).

> **Deployment tip:** Commit `models/model.pkl` to your repo after training.
> Render's build step will detect it and skip re-training, making deploys fast.
> The `data/` folder is gitignored by default; if you want Render to auto-retrain
> on deploy (e.g. after data updates), also commit your CSV and Render will train
> during the build phase automatically.

---

## Step 2 — Run locally (full stack)

**Terminal 1 — Backend:**
```bash
uvicorn backend.main:app --reload
#API running at http://localhost:8000
#Swagger docs at http://localhost:8000/docs
```

**Terminal 2 — Frontend:**
Just open `docs/index.html` in your browser.
Or serve it with any static server:
```bash
python -m http.server 3000 --directory docs
#Open http://localhost:3000
```

The `API_URL` in `docs/app.js` defaults to the Render URL. Change it to
`http://localhost:8000` for local development.

---

## Step 3 — Deploy backend to Render

1. Push this repo to GitHub (include `models/model.pkl`)
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — click **Deploy**
5. Note your service URL: `https://fraudshield-api.onrender.com`

> **Free tier note:** Render free tier spins down after 15 min of inactivity.
> The first request after sleep takes ~30 seconds. Upgrade to Starter ($7/mo)
> for always-on. Alternatively, use Railway or Fly.io.
>
> **CORS note:** `render.yaml` defaults `CORS_ORIGINS` to `*` for demo convenience.
> In production, replace it with your exact frontend origin(s).

---

## Step 4 — Update the frontend API URL

In `docs/app.js`, find line ~3:

```js
const API_URL = 'https://fraudshield-cv7g.onrender.com';
```

Change to your own Render URL:

```js
const API_URL = 'https://your-service-name.onrender.com';
```

---

## Step 5 — Deploy frontend to GitHub Pages

1. Push your repo to GitHub
2. Go to repo **Settings → Pages**
3. Source: **Deploy from branch**
4. Branch: `main` | Folder: `/docs`
5. Save → your site is live at `https://yourusername.github.io/fraudshield`

---

## Optional — Wake the Render API every day

This repo includes a GitHub Actions workflow at `.github/workflows/wake-render.yml`
that pings `/api/health` once per day at `13:00 UTC`.

By default, `scripts/wake_render.sh` targets:

```bash
https://fraudshield-cv7g.onrender.com/api/health
```

If your Render URL is different, set a GitHub repository variable named
`RENDER_HEALTHCHECK_URL` to your own health endpoint, for example:

```text
https://your-service-name.onrender.com/api/health
```

You can also run the script locally:

```bash
bash scripts/wake_render.sh
# or
RENDER_HEALTHCHECK_URL=https://your-service-name.onrender.com/api/health bash scripts/wake_render.sh
```

> Important: a **daily** ping only wakes the service once each day. On Render
> free tier, the service can still go back to sleep after ~15 minutes of
> inactivity. If you want to avoid most cold starts, change the workflow cron to
> run more frequently or upgrade the service plan.

---

## API Reference

| Method | Endpoint        | Description                                                    |
|--------|-----------------|----------------------------------------------------------------|
| `GET`  | `/api/health`   | Liveness check — model name, ROC-AUC, SHAP availability       |
| `GET`  | `/api/version`  | Artifact provenance — trained_at, sklearn/lgbm versions, etc. |
| `GET`  | `/api/eda`      | All pre-computed EDA stats                                     |
| `GET`  | `/api/model`    | Model comparison, curves, SHAP, calibration, threshold data    |
| `POST` | `/api/predict`  | Score a transaction → risk score, tier, flags, SHAP waterfall  |

Interactive docs: `https://your-render-url.onrender.com/docs`

### POST /api/predict — request body
```json
{
  "amount": 5000.0,
  "balance": 20000.0,
  "distance": 5570,
  "tx_time": "14:30",
  "tx_type": "Online",
  "merchant_cat": "Electronics",
  "card_type": "Credit",
  "tx_location": "London",
  "home_loc": "New York",
  "daily_tx": 3,
  "weekly_tx": 10,
  "avg_amount": 3000.0,
  "max_24h": 4000.0,
  "failed": 0,
  "prev_fraud": 0,
  "is_intl": "Yes",
  "is_new": "Yes",
  "unusual": "No"
}
```

### POST /api/predict — response fields

| Field              | Type    | Description                                                              |
|--------------------|---------|--------------------------------------------------------------------------|
| `risk_score`       | float   | Composite risk index, 0–100. ML component (0–60) + rule component (0–40) |
| `risk_score_pct`   | string  | Formatted as `"72.3%"` for display — **not** a calibrated probability    |
| `tier`             | string  | `HIGH` / `MEDIUM` / `LOW` — strictest of score-based and rule-based tier |
| `ml_probability`   | float   | Raw model output probability (0–1)                                       |
| `ml_probability_pct` | string | Formatted ML probability, e.g. `"7.3%"`                                |
| `decision_trace`   | object  | Full audit log: ML tier, rule fired, composite breakdown, final tier     |
| `flags`            | array   | Human-readable risk signals for display                                  |
| `shap_waterfall`   | array   | Top 6 positive and 6 negative SHAP contributors for this transaction     |

> **Note:** `risk_score_pct` and `ml_probability_pct` are distinct. The risk score
> is a composite index anchored to the model's optimal F1 threshold and rule engine
> points — it is not a fraud probability. The raw ML probability is in `ml_probability`.

---

## Scoring System

Transactions are scored via a two-stage hybrid system:

**Stage 1 — ML model** (0–60 pts)
The best-performing model's fraud probability is scaled relative to the optimal
F1 threshold: `min(60, (prob / opt_threshold) × 60)`.

**Stage 2 — Rule engine** (0–40 pts)
Points are awarded for individual risk signals (prior fraud, failed attempts,
international, new merchant, distance, unusual time, location mismatch) and
combination bonuses for co-occurring signals.

**Tier assignment**
The numeric score determines a score-based tier (≥70 → HIGH, ≥35 → MEDIUM).
The rule engine independently determines a rule-based tier. The final tier is
the **stricter of the two** — rules can only raise the tier, never lower it.

**Rule definitions**

| Rule    | Condition                                                   | Tier override |
|---------|-------------------------------------------------------------|---------------|
| RULE_01 | Prior fraud + ≥2 failed attempts + international            | HIGH          |
| RULE_02 | Distance > 5,000 km + international + new merchant          | HIGH          |
| RULE_03 | 5 or more of 6 critical risk signals active simultaneously  | HIGH          |
| RULE_04 | 4 of 6 critical signals active, ML tier is LOW              | MEDIUM        |

---

## Tech Stack

| Layer          | Tech                                              |
|----------------|---------------------------------------------------|
| Frontend       | HTML · CSS · Vanilla JS · Chart.js 4              |
| Backend        | FastAPI · Uvicorn · Pydantic v2                   |
| ML             | scikit-learn · LightGBM · XGBoost · SHAP · pandas |
| Hosting (FE)   | GitHub Pages (free, static)                       |
| Hosting (BE)   | Render (free tier / $7 Starter)                   |

---

## Architecture Notes

- **Model selection** is based on PR-AUC from 5-fold stratified CV — appropriate
  for imbalanced classification (≈10% fraud rate). ROC-AUC is reported but not
  used for selection.
- **Logistic Regression** is wrapped in a `StandardScaler` Pipeline for fair
  comparison against tree models. SHAP uses `LinearExplainer` on the LR step after
  the scaler transforms features; tree models use `TreeExplainer` directly.
- **Calibration** uses `CalibratedClassifierCV` (isotonic, 5-fold) on Random Forest.
  `_unwrap_clf()` correctly unwraps the calibration wrapper before SHAP and feature
  importance extraction — previously this caused both to silently fail.
- **Unseen categories** at inference time are mapped to an explicit `"Unknown"` class
  that is included in every `LabelEncoder` at fit time. Previously unseen values
  fell back to the alphabetically first class (e.g. `"ATM"`), which was incorrect.
- **Composite tier** takes the maximum of the score-based tier and the rule-engine
  tier. A rule override can only raise the final tier, never lower it.
- **Transaction date** uses today's real date so temporal features (`DayOfWeek`,
  `IsWeekend`, `IsNight`) reflect live conditions at scoring time.
- **No demographic features** are used anywhere in the pipeline — no protected-class
  risk.

---

## Known Limitations

- Trained on **synthetic data** — calibration and feature importance may not
  generalise to real transaction distributions without retraining.
- The **free Render tier** spins down after inactivity; the first request after
  sleep takes ~30 s. This repo includes a daily wake workflow, but avoiding most
  cold starts still requires more frequent pings (for example every 5 minutes)
  or upgrading to Render Starter.
- Business impact figures on the Impact tab use **illustrative cost units** from
  the dataset. Calibrate `cost-fn` and `cost-fp` inputs to your institution's
  actual cost structure before drawing conclusions.
- The **risk score (0–100)** displayed in the Live Scorer is a composite index,
  not a calibrated fraud probability. The raw ML probability is shown in the
  decision trace beneath it.
