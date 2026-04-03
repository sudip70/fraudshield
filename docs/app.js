// ── CONFIG ────────────────────────────────────────────────────────────────────
const API_URL = 'http://localhost:8000';
// Production: const API_URL = 'https://fraudshield-api.onrender.com';
// ─────────────────────────────────────────────────────────────────────────────

// ── CITY COORDINATES (for auto distance calculation) ──────────────────────────
const CITY_COORDS = {
  "New York":      [40.7128,  -74.0060],
  "Toronto":       [43.6532,  -79.3832],
  "Los Angeles":   [34.0522, -118.2437],
  "Chicago":       [41.8781,  -87.6298],
  "Vancouver":     [49.2827, -123.1207],
  "Montreal":      [45.5017,  -73.5673],
  "Dallas":        [32.7767,  -96.7970],
  "San Francisco": [37.7749, -122.4194],
  "Calgary":       [51.0447, -114.0719],
  "Detroit":       [42.3314,  -83.0458],
  "Miami":         [25.7617,  -80.1918],
  "Atlanta":       [33.7490,  -84.3880],
  "Seattle":       [47.6062, -122.3321],
  "Denver":        [39.7392, -104.9903],
  "Boston":        [42.3601,  -71.0589],
  "Phoenix":       [33.4484, -112.0742],
  "London":        [51.5074,   -0.1278],
  "Paris":         [48.8566,    2.3522],
  "Berlin":        [52.5200,   13.4050],
  "Madrid":        [40.4168,   -3.7038],
  "Rome":          [41.9028,   12.4964],
  "Amsterdam":     [52.3676,    4.9041],
  "Dublin":        [53.3498,   -6.2603],
  "Zurich":        [47.3769,    8.5472],
  "Vienna":        [48.2082,   16.3738],
  "Brussels":      [50.8503,    4.3517],
  "Copenhagen":    [55.6761,   12.5683],
  "Stockholm":     [59.3293,   18.0686],
  "Oslo":          [59.9139,   10.7522],
  "Helsinki":      [60.1695,   24.9354],
  "Lisbon":        [38.7223,   -9.1393],
};

function haversine(c1, c2) {
  if (!CITY_COORDS[c1] || !CITY_COORDS[c2]) return null;
  const [lat1, lon1] = CITY_COORDS[c1];
  const [lat2, lon2] = CITY_COORDS[c2];
  const R = 6371, toR = Math.PI / 180;
  const dLat = (lat2 - lat1) * toR, dLon = (lon2 - lon1) * toR;
  const a = Math.sin(dLat / 2) ** 2
          + Math.cos(lat1 * toR) * Math.cos(lat2 * toR) * Math.sin(dLon / 2) ** 2;
  return Math.round(R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)));
}

// ── STATE ─────────────────────────────────────────────────────────────────────
let EDA_DATA     = null;
let MODEL_DATA   = null;
let THRESH_DATA  = null;
let scoreHistory = [];
let charts       = {};

// ── CHART DEFAULTS ────────────────────────────────────────────────────────────
const GRID  = '#1E2A3A';
const TEXT  = '#C5D4E8';
const MUTED = '#3D5270';
const MINT  = '#00C896';
const RED   = '#E8455A';
const AMBER = '#E8A020';
const BLUE  = '#3B82F6';
const PUR   = '#8B5CF6';

Chart.defaults.color                              = '#7A92AE';
Chart.defaults.borderColor                        = GRID;
Chart.defaults.font.family                        = "'IBM Plex Mono', monospace";
Chart.defaults.font.size                          = 10;
Chart.defaults.plugins.legend.labels.boxWidth     = 10;
Chart.defaults.plugins.legend.labels.padding      = 14;
Chart.defaults.plugins.legend.labels.color        = TEXT;
Chart.defaults.plugins.tooltip.backgroundColor    = '#141920';
Chart.defaults.plugins.tooltip.borderColor        = GRID;
Chart.defaults.plugins.tooltip.borderWidth        = 1;
Chart.defaults.plugins.tooltip.titleColor         = TEXT;
Chart.defaults.plugins.tooltip.bodyColor          = '#7A92AE';
Chart.defaults.plugins.tooltip.padding            = 10;
Chart.defaults.animation.duration                 = 600;

function makeChart(id, config) {
  if (charts[id]) charts[id].destroy();
  const ctx = document.getElementById(id);
  if (!ctx) return null;
  charts[id] = new Chart(ctx, config);
  return charts[id];
}

// ── HELPERS ───────────────────────────────────────────────────────────────────
function lerpColor(a, b, t) {
  const ah = a.replace('#', ''), bh = b.replace('#', '');
  return '#' + [0, 2, 4].map(i => {
    const av = parseInt(ah.slice(i, i + 2), 16);
    const bv = parseInt(bh.slice(i, i + 2), 16);
    return Math.round(av + (bv - av) * t).toString(16).padStart(2, '0');
  }).join('');
}

function el(id)           { return document.getElementById(id); }
function setText(id, val) { const e = el(id); if (e) e.textContent = val; }

const pct = v => (v * 100).toFixed(2) + '%';

function animCount(element, target, suffix = '', dec = 0) {
  if (!element) return;
  const dur = 900, t0 = performance.now();
  const step = t => {
    const p = Math.min((t - t0) / dur, 1);
    element.textContent = (target * (1 - Math.pow(1 - p, 3))).toFixed(dec) + suffix;
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ── TABS ──────────────────────────────────────────────────────────────────────
function switchTab(id, btn) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  el('tab-' + id).classList.add('active');
  btn.classList.add('active');
  if (id === 'model'  && MODEL_DATA) renderModelCharts(MODEL_DATA);
  if (id === 'impact' && MODEL_DATA) recalcImpact();
}

// ── API ───────────────────────────────────────────────────────────────────────
async function apiFetch(path) {
  const res = await fetch(API_URL + path, { method: 'GET', headers: { 'Content-Type': 'application/json' } });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ── BOOT ──────────────────────────────────────────────────────────────────────
async function boot() {
  try {
    const health = await apiFetch('/api/health');
    el('status-badge').innerHTML =
      `<div class="status-dot"></div> API Online · ${health.model.split(' ')[0]} · AUC ${health.roc_auc}`;

    [EDA_DATA, MODEL_DATA] = await Promise.all([
      apiFetch('/api/eda'),
      apiFetch('/api/model'),
    ]);

    THRESH_DATA = MODEL_DATA.threshold_analysis.data;
    populateKPIs(EDA_DATA, MODEL_DATA);
    renderEDACharts(EDA_DATA);
  } catch (e) {
    el('status-badge').innerHTML = `<span style="color:var(--warning)">⚠ API Offline</span>`;
    el('api-banner').classList.add('show');
    console.warn('API unreachable:', e);
  }
}

// ── KPIs ──────────────────────────────────────────────────────────────────────
function populateKPIs(eda, model) {
  const o = eda.overview;
  animCount(el('kpi-total'), o.total_transactions, '', 0);
  animCount(el('kpi-fraud'), o.total_fraud, '', 0);
  setText('kpi-rate', pct(o.fraud_rate));

  // FIX: divide by 1,000,000 before appending 'M' — was showing raw number + 'M'
  setText('kpi-vol', (o.total_amount / 1_000_000).toFixed(1) + 'M');

  const best = model.comparison.find(m => m.is_best);
  setText('kpi-model', best.name.split(' ')[0]);
  setText('kpi-auc',   'ROC-AUC ' + best.roc_auc);
  setText('kpi-cv',    best.cv_mean + ' ± ' + best.cv_std);
}

// ── CHART HELPER ──────────────────────────────────────────────────────────────
function hBarConfig(labels, data, c1, c2) {
  const colors = labels.map((_, i) => lerpColor(c1, c2, labels.length > 1 ? i / (labels.length - 1) : 1));
  return {
    type: 'bar',
    data: { labels, datasets: [{ data, backgroundColor: colors, borderWidth: 0 }] },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: GRID }, ticks: { callback: v => (v * 100).toFixed(1) + '%' }, border: { color: GRID } },
        y: { grid: { display: false }, border: { display: false }, ticks: { color: TEXT } },
      },
    },
  };
}

// ══════════════════════════════════════════════════════════════════════════════
// EDA CHARTS
// ══════════════════════════════════════════════════════════════════════════════
function renderEDACharts(d) {
  const sort = arr => [...arr].sort((a, b) => a.fraud_rate - b.fraud_rate);

  const typ = sort(d.fraud_by_type);
  makeChart('chart-type', hBarConfig(typ.map(r => r.Transaction_Type), typ.map(r => r.fraud_rate), BLUE, MINT));

  const mer = sort(d.fraud_by_merchant);
  makeChart('chart-merchant', hBarConfig(mer.map(r => r.Merchant_Category), mer.map(r => r.fraud_rate), PUR, '#EC4899'));

  const cty = sort(d.fraud_by_location);
  makeChart('chart-city', hBarConfig(cty.map(r => r.Transaction_Location), cty.map(r => r.fraud_rate), BLUE, MINT));

  // Hourly line
  const hr = [...d.fraud_by_hour].filter(r => r.Hour != null).sort((a, b) => a.Hour - b.Hour);
  makeChart('chart-hour', {
    type: 'line',
    data: {
      labels: hr.map(r => r.Hour),
      datasets: [{
        data: hr.map(r => r.fraud_rate),
        borderColor: MINT, borderWidth: 1.5,
        pointBackgroundColor: MINT, pointRadius: 2,
        fill: true,
        backgroundColor: ctx => {
          const g = ctx.chart.ctx.createLinearGradient(0, 0, 0, ctx.chart.height);
          g.addColorStop(0, MINT + '30'); g.addColorStop(1, MINT + '00'); return g;
        },
        tension: 0.4,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: GRID }, border: { color: GRID }, ticks: { color: MUTED } },
        y: { grid: { color: GRID }, ticks: { callback: v => (v * 100).toFixed(1) + '%' }, border: { color: GRID } },
      },
    },
  });

  // Amount distribution
  makeChart('chart-amount', {
    type: 'line',
    data: {
      labels: d.amount_dist.normal.x.map(v => v.toFixed(0)),
      datasets: [
        { label: 'Normal', data: d.amount_dist.normal.y, borderColor: MINT, borderWidth: 1.5, fill: true, backgroundColor: MINT + '18', tension: 0.4, pointRadius: 0 },
        { label: 'Fraud',  data: d.amount_dist.fraud.y,  borderColor: RED,  borderWidth: 1.5, fill: true, backgroundColor: RED + '18',  tension: 0.4, pointRadius: 0 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: true, position: 'top' } },
      scales: {
        x: { grid: { color: GRID }, border: { color: GRID }, ticks: { color: MUTED, maxTicksLimit: 6 } },
        y: { grid: { color: GRID }, border: { color: GRID }, ticks: { color: MUTED } },
      },
    },
  });

  // Prior fraud bar
  const pf = d.fraud_by_prev_fraud;
  makeChart('chart-prevfraud', {
    type: 'bar',
    data: {
      labels: pf.map(r => String(r.Previous_Fraud_Count)),
      datasets: [{
        data: pf.map(r => r.fraud_rate),
        backgroundColor: pf.map((_, i) => lerpColor(AMBER, RED, pf.length > 1 ? i / (pf.length - 1) : 1)),
        borderWidth: 0, borderRadius: 2,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, border: { display: false }, ticks: { color: MUTED } },
        y: { grid: { color: GRID }, ticks: { callback: v => (v * 100).toFixed(1) + '%' }, border: { color: GRID } },
      },
    },
  });

  // Combo
  const cb = sort(d.fraud_by_combo);
  makeChart('chart-combo', hBarConfig(cb.map(r => r.Combo), cb.map(r => r.fraud_rate), BLUE, RED));
}

// ══════════════════════════════════════════════════════════════════════════════
// MODEL CHARTS
// ══════════════════════════════════════════════════════════════════════════════

function renderModelCharts(d) {
  // FIX: check live Chart instance instead of a stale boolean flag —
  // allows re-render if charts are destroyed without a full page reload.
  if (charts['chart-roc'] && !charts['chart-roc'].destroyed) return;

  const palette = [MINT, PUR, '#EC4899'];

  // Comparison table
  const thead = `<tr>
    <th>Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>CV AUC (5-fold)</th>
    <th>Brier ↓</th><th>Precision</th><th>Recall</th><th>F1</th>
  </tr>`;
  const tbody = d.comparison.map(m => `
    <tr class="${m.is_best ? 'best-row' : ''}">
      <td>${m.name} ${m.is_best ? '<span class="badge-best">BEST</span>' : ''}</td>
      <td class="num">${m.roc_auc}</td>
      <td class="num">${m.pr_auc}</td>
      <td class="num">${m.cv_mean} <span style="color:var(--text-faint)">±${m.cv_std}</span></td>
      <td>${m.brier}</td>
      <td>${m.precision}</td>
      <td>${m.recall}</td>
      <td class="num">${m.f1}</td>
    </tr>`).join('');
  el('model-table-wrap').innerHTML =
    `<table class="data-table"><thead>${thead}</thead><tbody>${tbody}</tbody></table>`;

  // ROC
  makeChart('chart-roc', {
    type: 'line',
    data: {
      datasets: [
        ...d.comparison.map((m, i) => ({
          label: `${m.name.split(' ')[0]} ${m.roc_auc}`,
          data: d.curves[m.name].roc.fpr.map((x, j) => ({ x, y: d.curves[m.name].roc.tpr[j] })),
          borderColor: palette[i], borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0,
        })),
        { label: 'Random', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: GRID, borderWidth: 1, borderDash: [5, 4], pointRadius: 0, fill: false },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: true, position: 'top' } },
      scales: {
        x: { type: 'linear', title: { display: true, text: 'FPR', color: MUTED }, grid: { color: GRID }, border: { color: GRID } },
        y: { title: { display: true, text: 'TPR', color: MUTED }, grid: { color: GRID }, border: { color: GRID } },
      },
    },
  });

  // PR
  makeChart('chart-pr', {
    type: 'line',
    data: {
      datasets: [
        ...d.comparison.map((m, i) => ({
          label: `${m.name.split(' ')[0]} ${m.pr_auc}`,
          data: d.curves[m.name].pr.recall.map((x, j) => ({ x, y: d.curves[m.name].pr.precision[j] })),
          borderColor: palette[i], borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0,
        })),
        { label: `Baseline ${pct(d.fraud_rate)}`, data: [{ x: 0, y: d.fraud_rate }, { x: 1, y: d.fraud_rate }], borderColor: GRID, borderWidth: 1, borderDash: [5, 4], pointRadius: 0, fill: false },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: true, position: 'top' } },
      scales: {
        x: { type: 'linear', title: { display: true, text: 'Recall', color: MUTED }, grid: { color: GRID }, border: { color: GRID } },
        y: { title: { display: true, text: 'Precision', color: MUTED }, grid: { color: GRID }, border: { color: GRID } },
      },
    },
  });

  // Confusion matrix
  const cm = d.confusion_matrix;
  el('cm-wrap').innerHTML = `
    <div class="cm-grid">
      <div class="cm-label"></div>
      <div class="cm-label">Pred Normal</div>
      <div class="cm-label">Pred Fraud</div>
      <div class="cm-label">Actual Normal</div>
      <div class="cm-cell tn">${cm[0][0].toLocaleString()}<div class="cm-cell-type">TN</div></div>
      <div class="cm-cell fp">${cm[0][1].toLocaleString()}<div class="cm-cell-type">FP</div></div>
      <div class="cm-label">Actual Fraud</div>
      <div class="cm-cell fn">${cm[1][0].toLocaleString()}<div class="cm-cell-type">FN</div></div>
      <div class="cm-cell tp">${cm[1][1].toLocaleString()}<div class="cm-cell-type">TP</div></div>
    </div>`;

  // Feature importance
  const fi = [...d.feature_importance].reverse();
  makeChart('chart-fi', hBarConfig(fi.map(f => f.feature), fi.map(f => f.importance), PUR, MINT));

  // SHAP
  if (d.shap_global && d.shap_global.length) {
    const sh = [...d.shap_global].reverse();
    makeChart('chart-shap', hBarConfig(sh.map(f => f.feature), sh.map(f => f.value), '#14B8A6', PUR));
  }

  // Calibration
  if (d.calibration && d.calibration.prob_pred) {
    const cal = d.calibration;
    makeChart('chart-cal', {
      type: 'line',
      data: {
        datasets: [
          { label: 'Model',   data: cal.prob_pred.map((x, i) => ({ x, y: cal.prob_true[i] })), borderColor: MINT, borderWidth: 1.5, pointBackgroundColor: MINT, pointRadius: 4, fill: false },
          { label: 'Perfect', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],                        borderColor: GRID, borderWidth: 1, borderDash: [5, 5], pointRadius: 0, fill: false },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: true, position: 'top' } },
        scales: {
          x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'Mean Predicted Prob', color: MUTED }, grid: { color: GRID }, border: { color: GRID } },
          y: { min: 0, max: 1, title: { display: true, text: 'Fraction Positives', color: MUTED }, grid: { color: GRID }, border: { color: GRID } },
        },
      },
    });
  }

  renderThresholdChart();

  // Model card
  const best = d.comparison.find(m => m.is_best);
  el('model-card-content').innerHTML = `
    <div class="model-card-title">${best.name}</div>
    <strong>DATASET</strong> — 50,000 banking transactions · ${pct(d.fraud_rate)} fraud rate<br>
    <strong>IMBALANCE</strong> — Handled via class weighting / scale_pos_weight<br>
    <strong>EVALUATION</strong> — Stratified 80/20 split + 5-fold stratified CV<br>
    <strong>PRIMARY METRIC</strong> — PR-AUC (appropriate for imbalanced classification)<br>
    <strong>OPT. THRESHOLD</strong> — ${d.threshold_analysis.optimal_f1_threshold} (F1) · ${d.threshold_analysis.optimal_cost_threshold} (cost-optimal)<br>
    <strong style="color:var(--warning)">LIMITATIONS</strong> — Trained on synthetic data; calibration may drift on real distributions<br>
    <strong style="color:var(--warning)">BIAS CHECK</strong> — No demographic features used — no protected-class risk`;

  // FIX: show both Normal (0) and Fraud (1) class metrics — was hardcoding '—' for Normal
  el('clf-report-wrap').innerHTML = `
    <table class="data-table">
      <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
      <tbody>
        <tr>
          <td>Normal (0)</td>
          <td class="num">${best.precision_normal}</td>
          <td class="num">${best.recall_normal}</td>
          <td class="num">${best.f1_normal}</td>
        </tr>
        <tr class="best-row">
          <td>Fraud (1)</td>
          <td class="num">${best.precision}</td>
          <td class="num">${best.recall}</td>
          <td class="num">${best.f1}</td>
        </tr>
      </tbody>
    </table>`;
}

// ── THRESHOLD ─────────────────────────────────────────────────────────────────
function renderThresholdChart() {
  if (!THRESH_DATA) return;
  makeChart('chart-threshold', {
    type: 'line',
    data: {
      labels: THRESH_DATA.map(r => r.threshold),
      datasets: [
        { label: 'Precision', data: THRESH_DATA.map(r => r.precision), borderColor: MINT,  borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0.3 },
        { label: 'Recall',    data: THRESH_DATA.map(r => r.recall),    borderColor: PUR,   borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0.3 },
        { label: 'F1',        data: THRESH_DATA.map(r => r.f1),        borderColor: AMBER, borderWidth: 2,   pointRadius: 0, fill: false, tension: 0.3 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: true, position: 'top' } },
      scales: {
        x: { grid: { color: GRID }, border: { color: GRID }, ticks: { maxTicksLimit: 10 } },
        y: { min: 0, max: 1, grid: { color: GRID }, border: { color: GRID } },
      },
    },
  });
  updateThresholdMetrics(0.40);
}

function updateThresholdMetrics(t) {
  if (!THRESH_DATA) return;
  const row = THRESH_DATA.reduce((p, c) => Math.abs(c.threshold - t) < Math.abs(p.threshold - t) ? c : p);
  setText('tm-prec', row.precision.toFixed(3));
  setText('tm-rec',  row.recall.toFixed(3));
  setText('tm-f1',   row.f1.toFixed(3));
  setText('tm-fp',   row.fp.toLocaleString());
  setText('tm-fn',   row.fn.toLocaleString());
}

// ══════════════════════════════════════════════════════════════════════════════
// LIVE SCORER
// ══════════════════════════════════════════════════════════════════════════════
function updateDistance() {
  const distEl = el('f-distance');
  if (!distEl) return;
  const d = haversine(el('f-home').value, el('f-location').value);
  // FIX: only update if both cities are known; preserve existing value otherwise
  if (d !== null) distEl.value = d;
}

async function scoreTransaction() {
  const btn = el('score-btn');
  btn.disabled   = true;
  btn.textContent = '⏳ Scoring…';

  const payload = {
    amount:       +el('f-amount').value,
    balance:      +el('f-balance').value,
    distance:     +el('f-distance').value,
    tx_time:      el('f-time').value,
    tx_type:      el('f-type').value,
    merchant_cat: el('f-merchant').value,
    card_type:    el('f-card').value,
    tx_location:  el('f-location').value,
    home_loc:     el('f-home').value,
    daily_tx:     +el('f-daily').value,
    weekly_tx:    +el('f-weekly').value,
    avg_amount:   +el('f-avg').value,
    max_24h:      +el('f-max24').value,
    failed:       +el('f-failed').value,
    prev_fraud:   +el('f-prevfraud').value,
    is_intl:      el('f-intl').value,
    is_new:       el('f-new').value,
    unusual:      el('f-unusual').value,
  };

  try {
    const res = await fetch(API_URL + '/api/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderResult(data, payload);
    scoreHistory.unshift({ ...payload, prob: data.probability_pct, tier: data.tier });
    renderHistory();
  } catch (e) {
    alert('Could not reach API. Make sure the backend is running.\n\n' + e.message);
  } finally {
    btn.disabled   = false;
    btn.textContent = '⚡ Analyse Transaction';
  }
}

function renderResult(data, input) {
  const tier  = data.tier.toLowerCase();
  const panel = el('result-panel');
  panel.className = 'result-panel ' + tier;

  el('result-empty').style.display   = 'none';
  el('result-content').style.display = 'block';

  el('result-prob').className   = 'prob-number ' + tier;
  el('result-prob').textContent = data.probability_pct;

  el('result-tier').className   = 'tier-badge ' + tier;
  el('result-tier').textContent = data.tier + ' RISK';

  const gaugeColors = { high: RED, medium: AMBER, low: MINT };
  el('gauge-fill').style.width      = (data.probability * 100) + '%';
  el('gauge-fill').style.background = gaugeColors[tier];

  setText('result-meta-line', `${data.probability_pct}  ·  optimal threshold ${data.optimal_threshold}`);

  // SHAP waterfall
  const shapSec = el('shap-section');
  const shapWf  = el('shap-waterfall');
  if (data.shap_waterfall && data.shap_waterfall.length) {
    shapSec.style.display = 'block';
    const maxAbs = Math.max(...data.shap_waterfall.map(r => Math.abs(r.value)));
    shapWf.innerHTML = data.shap_waterfall.map(r => {
      const pctW = maxAbs > 0 ? Math.abs(r.value) / maxAbs * 100 : 0;
      const pos  = r.value > 0;
      const col  = pos ? RED : MINT;
      return `<div class="shap-row">
        <div class="shap-feat" title="${r.feature}">${r.feature}</div>
        <div class="shap-bar-track">
          <div class="shap-bar-fill" style="left:${pos ? 50 : 50 - pctW / 2}%;width:${pctW / 2}%;background:${col}"></div>
          <div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--border2)"></div>
        </div>
        <div class="shap-val" style="color:${col}">${r.value > 0 ? '+' : ''}${r.value.toFixed(4)}</div>
      </div>`;
    }).join('');
  } else {
    shapSec.style.display = 'none';
  }

  // Flags
  const flagsEl = el('result-flags');
  flagsEl.innerHTML = data.flags.length
    ? data.flags.map(f => `<div class="flag-item"><span class="icon">${f.icon}</span><span>${f.text}</span></div>`).join('')
    : '<div class="flag-item"><span class="icon">✅</span><span>No strong individual risk signals detected</span></div>';

  el('result-meta').innerHTML =
    `<span>Model: ${data.model}</span><span>ROC-AUC ${data.roc_auc}</span>`;
}

function renderHistory() {
  if (!scoreHistory.length) return;
  el('history-section').style.display = 'block';
  const tierColors = { HIGH: RED, MEDIUM: AMBER, LOW: MINT };
  el('history-body').innerHTML = scoreHistory.map((r, i) => `
    <tr>
      <td style="color:var(--text-faint)">${scoreHistory.length - i}</td>
      <td>${r.tx_type}</td>
      <td class="num">$${r.amount.toLocaleString()}</td>
      <td>${r.tx_location}</td>
      <td class="num">${r.prob}</td>
      <td><span style="color:${tierColors[r.tier]};font-weight:600">${r.tier}</span></td>
    </tr>`).join('');
}

function clearHistory() {
  scoreHistory = [];
  el('history-section').style.display = 'none';
}

// ══════════════════════════════════════════════════════════════════════════════
// BUSINESS IMPACT
// ══════════════════════════════════════════════════════════════════════════════
function recalcImpact() {
  if (!THRESH_DATA || !MODEL_DATA) return;
  const costFn  = +el('cost-fn').value    || 5.0;
  const costFp  = +el('cost-fp').value    || 0.00005;
  const monthly = +el('monthly-vol').value || 50000;

  // FIX: use actual test set size from the API instead of hardcoded 10,000
  // so business impact projections stay correct if the dataset size changes.
  const testSize = MODEL_DATA.test_set_size || 10000;
  const scale    = monthly / testSize;

  const rows = THRESH_DATA.map(r => ({
    t:      r.threshold,
    total:  (r.fn * costFn + r.fp * costFp) * scale,
    fn_c:   r.fn * costFn * scale,
    fp_c:   r.fp * costFp * scale,
    caught: r.tp * costFn * scale,
    net:    (r.tp * costFn - r.fp * costFp) * scale,
    tp:     r.tp,
  }));

  const optRow       = rows.reduce((a, b) => a.total < b.total ? a : b);
  const baselineCost = rows[rows.length - 1].fn_c;
  const savingsRows  = rows.map(r => ({ t: r.t, s: baselineCost - r.total }));
  const bestSave     = savingsRows.reduce((a, b) => a.s > b.s ? a : b);

  setText('biz-caught',         'M ' + optRow.caught.toFixed(1));
  setText('biz-missed',         'M ' + optRow.fn_c.toFixed(1));
  setText('biz-fp-cost',        'M ' + optRow.fp_c.toFixed(4));
  setText('biz-net',            'M ' + optRow.net.toFixed(1));
  setText('biz-annual-save',    'M ' + (bestSave.s * 12).toFixed(1));
  setText('biz-annual-cost',    'M ' + (optRow.total * 12).toFixed(1));
  setText('biz-monthly-caught', Math.round(optRow.tp / testSize * monthly).toLocaleString());

  makeChart('chart-cost', {
    type: 'line',
    data: {
      labels: rows.map(r => r.t),
      datasets: [
        { label: 'Total Cost', data: rows.map(r => r.total), borderColor: RED,   borderWidth: 1.5, fill: true, backgroundColor: RED + '12', tension: 0.3, pointRadius: 0 },
        { label: 'FN Cost',    data: rows.map(r => r.fn_c),  borderColor: AMBER, borderWidth: 1.5, fill: false, tension: 0.3, pointRadius: 0 },
        { label: 'FP Cost',    data: rows.map(r => r.fp_c),  borderColor: PUR,   borderWidth: 1,   fill: false, tension: 0.3, pointRadius: 0, borderDash: [4, 3] },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: true, position: 'top' } },
      scales: {
        x: { grid: { color: GRID }, border: { color: GRID }, ticks: { maxTicksLimit: 10 } },
        y: { grid: { color: GRID }, border: { color: GRID } },
      },
    },
  });

  makeChart('chart-savings', {
    type: 'line',
    data: {
      labels: savingsRows.map(r => r.t),
      datasets: [{
        label: 'Savings vs no-model',
        data:  savingsRows.map(r => r.s),
        borderColor: MINT, borderWidth: 1.5,
        fill: true,
        backgroundColor: ctx => {
          const g = ctx.chart.ctx.createLinearGradient(0, 0, 0, ctx.chart.height);
          g.addColorStop(0, MINT + '30'); g.addColorStop(1, MINT + '00'); return g;
        },
        tension: 0.3, pointRadius: 0,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: GRID }, border: { color: GRID }, ticks: { maxTicksLimit: 10 } },
        y: { grid: { color: GRID }, border: { color: GRID } },
      },
    },
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// INIT — single entry point
// ══════════════════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  // Set API URL display
  const urlEl = el('api-url-display');
  if (urlEl) urlEl.textContent = API_URL;

  // Wire up auto-distance
  const homeEl = el('f-home');
  const locEl  = el('f-location');
  if (homeEl) homeEl.addEventListener('change', updateDistance);
  if (locEl)  locEl.addEventListener('change',  updateDistance);
  updateDistance();

  // FIX: threshold slider listener belongs inside DOMContentLoaded —
  // was at module level, would silently fail if script were ever moved to <head>.
  const slider = el('threshold-slider');
  if (slider) {
    slider.addEventListener('input', function () {
      const v = parseFloat(this.value);
      setText('slider-val-display', v.toFixed(2));
      updateThresholdMetrics(v);
    });
  }

  boot();
});