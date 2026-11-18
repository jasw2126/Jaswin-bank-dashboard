"""JASWIN BANK dashboard.  Run: streamlit run app.py"""
import io
import json
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

BASE = Path(__file__).resolve().parent
DATA, MODELS, REPORTS = BASE / "data", BASE / "models", BASE / "reports"
DB_FILE = DATA / "review_queue.db"

# ---------- Theme: colours taken from the named-colour chart (goldenrod family on black) ----------
BLACK      = "#0A0A0A"
GOLD       = "#D4AF37"   # metallic gold, main accent
GOLDENROD  = "#DAA520"   # goldenrod
DARKGOLD   = "#B8860B"   # darkgoldenrod
LIGHT_GOLD = "#F5D77A"
BURLYWOOD  = "#DEB887"   # burlywood
CHAMPAGNE  = "#FFF8DC"   # cornsilk, body text
MUTED      = "#A8A08A"
PANEL      = "#14110A"
EDGE       = "#3A2F10"
GOLD_SCALE = ["#1A1608", DARKGOLD, GOLD, LIGHT_GOLD]
SERIES     = [GOLD, BURLYWOOD, DARKGOLD, LIGHT_GOLD, "#8C7A3C"]

# ---------- Fonts: picked from the specimen sheet ----------
# Poiret One  -> "Deco" style wordmark      | Cormorant Garamond -> "Holmes / Rouge" style headings
# Manrope     -> clean, readable body text  | all free on Google Fonts
FONT_BRAND, FONT_HEAD, FONT_BODY = "Poiret One", "Cormorant Garamond", "Manrope"

px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = SERIES
px.defaults.color_continuous_scale = GOLD_SCALE
PAGES = ["Portfolio Overview", "Customer View", "Model Comparison", "Review Queue"]
STATUSES = ["Open", "In review", "Contacted", "Closed"]

st.set_page_config(page_title="JASWIN BANK", page_icon="🏦", layout="wide")
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Poiret+One&family=Cormorant+Garamond:ital,wght@0,500;0,600;0,700;1,500&family=Manrope:wght@400;500;600;700&display=swap');

html, body, .stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea,
.stApp button, .stApp [data-baseweb], .stApp [data-testid="stCaptionContainer"] {{
    font-family: '{FONT_BODY}', sans-serif;}}
[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded {{
    font-family: 'Material Symbols Rounded' !important;}}

.stApp {{background: radial-gradient(circle at 12% 0%, #2A200B 0%, #120E05 38%, {BLACK} 70%);
    color: {CHAMPAGNE}; line-height: 1.6;}}
[data-testid="stHeader"] {{background: transparent;}}
.block-container {{padding-top: 2.2rem; max-width: 1400px;}}

/* Headings: Cormorant Garamond, gold */
h1, h2, h3, [data-testid="stHeading"] * {{font-family: '{FONT_HEAD}', serif !important; color: {GOLD};
    letter-spacing: .01em; font-weight: 600;}}
h1 {{font-size: 3rem !important; line-height: 1.1; padding-bottom: .6rem;
    border-bottom: 1px solid {EDGE}; margin-bottom: 1.2rem;}}
h2, h3 {{font-size: 1.7rem !important;}}

/* Sidebar */
[data-testid="stSidebar"] {{background: linear-gradient(180deg, #0D0A03 0%, #050505 100%);
    border-right: 1px solid {EDGE};}}
[data-testid="stSidebar"] h2 {{font-family: '{FONT_BRAND}', cursive !important; font-size: 2.1rem !important;
    letter-spacing: .14em; color: {LIGHT_GOLD}; font-weight: 400; padding-bottom: .4rem;}}
[data-testid="stSidebar"] button {{background: transparent; border: none; box-shadow: none;
    justify-content: flex-start; color: {MUTED}; font-size: 1.02rem; font-weight: 500;
    border-left: 3px solid transparent; border-radius: 0; transition: color .2s, border-color .2s;}}
[data-testid="stSidebar"] button:hover {{color: {LIGHT_GOLD}; border-left-color: {DARKGOLD};}}
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {{color: {GOLD}; font-weight: 700;
    border-left: 3px solid {GOLD}; background: linear-gradient(90deg, rgba(212,175,55,.14), transparent);}}
[data-testid="stSidebar"] hr {{border-color: {EDGE};}}

/* Metric cards */
[data-testid="stMetric"] {{background: linear-gradient(145deg, #1A1508 0%, {PANEL} 100%);
    border: 1px solid {EDGE}; border-top: 2px solid {DARKGOLD}; border-radius: 10px; padding: 16px 18px;}}
[data-testid="stMetricLabel"] * {{color: {MUTED}; font-weight: 500; font-size: .85rem;}}
[data-testid="stMetricValue"] * {{font-family: '{FONT_HEAD}', serif !important; color: {LIGHT_GOLD};
    font-weight: 700; font-size: 2.1rem; line-height: 1.15;}}

/* Tables, inputs, expanders */
[data-testid="stDataFrame"] {{border: 1px solid {EDGE}; border-radius: 10px; overflow: hidden;}}
[data-baseweb="input"], [data-baseweb="select"] > div {{background: {PANEL}; border-color: {EDGE};}}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {{border-color: {GOLD};}}
[data-testid="stForm"], [data-testid="stExpander"] {{background: {PANEL}; border: 1px solid {EDGE}; border-radius: 10px;}}
.stButton > button, [data-testid="stFormSubmitButton"] > button {{border-color: {DARKGOLD}; color: {GOLD};}}
[data-testid="stFormSubmitButton"] > button {{background: {GOLD}; color: {BLACK}; font-weight: 700; border: none;}}
[data-testid="stFormSubmitButton"] > button:hover {{background: {LIGHT_GOLD}; color: {BLACK};}}
[data-testid="stAlert"] {{border-radius: 10px;}}
::selection {{background: {GOLD}; color: {BLACK};}}
:focus-visible {{outline: 2px solid {LIGHT_GOLD}; outline-offset: 2px;}}
</style>""", unsafe_allow_html=True)


# ---------- 1. Files, models and database ----------
def need(path):
    if not path.exists():
        st.error(f"Missing file: {path.relative_to(BASE)}. Run train_models.py first.")
        st.stop()
    return path


def load_csv(path, **kwargs):
    """Reads normal CSVs and CSVs that have every line wrapped in quotes."""
    with open(path, encoding="utf-8-sig") as f:
        lines = [line.strip() for line in f]
    lines = [l[1:-1] if l.startswith('"') and l.endswith('"') else l for l in lines]
    return pd.read_csv(io.StringIO("\n".join(lines)), **kwargs)


def _load_models():
    return joblib.load(need(MODELS / "balance_model.joblib")), joblib.load(need(MODELS / "churn_model.joblib"))


@st.cache_resource(show_spinner="Preparing models for this server (first start only, about a minute)...")
def load_models():
    """Saved models only load on the same scikit-learn version that trained them.
    If loading fails, retrain on this machine with train_models.py, then load again."""
    try:
        return _load_models()
    except Exception:
        subprocess.run([sys.executable, "train_models.py"], cwd=BASE, check=True, capture_output=True)
        return _load_models()


balance_model, churn_model = load_models()

settings = json.load(open(need(MODELS / "churn_settings.json")))
FEATURES, THRESHOLD = settings["features"], settings["threshold"]
NUM = [f for f in FEATURES if f not in ("account_type", "branch_code")]
COLS = ["customer_id", "snapshot_date"] + FEATURES


MODEL_VERSION = f"{settings['classifier']} | {datetime.fromtimestamp((MODELS / 'churn_model.joblib').stat().st_mtime):%Y-%m-%d}"


def run_sql(sql, params=(), many=False):
    con = sqlite3.connect(DB_FILE)
    con.execute("CREATE TABLE IF NOT EXISTS review_queue (customer_id TEXT PRIMARY KEY, status TEXT, note TEXT, updated_at TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS prediction_log (request_id TEXT PRIMARY KEY, customer_id TEXT, logged_at TEXT, "
                "model_version TEXT, churn_probability REAL, balance_forecast_inr REAL, outcome TEXT)")
    result = None
    if sql.lstrip().upper().startswith("SELECT"):
        result = pd.read_sql(sql, con, params=params)
    else:
        con.executemany(sql, params) if many else con.execute(sql, params)
        con.commit()
    con.close()
    return result


# ---------- 2. Load data and score with the ML models ----------
@st.cache_data
def load_history():
    return load_csv(need(DATA / "synthetic_customers.csv"), parse_dates=["snapshot_date"])[COLS]


def load_live():
    """Read the simulated feed. Malformed rows are rejected (negative or non-numeric values)."""
    f = DATA / "live_feed.csv"
    if not f.exists():
        return pd.DataFrame(), pd.DataFrame()
    live = pd.read_csv(f, parse_dates=["snapshot_date"])
    nums = live[NUM].apply(pd.to_numeric, errors="coerce")
    bad = nums.isna().any(axis=1) | (nums < 0).any(axis=1)
    live[NUM] = nums
    return live[~bad], live[bad]


@st.cache_data(show_spinner=False)
def score(df):
    out = df.copy()
    out["balance_forecast_inr"] = balance_model.predict(out[FEATURES]).round(2)
    out["churn_probability"] = churn_model.predict_proba(out[FEATURES])[:, 1].round(4)
    out["flagged"] = out["churn_probability"] >= THRESHOLD
    out["model_version"] = MODEL_VERSION
    out["scored_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return out


history = load_history()
live_ok, live_bad = load_live()
live_scored = score(live_ok[["request_id", "ingested_at"] + COLS]) if len(live_ok) else pd.DataFrame()

all_rows = pd.concat([history, live_ok[COLS] if len(live_ok) else None]).sort_values("snapshot_date")
current = score(all_rows.groupby("customer_id").tail(1).reset_index(drop=True))

# Audit log: request ID, fictional customer ID, timestamp, model version and outcome
log_rows = [(r.request_id, r.customer_id, r.scored_at, r.model_version, float(r.churn_probability),
             float(r.balance_forecast_inr), "flagged for review" if r.flagged else "scored: low risk")
            for r in live_scored.itertuples()]
log_rows += [(r.request_id, r.customer_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), MODEL_VERSION,
              None, None, "rejected: invalid or negative values") for r in live_bad.itertuples()]
if log_rows:
    run_sql("INSERT OR IGNORE INTO prediction_log VALUES (?,?,?,?,?,?,?)", log_rows, many=True)


def style(fig):
    """Gold-on-black chart look with the same fonts as the page."""
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=56, b=10), colorway=SERIES,
        font=dict(family=f"{FONT_BODY}, sans-serif", color=CHAMPAGNE, size=13),
        title=dict(font=dict(family=f"{FONT_HEAD}, serif", size=24, color=GOLD)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=GOLD, font=dict(family=f"{FONT_BODY}, sans-serif", color=CHAMPAGNE)))
    fig.update_xaxes(gridcolor="#2A220E", linecolor=EDGE, zerolinecolor=EDGE, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor="#2A220E", linecolor=EDGE, zerolinecolor=EDGE, tickfont=dict(color=MUTED))
    return fig


# ---------- 3. Sidebar navigation (clickable titles) ----------
@st.cache_resource
def feed_state():
    return {"proc": None}          # remembers the feed process between reruns


FEED = feed_state()
st.session_state.setdefault("page", PAGES[0])
with st.sidebar:
    st.markdown("## JASWIN BANK")
    for p in PAGES:
        if st.button(p, key=f"nav_{p}", width="stretch",
                     type="primary" if st.session_state.page == p else "secondary"):
            st.session_state.page = p
            st.rerun()
    st.divider()
    running = FEED["proc"] is not None and FEED["proc"].poll() is None
    if st.button("⏹ Stop live feed" if running else "▶ Start live feed", key="feed_btn", width="stretch"):
        if running:
            FEED["proc"].terminate()
        else:
            FEED["proc"] = subprocess.Popen([sys.executable, str(BASE / "simulate_feed.py")], cwd=BASE,
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        st.rerun()
    st.caption("Feed is running: new record every 30 s" if running else "Feed is stopped")
    live_on = st.toggle("Live refresh (every 10 s)", value=True)
    max_queue = st.slider("Review queue size", 10, 200, 50, 10)
    st.caption(f"Model: {MODEL_VERSION}\n\nThreshold: {THRESHOLD:.3f}\n\nFictional data only.")
if live_on:
    st_autorefresh(interval=10_000, key="auto_refresh")


# ---------- 4. Pages ----------
def page_overview():
    st.title("Portfolio Overview")
    fresh = live_ok["ingested_at"].max() if len(live_ok) else f"{history.snapshot_date.max():%Y-%m-%d} (history)"
    k = st.columns(5)
    k[0].metric("Customers", f"{len(current):,}")
    k[1].metric("Average balance (INR)", f"{current.avg_balance_inr.mean():,.0f}")
    k[2].metric("Flagged for review", f"{int(current.flagged.sum()):,}")
    k[3].metric("Live records scored", f"{len(live_scored):,}")
    k[4].metric("Data freshness", str(fresh))

    a, b = st.columns(2)
    a.plotly_chart(style(px.histogram(current, x="avg_balance_inr", nbins=40, title="Balance distribution")),
                   width="stretch")
    trend = all_rows.groupby("snapshot_date")[["transaction_count_30d", "digital_sessions_30d"]].mean().reset_index()
    b.plotly_chart(style(px.line(trend, x="snapshot_date", y=["transaction_count_30d", "digital_sessions_30d"],
                                 title="Average activity per month")), width="stretch")
    risk = current.groupby("account_type")["churn_probability"].mean().reset_index()
    a.plotly_chart(style(px.bar(risk, x="account_type", y="churn_probability", title="Average churn score by account type")),
                   width="stretch")
    b.subheader("Latest feed records")
    b.dataframe(live_scored.tail(8)[["ingested_at", "customer_id", "balance_forecast_inr", "churn_probability"]]
                if len(live_scored) else pd.DataFrame({"info": ["Start simulate_feed.py to see live records"]}),
                width="stretch", hide_index=True)


def page_customer():
    st.title("Customer View")
    c1, c2, c3 = st.columns([2, 1, 1])
    query = c1.text_input("Search customer ID", placeholder="e.g. CUST00001").strip().upper()
    types = c2.multiselect("Account type", sorted(current.account_type.unique()))
    only_flagged = c3.checkbox("Flagged customers only")
    pool = current
    if types:
        pool = pool[pool.account_type.isin(types)]
    if only_flagged:
        pool = pool[pool.flagged]
    if query:
        pool = pool[pool.customer_id.str.contains(query)]
    if pool.empty:
        st.warning("No customer matches these filters.")
        return
    customer_id = st.selectbox("Customer", pool.customer_id.tolist()[:500])
    row = pool[pool.customer_id == customer_id].iloc[0]

    m = st.columns(4)
    m[0].metric("Balance forecast (INR)", f"{row.balance_forecast_inr:,.0f}")
    m[1].metric("Churn probability", f"{row.churn_probability:.1%}")
    m[2].metric("Risk status", "Flagged" if row.flagged else "Low risk")
    m[3].metric("Scored at", row.scored_at)
    st.caption(f"Model version: {row.model_version}")

    hist = all_rows[all_rows.customer_id == customer_id]
    fig = px.line(hist, x="snapshot_date", y="avg_balance_inr", markers=True, title=f"{customer_id} balance history and forecast")
    fig.add_scatter(x=[row.snapshot_date + pd.offsets.MonthEnd(1)], y=[row.balance_forecast_inr],
                    mode="markers", name="Forecast", marker=dict(size=14, color=LIGHT_GOLD, symbol="diamond"))
    st.plotly_chart(style(fig), width="stretch")
    st.subheader("Current features")
    st.dataframe(row[FEATURES].astype(str).to_frame("value"), width="stretch")


def page_models():
    st.title("Model Comparison")
    files = {n: REPORTS / n for n in ["churn_model_comparison.csv", "balance_test_results.csv", "final_test_metrics.json",
                                      "test_confusion_matrix.csv", "balance_test_predictions.csv"]}
    missing = [n for n, p in files.items() if not p.exists()]
    if missing:
        st.info("Run train_models.py to create: " + ", ".join(missing))
        return
    metrics = json.load(open(files["final_test_metrics.json"]))
    k = st.columns(5)
    for col, (label, key) in zip(k, [("Test PR-AUC", "test_pr_auc"), ("Precision", "test_precision"), ("Recall", "test_recall"),
                                     ("F1", "test_f1"), ("ROC-AUC", "test_roc_auc")]):
        col.metric(label, f"{metrics[key]:.3f}")
    st.caption(f"Selected classifier: {metrics['classifier']} | test rows: {metrics['test_rows']:,}. "
               "Synthetic-data results show software behaviour, not real-bank accuracy.")

    comp = pd.read_csv(files["churn_model_comparison.csv"])
    a, b = st.columns(2)
    a.subheader("Churn models (validation data)")
    a.dataframe(comp, width="stretch", hide_index=True)
    b.plotly_chart(style(px.bar(comp, x="model", y="PR-AUC", title="Validation PR-AUC")), width="stretch")

    cm = pd.read_csv(files["test_confusion_matrix.csv"], index_col=0)
    a.plotly_chart(style(px.imshow(cm, text_auto=True, color_continuous_scale=GOLD_SCALE, title="Test confusion matrix")),
                   width="stretch")
    pred = pd.read_csv(files["balance_test_predictions.csv"]).sample(3000, random_state=42)
    b.plotly_chart(style(px.scatter(pred, x="predicted_balance_inr", y="residual_inr", opacity=0.4,
                                    title="Balance residuals (actual - predicted)")), width="stretch")
    st.subheader("Balance forecast: baseline vs Linear Regression (test data)")
    st.dataframe(pd.read_csv(files["balance_test_results.csv"]), width="stretch", hide_index=True)


def page_queue():
    st.title("Review Queue")
    st.caption("Customers above the model threshold. The model supports analysts; it never acts on customers.")
    if st.session_state.get("saved"):
        st.success(st.session_state.pop("saved"))
    saved = run_sql("SELECT customer_id, status, note, updated_at FROM review_queue")
    queue = (current[current.flagged].sort_values("churn_probability", ascending=False).head(max_queue)
             .merge(saved, on="customer_id", how="left"))
    queue["status"] = queue["status"].fillna("Open")

    k = st.columns(4)
    k[0].metric("In queue", len(queue))
    for col, s in zip(k[1:], STATUSES[:3]):
        col.metric(s, int((queue.status == s).sum()))

    shown = st.multiselect("Show status", STATUSES, default=STATUSES[:3])
    table = queue[queue.status.isin(shown)][["customer_id", "account_type", "churn_probability",
                                              "balance_forecast_inr", "status", "note", "updated_at"]]
    st.dataframe(table, width="stretch", hide_index=True)

    if not queue.empty:
        with st.form("review_form"):
            st.subheader("Update a review")
            c1, c2 = st.columns(2)
            customer_id = c1.selectbox("Customer", queue.customer_id.tolist())
            status = c2.selectbox("Status", STATUSES)
            note = st.text_input("Review note (no personal information)")
            if st.form_submit_button("Save review"):
                run_sql("INSERT OR REPLACE INTO review_queue VALUES (?,?,?,?)",
                        (customer_id, status, note, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                st.session_state["saved"] = f"Saved: {customer_id} is now '{status}'."
                st.rerun()

    with st.expander("Prediction audit log (latest 50)"):
        log = run_sql("SELECT * FROM prediction_log ORDER BY logged_at DESC LIMIT 50")
        st.dataframe(log, width="stretch", hide_index=True)
        if len(live_bad):
            st.warning(f"{len(live_bad)} malformed feed record(s) rejected.")


{"Portfolio Overview": page_overview, "Customer View": page_customer,
 "Model Comparison": page_models, "Review Queue": page_queue}[st.session_state.page]()
