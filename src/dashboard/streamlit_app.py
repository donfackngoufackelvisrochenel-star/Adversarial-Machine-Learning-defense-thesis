"""
Streamlit interactive dashboard for the AML Defense project.

Provides a GUI for:
  - Selecting a dataset from data/raw/
  - Configurable chunked loading
  - Training RF, XGBoost, and LightGBM models
  - Running FGSM / PGD evasion attacks
  - Applying defense strategies
  - Visualising results with plots, tables, and confusion matrices

Run with:  streamlit run src/dashboard/streamlit_app.py
"""

import sys
import io
import time
import datetime
import getpass
import os
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.config import DATA_RAW_DIR, MODELS_DIR, CHUNKSIZE, MAX_ROWS

# Authentication credentials — matched to configs/config.py for consistency
# Change these in configs/config.py or the .env file before production deployment.
import os as _os
_AUTH_USERNAME = _os.environ.get("AUTH_USERNAME", "admin")
_AUTH_PASSWORD = _os.environ.get("AUTH_PASSWORD", "ciciomt2024")
from src.preprocessing.loader import load_data, clean_data, split_data, find_dataset
from src.preprocessing.processor import DataProcessor
from src.models.trainer import train_all_models
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.defenses.adversarial_training import adversarial_training
from src.defenses.feature_selection import feature_selection_defense
from src.defenses.ensemble import ensemble_defense
from src.defenses.feature_squeezing import feature_squeezing_defense
from src.evaluation.metrics import compute_metrics, attack_success_rate
from src.models.trainer import load_model
from src.utils.helpers import check_dataset_exists

# ---------------------------------------------------------------------------
# Page configuration — MUST be first Streamlit command
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AML Defense Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True

    st.markdown("""
    <style>
        .login-box {
            max-width: 400px; margin: 100px auto; padding: 2rem;
            background: #f8f9fa; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            text-align: center;
        }
        .login-box h1 { font-size: 1.8rem; color: #1E3A5F; margin-bottom: 0.5rem; }
        .login-box p { color: #6B7C93; margin-bottom: 1.5rem; }
    </style>
    <div class="login-box">
        <h1>🛡️ AML Defense Dashboard</h1>
        <p>Secured access — only authorised personnel</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            login_user = st.text_input("Username", placeholder="Enter username")
            login_pass = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("🔐 Login", type="primary", use_container_width=True)
            if submitted:
                if login_user == _AUTH_USERNAME and login_pass == _AUTH_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    st.stop()


check_login()

# ---------------------------------------------------------------------------
# Custom CSS for a professional, clean look
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Main title */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A5F;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #1E3A5F;
        margin-bottom: 1.5rem;
    }
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2C5F8A;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
        padding-left: 0.5rem;
        border-left: 4px solid #2C5F8A;
    }
    /* Metric cards */
    .metric-card {
        background: #F0F4F8;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A5F;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6B7C93;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .good { color: #27AE60; }
    .medium { color: #F39C12; }
    .bad { color: #E74C3C; }
    /* Sidebar tweaks */
    .css-1d391kg { padding-top: 1rem; }
    .stButton button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
    }
    /* DataFrame */
    .dataframe { font-size: 0.85rem; }
    /* Expander */
    .streamlit-expanderHeader {
        font-size: 1rem;
        font-weight: 600;
        color: #2C5F8A;
    }
    hr { margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🛡️ AML Defense — Evasion Attack Analysis</div>',
            unsafe_allow_html=True)
st.caption("Train classifiers, simulate FGSM / PGD attacks, apply defenses, and compare robustness.")

# ---------------------------------------------------------------------------
# Helper: colour a metric based on its value
# ---------------------------------------------------------------------------

def _colour(val: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        if val >= 0.95: return "good"
        if val >= 0.70: return "medium"
        return "bad"
    if val <= 0.10: return "good"
    if val <= 0.40: return "medium"
    return "bad"

# ---------------------------------------------------------------------------
# Helper: compute all metrics for a model on given data
# ---------------------------------------------------------------------------

def _full_metrics(model, X, y, prefix=""):
    preds = model.predict(X)
    probs = model.predict_proba(X)
    return compute_metrics(y, preds, probs, prefix=prefix)

# ---------------------------------------------------------------------------
# Helper: confusion matrix figure
# ---------------------------------------------------------------------------

def _per_class_confusion(model, X, y):
    """Return a DataFrame with TP / TN / FP / FN per class."""
    preds = model.predict(X)
    cm = confusion_matrix(y, preds)
    n = cm.shape[0]
    classes = np.unique(y)
    rows = []
    for i in range(n):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rows.append({
            "Class": str(classes[i])[:40],
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Precision": prec, "Recall": rec,
        })
    return pd.DataFrame(rows)

# ===========================================================================
# SIDEBAR
# ===========================================================================

with st.sidebar:
    # Logout button
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()
    st.divider()

    st.markdown("## 🗂️ Data")

    if st.button("🔄 Refresh file list"):
        st.cache_data.clear()

    @st.cache_data
    def list_datasets():
        patterns = ["*.parquet", "*.csv", "*.txt", "*.zip", "*.gz"]
        files = []
        for p in patterns:
            files.extend(DATA_RAW_DIR.glob(p))
        return sorted(f.name for f in files)

    dataset_files = list_datasets()
    if not dataset_files:
        st.warning("No datasets found in `data/raw/`.", icon="⚠️")
        st.stop()

    # Default to the first training file if present
    default_idx = 0
    for i, f in enumerate(dataset_files):
        if "train" in f.lower():
            default_idx = i
            break
    selected_file = st.selectbox("Dataset file", dataset_files, index=default_idx)

    col1, col2 = st.columns(2)
    with col1:
        chunksize_input = st.number_input("Chunk size", min_value=1000,
                                           value=CHUNKSIZE, step=5000)
    with col2:
        max_rows_input = st.number_input("Max rows (0 = all)", min_value=0,
                                          value=MAX_ROWS, step=10000)

    load_btn = st.button("📥 Load & Process", type="primary", use_container_width=True)

    st.divider()
    st.markdown("## 🤖 Models")
    train_btn = st.button("🏋️ Train All Models", type="primary", use_container_width=True)
    load_pipeline_btn = st.button("📂 Load Pipeline Results", type="secondary", use_container_width=True)

    st.divider()
    st.markdown("## ⚡ Attacks")
    attack_type = st.selectbox("Attack type", ["fgsm", "pgd"])
    eps = st.slider("Epsilon", 0.0, 1.0, 0.1, 0.05)
    alpha = st.slider("Step size (alpha)", 0.001, 0.1, 0.01, 0.005)
    num_iter = st.slider("Iterations", 1, 50, 10)
    attack_btn = st.button("💥 Run Attack", type="secondary", use_container_width=True)

    st.divider()
    st.markdown("## 🛡️ Defenses")
    defense_type = st.selectbox("Defense type",
                                ["adversarial_training", "feature_selection",
                                 "ensemble", "feature_squeezing"])
    defense_model_choice = st.selectbox("Target model",
                                        ["random_forest", "xgboost", "lightgbm"])
    defense_btn = st.button("🔒 Apply Defense", type="secondary", use_container_width=True)

# ===========================================================================
# MAIN PANEL
# ===========================================================================

# ---- Step 1: Data Loading ----

if load_btn or "df" not in st.session_state:
    # Fast path: pre-processed data from a prior pipeline run
    import joblib as _jl
    preprocessed = MODELS_DIR / "X_train.pkl"
    if preprocessed.exists() and not load_btn:
        st.session_state["X_train"] = _jl.load(MODELS_DIR / "X_train.pkl")
        st.session_state["X_val"]   = _jl.load(MODELS_DIR / "X_val.pkl")
        st.session_state["X_test"]  = _jl.load(MODELS_DIR / "X_test.pkl")
        st.session_state["y_train"] = _jl.load(MODELS_DIR / "y_train.pkl")
        st.session_state["y_val"]   = _jl.load(MODELS_DIR / "y_val.pkl")
        st.session_state["y_test"]  = _jl.load(MODELS_DIR / "y_test.pkl")
        preview_path = MODELS_DIR / "df_preview.pkl"
        st.session_state["df"] = pd.read_pickle(preview_path) if preview_path.exists() else st.session_state["X_train"].head(100)
        st.session_state["data_loaded"] = True
        st.session_state["processor"] = DataProcessor() if (MODELS_DIR / "processor.pkl").exists() else None
        if st.session_state["processor"] is not None:
            st.session_state["processor"].load(MODELS_DIR / "processor.pkl")
    else:
        # Slow path: load and process raw CSV
        with st.spinner("Loading dataset..."):
            actual_max_rows = int(max_rows_input) if int(max_rows_input) != 0 else None
            df = load_data(DATA_RAW_DIR / selected_file, chunksize=int(chunksize_input), max_rows=actual_max_rows)
            df = clean_data(df)
            st.session_state["df"] = df
            st.session_state["data_loaded"] = True
            X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(df)
            # Apply processor scaling if a pre-trained processor exists
            # (models saved by the pipeline were trained on scaled data)
            proc_path = MODELS_DIR / "processor.pkl"
            if proc_path.exists():
                from src.preprocessing.processor import DataProcessor as _DP
                proc = _DP()
                proc.load(proc_path)
                X_tr, y_tr = proc.transform(X_tr, y_tr)
                X_va, y_va = proc.transform(X_va, y_va)
                X_te, y_te = proc.transform(X_te, y_te)
                st.session_state["processor"] = proc
            else:
                st.session_state["processor"] = None
            st.session_state["X_train"] = X_tr
            st.session_state["X_val"]   = X_va
            st.session_state["X_test"]  = X_te
            st.session_state["y_train"] = y_tr
            st.session_state["y_val"]   = y_va
            st.session_state["y_test"]  = y_te
        st.toast("Dataset loaded successfully!", icon="✅")

    # Auto-load pre-trained models from disk (demo / presentation mode)
    if "models" not in st.session_state:
        import json as _json
        model_names = ["random_forest", "xgboost", "lightgbm"]
        model_paths = [MODELS_DIR / f"{n}_clean.pkl" for n in model_names]
        if all(p.exists() for p in model_paths):
            models = {}
            for name in model_names:
                models[name] = load_model(MODELS_DIR / f"{name}_clean.pkl")
            st.session_state["models"] = models
            audit_log = Path("reports") / "pipeline_audit.json"
            if audit_log.exists():
                with open(audit_log) as _f:
                    st.session_state["pipeline_audit"] = _json.load(_f)

if not st.session_state.get("data_loaded"):
    st.info("👈 Select a dataset and click **Load & Process** to begin.")
    st.stop()

# ---- Dataset Overview ----

df = st.session_state["df"]
st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Rows", f"{df.shape[0]:,}")
c2.metric("Features", df.shape[1])
c3.metric("Train (72%)", f"{st.session_state['X_train'].shape[0]:,}")
c4.metric("Validation (8%)", f"{st.session_state['X_val'].shape[0]:,}")
c5.metric("Test (20%)", f"{st.session_state['X_test'].shape[0]:,}")

with st.expander("🔍 Data Preview", expanded=False):
    st.dataframe(df.head(100), use_container_width=True)

# ---- Pipeline Audit Trail ----
if st.session_state.get("pipeline_audit"):
    audit = st.session_state["pipeline_audit"]
    st.markdown("""
    <style>
        .audit-card {
            background: linear-gradient(135deg, #1E3A5F 0%, #2C5F8A 100%);
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin: 0.8rem 0;
            color: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .audit-card table { width: 100%; border-collapse: collapse; }
        .audit-card td { padding: 6px 12px; border-bottom: 1px solid rgba(255,255,255,0.1); }
        .audit-card td:first-child {
            font-weight: 600; color: #8AB4F8; white-space: nowrap; width: 100px;
        }
        .audit-card tr:last-child td { border-bottom: none; }
        .audit-title {
            font-size: 1rem; font-weight: 700; margin-bottom: 0.5rem;
            letter-spacing: 0.5px;
        }
    </style>
    <div class="audit-card">
        <div class="audit-title">📋 Pipeline Execution Report</div>
        <table>
            <tr><td>Started</td><td>""" + audit["start_time"] + """</td></tr>
            <tr><td>Finished</td><td>""" + audit["end_time"] + """</td></tr>
            <tr><td>Duration</td><td>""" + str(audit["duration_sec"]) + """s (""" + str(audit["duration_min"]) + """ min)</td></tr>
            <tr><td>Run by</td><td>""" + audit["user"] + """ @ """ + audit["host"] + """</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ---- Step 1b: Load Pipeline Results ----

if load_pipeline_btn:
    if not st.session_state.get("data_loaded"):
        st.error("Load data first.", icon="🚫")
    else:
        proc_path = MODELS_DIR / "processor.pkl"
        if not proc_path.exists():
            st.error(f"Processor not found at {proc_path}. Run the pipeline first.", icon="🚫")
        else:
            with st.spinner("Loading pipeline results..."):
                from src.preprocessing.processor import DataProcessor
                # Only transform if data hasn't been scaled yet
                if st.session_state.get("processor") is None:
                    proc = DataProcessor()
                    proc.load(proc_path)
                    X_train_p, y_train_p = proc.transform(st.session_state["X_train"], st.session_state["y_train"])
                    X_val_p, y_val_p = proc.transform(st.session_state["X_val"], st.session_state["y_val"])
                    X_test_p, y_test_p = proc.transform(st.session_state["X_test"], st.session_state["y_test"])
                    st.session_state["X_train"] = X_train_p
                    st.session_state["X_val"] = X_val_p
                    st.session_state["X_test"] = X_test_p
                    st.session_state["y_train"] = y_train_p
                    st.session_state["y_val"] = y_val_p
                    st.session_state["y_test"] = y_test_p
                    st.session_state["processor"] = proc

                models = {}
                for name in ["random_forest", "xgboost", "lightgbm"]:
                    model_path = MODELS_DIR / f"{name}_clean.pkl"
                    if model_path.exists():
                        models[name] = load_model(model_path)

                if models:
                    st.session_state["models"] = models
                    # Look for existing audit log in reports
                    import json as _json
                    audit_log = Path("reports") / "pipeline_audit.json"
                    if audit_log.exists():
                        with open(audit_log) as _f:
                            st.session_state["pipeline_audit"] = _json.load(_f)
                    st.toast(f"Loaded {len(models)} models from pipeline!", icon="✅")
                else:
                    st.error("No pre-trained models found. Run the pipeline first.", icon="🚫")

# ---- Step 2: Model Training ----

if train_btn:
    if not st.session_state.get("data_loaded"):
        st.error("Load data first.", icon="🚫")
    else:
        pipeline_start = time.time()
        run_start = datetime.datetime.now()
        run_user = getpass.getuser()
        run_host = os.environ.get("COMPUTERNAME", os.environ.get("HOSTNAME", "Unknown"))

        with st.spinner("Training models... (this may take a minute)"):
            models = train_all_models(
                st.session_state["X_train"],
                st.session_state["y_train"],
                st.session_state["X_val"],
                st.session_state["y_val"],
            )
            st.session_state["models"] = models

            # Fit processor only on raw data (not already-scaled data)
            if st.session_state.get("processor") is None:
                proc = DataProcessor()
                proc.fit(st.session_state["X_train"], st.session_state["y_train"])
                proc.save(MODELS_DIR / "processor.pkl")
                st.session_state["processor"] = proc

        run_end = datetime.datetime.now()
        elapsed = time.time() - pipeline_start

        st.session_state["pipeline_audit"] = {
            "start_time": run_start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": run_end.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": round(elapsed, 1),
            "duration_min": round(elapsed / 60, 1),
            "user": run_user,
            "host": run_host,
        }
        st.toast("All models trained!", icon="✅")

if st.session_state.get("models"):
    models = st.session_state["models"]

    st.markdown('<div class="section-header">🤖 Model Performance — Test Set</div>',
                unsafe_allow_html=True)

    # Compute full metrics for every model
    rows = []
    for name, model in models.items():
        m = _full_metrics(model, st.session_state["X_test"], st.session_state["y_test"])
        rows.append({
            "Model": name.replace("_", " ").title(),
            "Accuracy":  m["accuracy"],
            "Precision": m["precision"],
            "Recall":    m["recall"],
            "F1 Score":  m["f1_score"],
        })
        # Store in session for attack/defense steps
        st.session_state[f"metrics_{name}"] = m
    perf_df = pd.DataFrame(rows).set_index("Model")

    # Identify best model by F1
    best_model_name = perf_df["F1 Score"].idxmax()
    best_f1 = perf_df.loc[best_model_name, "F1 Score"]
    best_acc = perf_df.loc[best_model_name, "Accuracy"]
    best_prec = perf_df.loc[best_model_name, "Precision"]
    best_rec = perf_df.loc[best_model_name, "Recall"]

    # ---- Best model banner ----
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a6b3c,#27AE60);border-radius:10px;
                padding:0.8rem 1.5rem;margin:0.5rem 0 1rem 0;color:white;
                box-shadow:0 3px 10px rgba(39,174,96,0.3);">
        <span style="font-size:1.3rem;">🏆</span>
        <strong>Best Algorithm:</strong> {best_model_name}
        &nbsp;&nbsp;|&nbsp;&nbsp; F1 = {best_f1:.4f}
        &nbsp;&nbsp;|&nbsp;&nbsp; Accuracy = {best_acc:.4f}
        &nbsp;&nbsp;|&nbsp;&nbsp; Precision = {best_prec:.4f}
        &nbsp;&nbsp;|&nbsp;&nbsp; Recall = {best_rec:.4f}
    </div>
    """, unsafe_allow_html=True)

    # ---- Metric comparison chart (Plotly) ----
    try:
        import plotly.express as px
        melted = perf_df.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig = px.bar(
            melted,
            x="Model", y="Score", color="Metric", barmode="group",
            title="Model Comparison — All Metrics",
            color_discrete_sequence=px.colors.qualitative.Bold,
            text_auto=".3f",
        )
        fig.update_layout(
            yaxis_range=[0, 1.05],
            legend_title_text="",
            font=dict(size=12),
            title_font_size=16,
        )
        fig.update_traces(textposition="outside")
        # Add star marker on best model's bars
        best_idx = list(models.keys()).index(best_model_name.replace(" ", "_").lower())
        best_model_key = list(models.keys())[best_idx]
        # highlight the best model x-axis label
        fig.update_xaxes(
            tickangle=0,
            tickfont=dict(
                size=13 if best_model_key.replace("_", " ").title() == best_model_name else 12,
                color="#27AE60" if best_model_key.replace("_", " ").title() == best_model_name else "#333",
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.warning("Install plotly for interactive charts: `pip install plotly`")

    # ---- Data table: all models with all metrics ----
    st.markdown("#### Clean Performance of Base Classifiers on CIC_IoMT_2024")
    st.dataframe(
        perf_df.style
        .format("{:.4f}")
        .highlight_max(color="#27AE60", axis=0),
        use_container_width=True,
    )

    # ---- Metric cards (highlight best) ----
    cols = st.columns(len(models))
    for i, (name, model) in enumerate(models.items()):
        with cols[i]:
            m = st.session_state[f"metrics_{name}"]
            display_name = name.replace("_", " ").title()
            is_best = (display_name == best_model_name)
            if is_best:
                st.markdown(
                    f'<div style="border:2px solid #27AE60;border-radius:10px;padding:10px;'
                    f'background:#f0fdf4;">'
                    f'<div style="text-align:center;color:#1a6b3c;font-weight:700;">'
                    f'🏆 {display_name} ⬅ BEST</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f"**{display_name}**")
            for label, key in [("Accuracy", "accuracy"), ("Precision", "precision"),
                                ("Recall", "recall"), ("F1", "f1_score")]:
                val = m[key]
                cls = _colour(val)
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;padding:1px 0;">'
                    f'<span style="color:#6B7C93;">{label}</span>'
                    f'<span class="{cls}" style="font-weight:600;">{val:.4f}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            if is_best:
                st.markdown('</div>', unsafe_allow_html=True)

    # ---- Confusion: TP / TN / FP / FN per class ----
    st.markdown("#### Confusion Matrix — TP / TN / FP / FN per class")
    cm_tabs = st.tabs([n.replace("_", " ").title() for n in models.keys()])
    for i, (name, model) in enumerate(models.items()):
        with cm_tabs[i]:
            cm_df = _per_class_confusion(model, st.session_state["X_test"],
                                          st.session_state["y_test"])
            total_tp = cm_df["TP"].sum()
            total_fp = cm_df["FP"].sum()
            total_fn = cm_df["FN"].sum()
            total_tn = cm_df["TN"].iloc[0]  # same for all rows

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total TP", f"{total_tp:,}")
            c2.metric("Total FP", f"{total_fp:,}")
            c3.metric("Total FN", f"{total_fn:,}")
            c4.metric("Total TN", f"{total_tn:,}")

            st.markdown("**Worst 5 (most false negatives — missed detections)**")
            st.dataframe(
                cm_df.sort_values("FN", ascending=False).head(5)
                    .style.format({"Precision": "{:.0%}", "Recall": "{:.0%}"}),
                use_container_width=True,
            )
            st.markdown("**Best 5 (highest recall)**")
            st.dataframe(
                cm_df.sort_values("Recall", ascending=False).head(5)
                    .style.format({"Precision": "{:.0%}", "Recall": "{:.0%}"}),
                use_container_width=True,
            )
            with st.expander("📋 Full per-class breakdown"):
                st.dataframe(
                    cm_df.style.format({"Precision": "{:.0%}", "Recall": "{:.0%}"}),
                    use_container_width=True,
                )

    # ---- Donut charts per model (circular) ----
    st.markdown("#### Circular Metrics — Per Model")
    try:
        import plotly.graph_objects as go
        donut_cols = st.columns(len(models))
        for i, (name, model) in enumerate(models.items()):
            with donut_cols[i]:
                m = st.session_state[f"metrics_{name}"]
                labels = ["Accuracy", "Precision", "Recall", "F1"]
                values = [m["accuracy"], m["precision"], m["recall"], m["f1_score"]]
                colors = ["#2C5F8A", "#27AE60", "#F39C12", "#8E44AD"]
                fig_donut = go.Figure(data=[go.Pie(
                    labels=labels, values=values,
                    hole=0.5, marker=dict(colors=colors),
                    textinfo="label+percent", textfont=dict(size=11),
                )])
                fig_donut.update_layout(
                    title=name.replace("_", " ").title(),
                    height=300, margin=dict(t=30, b=0, l=0, r=0),
                    showlegend=False,
                )
                st.plotly_chart(fig_donut, use_container_width=True)
    except ImportError:
        pass

    # ---- Heatmap: metrics × models ----
    st.markdown("#### Heatmap — Metrics × Models")
    try:
        import plotly.express as px
        heatmap_df = perf_df.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig_heat = px.density_heatmap(
            heatmap_df, x="Metric", y="Model", z="Score",
            text_auto=".3f", range_color=[0, 1],
            color_continuous_scale="Blues",
            title="Metric Heatmap — Darker = Better",
        )
        fig_heat.update_layout(font=dict(size=12), xaxis_tickangle=0)
        st.plotly_chart(fig_heat, use_container_width=True)
    except ImportError:
        pass

    st.divider()

# ---- Step 3: Evasion Attacks ----

if attack_btn and st.session_state.get("models"):
    models = st.session_state["models"]
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]

    st.markdown(f'<div class="section-header">⚡ Attack Results — {attack_type.upper()} (eps={eps})</div>',
                unsafe_allow_html=True)

    attack_rows = []
    for name, model in models.items():
        with st.spinner(f"Attacking {name}..."):
            if attack_type == "fgsm":
                X_adv = fgsm_attack(model, X_test, y_test, epsilon=eps)
            else:
                X_adv = pgd_attack(model, X_test, y_test, epsilon=eps, alpha=alpha, num_iter=num_iter)

        clean_acc = model.score(X_test, y_test)
        adv_acc   = model.score(X_adv, y_test)
        asr       = attack_success_rate(y_test, model.predict(X_adv))

        # Full metrics on adversarial examples
        adv_m = compute_metrics(y_test, model.predict(X_adv), model.predict_proba(X_adv))

        attack_rows.append({
            "Model": name.replace("_", " ").title(),
            "Clean Acc": clean_acc,
            "Adv Acc":   adv_acc,
            "Adv Prec":  adv_m["precision"],
            "Adv Rec":   adv_m["recall"],
            "Adv F1":    adv_m["f1_score"],
            "ASR":       asr,
        })
        st.session_state[f"X_adv_{name}"] = X_adv

    att_df = pd.DataFrame(attack_rows).set_index("Model")

    # ---- Accuracy drop bar chart ----
    try:
        import plotly.graph_objects as go
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Clean Accuracy", x=att_df.index, y=att_df["Clean Acc"],
                               marker_color="#2C5F8A"))
        fig2.add_trace(go.Bar(name="Adversarial Accuracy", x=att_df.index, y=att_df["Adv Acc"],
                               marker_color="#E74C3C"))
        fig2.update_layout(
            barmode="group", title="Clean vs Adversarial Accuracy",
            yaxis_range=[0, 1.05], font=dict(size=12),
        )
        st.plotly_chart(fig2, use_container_width=True)
    except ImportError:
        st.bar_chart(att_df[["Clean Acc", "Adv Acc"]])

    # ---- ASR chart ----
    try:
        fig3 = px.bar(
            att_df.reset_index(), x="Model", y="ASR",
            title="Attack Success Rate",
            color="ASR", color_continuous_scale="Reds",
            text_auto=".3f",
        )
        fig3.update_layout(yaxis_range=[0, 1.05], font=dict(size=12))
        st.plotly_chart(fig3, use_container_width=True)
    except ImportError:
        st.bar_chart(att_df["ASR"])

    # ---- Detailed table ----
    with st.expander("📋 Attack Metrics (full table)", expanded=False):
        st.dataframe(att_df.style.format("{:.4f}"), use_container_width=True)

    st.divider()

# ---- Step 4: Defenses ----

if defense_btn and st.session_state.get("models"):
    models = st.session_state["models"]
    X_train, X_val = st.session_state["X_train"], st.session_state["X_val"]
    y_train, y_val = st.session_state["y_train"], st.session_state["y_val"]
    X_test, y_test = st.session_state["X_test"], st.session_state["y_test"]

    st.markdown(f'<div class="section-header">🛡️ Defense: {defense_type.replace("_", " ").title()}</div>',
                unsafe_allow_html=True)

    def_model = None

    if defense_type == "adversarial_training":
        with st.spinner(f"Adversarially training {defense_model_choice}..."):
            def_model = adversarial_training(defense_model_choice, X_train, y_train, X_val, y_val)
        st.success(f"Adversarially trained **{defense_model_choice}**")

    elif defense_type == "feature_selection":
        with st.spinner(f"Applying feature selection on {defense_model_choice}..."):
            def_model, kept = feature_selection_defense(defense_model_choice, X_train, y_train, X_val, y_val)
        st.session_state["feat_select_kept"] = kept
        st.success(f"Feature selection: kept **{len(kept)}** of {X_train.shape[1]} features")

    elif defense_type == "ensemble":
        with st.spinner("Training ensemble (RF + XGB + LGBM)..."):
            def_model = ensemble_defense(X_train, y_train, X_val, y_val)
        st.success("Ensemble trained with soft voting")

    elif defense_type == "feature_squeezing":
        target_model = models.get(defense_model_choice)
        if target_model:
            with st.spinner(f"Applying feature squeezing to {defense_model_choice}..."):
                X_sqz, acc = feature_squeezing_defense(target_model, X_test, y_test)
            st.metric("Accuracy after squeezing", f"{acc:.4f}")
            def_model = target_model

    st.session_state["defended_model"] = def_model

    # ---- Compare defended vs undefended ----
    if def_model is not None and defense_type != "feature_squeezing":
        st.markdown("#### 🔍 Defended vs Undefended — Test Set")
        original = models.get(defense_model_choice) or models.get("random_forest")
        if original:
            X_test_compare = X_test
            if defense_type == "feature_selection":
                kept = st.session_state.get("feat_select_kept")
                if kept is not None:
                    X_test_compare = X_test[kept]
            orig_m = _full_metrics(original, X_test, y_test)
            def_m  = _full_metrics(def_model, X_test_compare, y_test)
            comp_rows = []
            for met in ["accuracy", "precision", "recall", "f1_score"]:
                comp_rows.append({
                    "Metric": met.replace("_", " ").title(),
                    "Undefended": orig_m[met],
                    "Defended":   def_m[met],
                })
            comp_df = pd.DataFrame(comp_rows).set_index("Metric")
            try:
                fig4 = px.bar(
                    comp_df.reset_index().melt(id_vars="Metric", var_name="Model", value_name="Score"),
                    x="Metric", y="Score", color="Model", barmode="group",
                    title="Defense Effectiveness",
                    color_discrete_sequence=["#2C5F8A", "#27AE60"],
                    text_auto=".3f",
                )
                fig4.update_layout(yaxis_range=[0, 1.05], font=dict(size=12))
                st.plotly_chart(fig4, use_container_width=True)
            except ImportError:
                st.dataframe(comp_df.style.format("{:.4f}"), use_container_width=True)

    st.divider()

# ===========================================================================
# Step 5: Real-Time Live Inference Demo
# ===========================================================================

st.markdown('<div class="section-header">⚡ Live IoMT Inference Demo</div>',
            unsafe_allow_html=True)
st.caption("Simulates real-time prediction on streaming test samples. Each row is one inference call.")

if not st.session_state.get("models"):
    st.info("Train models first to use the live demo.")
else:
    models = st.session_state["models"]

    # Initialise session state for live demo
    if "live_history" not in st.session_state:
        st.session_state.live_history = []
        st.session_state.live_index = 0
        st.session_state.live_attack = False
        st.session_state.live_epsilon = 0.1
        st.session_state.live_start_time = None

    # Controls row
    cc1, cc2, cc3, cc4, cc5 = st.columns([1.5, 1, 1, 1, 1.2])
    with cc1:
        live_model = st.selectbox("Model", list(models.keys()),
                                  format_func=lambda x: x.replace("_", " ").title(),
                                  key="live_model_select")
    with cc2:
        live_attack_type = st.selectbox("Attack", ["None", "FGSM", "PGD"],
                                        index=0 if not st.session_state.live_attack else 1,
                                        key="live_attack_select")
    with cc3:
        live_eps = st.number_input("Epsilon", 0.0, 1.0,
                                    value=st.session_state.live_epsilon, step=0.05,
                                    key="live_eps_input")
    with cc4:
        live_delay = st.number_input("Delay (s)", 0.0, 2.0, 0.05, 0.01,
                                     key="live_delay_input")

    # Update session state from widgets
    st.session_state.live_attack = live_attack_type != "None"
    st.session_state.live_epsilon = live_eps

    col_step, col_reset, col_clear = st.columns([1, 1, 1.5])
    with col_step:
        step_clicked = st.button("▶ Next Sample", type="primary", use_container_width=True)
    with col_reset:
        reset_clicked = st.button("⟳ Reset Stream", use_container_width=True)
    with col_clear:
        n_history = st.selectbox("Show last N", [10, 20, 50, 100], index=1, key="live_n_history")

    if reset_clicked:
        st.session_state.live_history = []
        st.session_state.live_index = 0
        st.session_state.live_start_time = None

    # Step: fetch one sample and record
    if step_clicked:
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        model = models[live_model]
        idx = st.session_state.live_index % X_test.shape[0]
        sample = X_test.iloc[idx]
        true_label = int(y_test.iloc[idx])

        # Attack perturbation
        if live_attack_type != "None" and live_eps > 0:
            df = pd.DataFrame([sample])
            if live_attack_type == "PGD":
                from src.attacks.pgd import pgd_attack
                df_adv = pgd_attack(model, df, pd.Series([true_label]), epsilon=live_eps)
            else:
                df_adv = fgsm_attack(model, df, pd.Series([true_label]), epsilon=live_eps)
            sample = df_adv.iloc[0]

        # Predict
        t0 = time.time()
        df_in = pd.DataFrame([sample])
        pred = int(model.predict(df_in)[0])
        prob = float(np.max(model.predict_proba(df_in)[0]))
        latency_ms = (time.time() - t0) * 1000
        correct = pred == true_label

        if st.session_state.live_start_time is None:
            st.session_state.live_start_time = time.time()

        entry = {
            "time": datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "idx": idx,
            "true": true_label,
            "pred": pred,
            "conf": prob,
            "correct": "✅" if correct else "❌",
            "latency_ms": round(latency_ms, 2),
            "attack": live_attack_type if live_attack_type != "None" else "clean",
        }
        st.session_state.live_history.append(entry)
        st.session_state.live_index += 1

    # ---- Display live feed ----
    history = st.session_state.live_history
    if not history:
        st.info("Click **Next Sample** to start the live inference stream.")
    else:
        # Metrics summary row
        total = len(history)
        n_correct = sum(1 for h in history if h["correct"] == "✅")
        running_acc = n_correct / total if total > 0 else 0.0
        avg_latency = sum(h["latency_ms"] for h in history[-50:]) / min(len(history[-50:]), 1)
        n_attacked = sum(1 for h in history if h["attack"] == "FGSM")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Samples Processed", total)
        m2.metric("Running Accuracy", f"{running_acc:.2%}")
        m3.metric("Avg Latency (ms)", f"{avg_latency:.1f}")
        m4.metric("FGSM Samples", n_attacked)
        m5.metric("Correct / Total", f"{n_correct} / {total}")

        # Bar: running accuracy trend (last 100 in groups of 10)
        if total >= 10:
            try:
                import plotly.express as px
                window = min(total, 100)
                recent = history[-window:]
                group_size = max(1, window // 10)
                groups = []
                for g in range(0, window, group_size):
                    batch = recent[g:g+group_size]
                    acc = sum(1 for h in batch if h["correct"] == "✅") / len(batch)
                    groups.append({"Batch": f"{g+1}-{min(g+group_size, window)}",
                                   "Accuracy": acc})
                dfg = pd.DataFrame(groups)
                fig_live = px.bar(dfg, x="Batch", y="Accuracy",
                                   title="Running Accuracy (last 100 samples, batched)",
                                   range_y=[0, 1], color="Accuracy",
                                   color_continuous_scale="RdYlGn",
                                   text_auto=".0%")
                fig_live.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(fig_live, use_container_width=True)
            except ImportError:
                pass

        # Latest live feed table
        n_show = min(len(history), int(n_history))
        display = list(reversed(history[-n_show:]))
        df_live = pd.DataFrame(display)
        st.markdown("#### Latest Predictions")
        st.dataframe(
            df_live.rename(columns={"conf": "Confidence"})
            .style
            .map(lambda v: "color: green" if v == "✅" else "color: red", subset=["correct"])
            .format({"conf": "{:.2%}", "latency_ms": "{:.1f}ms"}),
            use_container_width=True,
            height=min(60 + n_show * 35, 450),
        )

    st.divider()

# ===========================================================================
# Summary / empty state
# ===========================================================================

if not st.session_state.get("models"):
    st.info("👈 Train models in the sidebar to see performance metrics, attacks, and defenses.")

st.markdown("---")
st.caption("AML Defense Dashboard • Built with Streamlit • sklearn • XGBoost • LightGBM • SHAP (optional)")
st.caption("Real-time live inference: click Next Sample to simulate streaming IoMT predictions.")
