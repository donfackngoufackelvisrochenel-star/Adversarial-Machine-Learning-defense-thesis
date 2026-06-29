#!/usr/bin/env python
"""
End-to-end AML Defense Pipeline.

Orchestrates the full workflow:
  1. Load & preprocess data (with chunked reading for memory efficiency)
  2. Train all three models (RF, XGBoost, LightGBM)
  3. Evaluate clean performance
  4. Run FGSM and PGD evasion attacks
  5. Apply defense strategies (adversarial training, feature selection,
     ensemble, feature squeezing)
  6. Generate robustness comparison plots and report tables
  7. Generate SHAP explanations for model interpretability

Usage:
    python scripts/run_pipeline.py                     # full pipeline
    python scripts/run_pipeline.py --skip-attack       # skip attack step
    python scripts/run_pipeline.py --skip-defense      # skip defense step
    python scripts/run_pipeline.py --skip-explain      # skip SHAP step
    python scripts/run_pipeline.py --chunksize 5000    # custom chunk size

Place your dataset (CSV, TXT, ZIP, or GZ) in data/raw/ before running.
"""

import argparse
import sys
import warnings
import datetime
import getpass
import os
import json
import time as _time
from pathlib import Path

# Suppress the harmless sklearn "X does not have valid feature names" warning
# that occurs when models are fitted on DataFrames and evaluated on arrays.
warnings.filterwarnings("ignore", message="X does not have valid feature names")
# Suppress sklearn 1.6+ internal parallel API warnings (cosmetic only)
warnings.filterwarnings("ignore", message="`sklearn.utils.parallel.delayed` should be used")

# Ensure the project root is on sys.path so we can import src.* modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.loader import load_data, clean_data, split_data
from src.preprocessing.processor import DataProcessor
from src.models.trainer import train_all_models, load_model, save_model
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.defenses.adversarial_training import adversarial_training
from src.defenses.feature_selection import feature_selection_defense
from src.defenses.ensemble import ensemble_defense
from src.defenses.feature_squeezing import feature_squeezing_defense
from src.evaluation.metrics import compute_metrics, attack_success_rate, summarize_defense_comparison
from src.evaluation.comparison import compare_attack_defense, plot_robustness_comparison, generate_report_table
from src.explainability.explainer import explain_model
from src.utils.helpers import setup_logger, save_results
from configs.config import MODELS_DIR, MAX_ROWS as CONFIG_MAX_ROWS


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _step_timer(label: str):
    print(f"\n  >>> {label} ...")
    return _time.time()

def _step_done(label: str, start: float):
    elapsed = _time.time() - start
    print(f"  <<< {label} done  [{elapsed:.1f}s]")
    return elapsed

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main():
    # ---- Parse command-line arguments ----
    parser = argparse.ArgumentParser(description="AML Defense Pipeline")
    parser.add_argument("--skip-attack", action="store_true", help="Skip attack simulation")
    parser.add_argument("--skip-defense", action="store_true", help="Skip defense application")
    parser.add_argument("--skip-explain", action="store_true", help="Skip SHAP explanation")
    parser.add_argument("--chunksize", type=int, default=None, help="Rows per chunk for memory-efficient loading")
    parser.add_argument("--max-rows", type=int, default=CONFIG_MAX_ROWS, help="Maximum rows to load from dataset (0 = all)")
    args = parser.parse_args()

    logger = setup_logger("pipeline")

    run_user = getpass.getuser()
    run_host = os.environ.get("COMPUTERNAME", os.environ.get("HOSTNAME", "Unknown"))
    run_start = datetime.datetime.now()

    print("\n" + "=" * 60)
    print("  ML Defense Pipeline — Evasion Attack Analysis")
    print("=" * 60)
    print(f"  Started : {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  User    : {run_user} @ {run_host}")
    actual_max_rows = args.max_rows if args.max_rows != 0 else None
    print(f"  Max rows: {actual_max_rows if actual_max_rows else 'ALL'}")
    print(f"  Chunks  : {args.chunksize if args.chunksize else 'default'}")
    print("=" * 60)

    # =====================================================================
    # Step 1: Load & Process Data
    # =====================================================================
    t1 = _step_timer("Step 1/7  Load & Process Data")
    df = load_data(chunksize=args.chunksize, max_rows=actual_max_rows)
    df = clean_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    processor = DataProcessor()
    X_train, y_train = processor.fit_transform(X_train, y_train)
    X_val, y_val = processor.transform(X_val, y_val)
    X_test, y_test = processor.transform(X_test, y_test)
    processor.save(MODELS_DIR / "processor.pkl")
    _step_done("Step 1/7", t1)

    # =====================================================================
    # Step 2: Train Models
    # =====================================================================
    t2 = _step_timer("Step 2/7  Train Models")
    clean_models = train_all_models(X_train, y_train, X_val, y_val)
    _step_done("Step 2/7", t2)

    # =====================================================================
    # Step 3: Evaluate Clean (undefended) Performance
    # =====================================================================
    t3 = _step_timer("Step 3/7  Evaluate Clean Models")
    clean_results = {}
    for name, model in clean_models.items():
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, preds, probs, prefix=name)
        clean_results[name] = metrics
        print(f"  {name}: acc={metrics[f'{name}_accuracy']:.4f}, f1={metrics[f'{name}_f1_score']:.4f}")
    save_results(clean_results, "clean_metrics")
    _step_done("Step 3/7", t3)

    # =====================================================================
    # Step 4: Evasion Attacks (FGSM + PGD)
    # =====================================================================
    defended_models = {}
    if not args.skip_attack:
        t4 = _step_timer("Step 4/7  Evasion Attacks")
        for eps in [0.05, 0.1, 0.2]:
            for name, model in clean_models.items():
                fgsm_attack(model, X_test, y_test, epsilon=eps)
                pgd_attack(model, X_test, y_test, epsilon=eps)
        _step_done("Step 4/7", t4)

    # =====================================================================
    # Step 5: Defenses
    # =====================================================================
    if not args.skip_defense:
        t5 = _step_timer("Step 5/7  Defenses")

        # 5a. Adversarial Training (only on RF for speed)
        print("\n  --- Adversarial Training ---")
        for m in ["random_forest"]:
            adv_model = adversarial_training(m, X_train, y_train, X_val, y_val)
            defended_models[f"adv_train_{m}"] = adv_model

        # 5b. Feature Selection (keep only K least vulnerable features)
        print("\n  --- Feature Selection Defense ---")
        fs_model, kept = feature_selection_defense("xgboost", X_train, y_train, X_val, y_val)
        defended_models["feat_select_xgb"] = fs_model

        # 5c. Ensemble (RF + XGBoost + LightGBM with soft voting)
        print("\n  --- Ensemble Defense ---")
        ensemble = ensemble_defense(X_train, y_train, X_val, y_val)
        defended_models["ensemble"] = ensemble

        # 5d. Feature Squeezing (bit-depth reduction)
        print("\n  --- Feature Squeezing ---")
        for name, model in clean_models.items():
            X_sqz, acc = feature_squeezing_defense(model, X_test, y_test)
            print(f"  {name} after squeeze: acc={acc:.4f}")

        _step_done("Step 5/7", t5)

    # =====================================================================
    # Step 6: Robustness Comparison & Reports
    # =====================================================================
    t6 = _step_timer("Step 6/7  Robustness Comparison")
    if defended_models:
        for attack_name, attack_fn in [("FGSM", fgsm_attack), ("PGD", pgd_attack)]:
            comp_df = compare_attack_defense(
                clean_models, defended_models, X_test, y_test, attack_fn, attack_name
            )
            if comp_df is not None and not comp_df.empty:
                plot_robustness_comparison(comp_df, attack_name)
                generate_report_table(comp_df, attack_name)
    _step_done("Step 6/7", t6)

    # =====================================================================
    # Step 7: Explainability (SHAP)
    # =====================================================================
    if not args.skip_explain:
        t7 = _step_timer("Step 7/7  SHAP Explanations")
        for name, model in clean_models.items():
            explain_model(model, X_test.head(100), name)
        _step_done("Step 7/7", t7)

    # =====================================================================
    # Summary — Best Algorithm (like dashboard)
    # =====================================================================
    print("\n" + "=" * 60)
    print("  *** BEST ALGORITHM SUMMARY ***")
    print("=" * 60)
    best = max(clean_results.items(), key=lambda x: x[1].get(f"{x[0]}_f1_score", 0))
    best_name = best[0]
    print(f"  Best model: {best_name}")
    for k in ["accuracy", "precision", "recall", "f1_score"]:
        val = best[1].get(f"{best_name}_{k}", 0)
        print(f"    {k.replace('_', ' ').title():12s} = {val:.4f}")

    # =====================================================================
    # Done
    # =====================================================================
    run_end = datetime.datetime.now()
    elapsed = (run_end - run_start).total_seconds()

    # Save pipeline audit trail
    audit = {
        "start_time": run_start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": run_end.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": round(elapsed, 1),
        "duration_min": round(elapsed / 60, 1),
        "user": run_user,
        "host": run_host,
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/pipeline_audit.json", "w") as _f:
        json.dump(audit, _f, indent=2)

    print("\n" + "=" * 60)
    print("  Pipeline complete! Check reports/ for outputs.")
    print(f"  Finished: {run_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration: {audit['duration_min']} min ({audit['duration_sec']} s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
