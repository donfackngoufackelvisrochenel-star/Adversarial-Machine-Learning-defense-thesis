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
    python scripts/run_pipeline.py                                  # full pipeline
    python scripts/run_pipeline.py --max-rows 1000                  # quick demo (first 1000 rows)
    python scripts/run_pipeline.py --skip-attack                    # skip attack step
    python scripts/run_pipeline.py --skip-defense                   # skip defense step
    python scripts/run_pipeline.py --skip-explain                   # skip SHAP step
    python scripts/run_pipeline.py --chunksize 5000                 # custom chunk size

Place your dataset (CSV, TXT, ZIP, or GZ) in data/raw/ before running.
"""

import argparse
import sys
import warnings
import datetime
import getpass
import os
import time
from pathlib import Path

# Suppress noisy but harmless sklearn/joblib warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

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
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from src.evaluation.metrics import compute_metrics, attack_success_rate, summarize_defense_comparison
from src.evaluation.comparison import compare_attack_defense, plot_robustness_comparison, generate_report_table
from src.explainability.explainer import explain_model
from src.utils.helpers import setup_logger, save_results
from configs.config import MODELS_DIR, REPORTS_DIR, MAX_ROWS as CONFIG_MAX_ROWS


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
    parser.add_argument("--max-rows", type=int, default=CONFIG_MAX_ROWS, help="Max rows to load (0 = all rows, e.g. --max-rows 50000)")
    args = parser.parse_args()

    logger = setup_logger("pipeline")

    now = datetime.datetime.now()
    user = getpass.getuser()
    host = os.environ.get("COMPUTERNAME", "Unknown")
    print("\n" + "=" * 70)
    print("  ML Defense Pipeline — Evasion Attack Analysis")
    print("=" * 70)
    print(f"  Started  : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  User     : {user}")
    print(f"  Host     : {host}")
    print(f"  Dataset  : CIC_IoMT_2024 (auto-detected)")
    print(f"  Max rows : {args.max_rows if args.max_rows != 0 else 'ALL'}")
    print("=" * 70)
    pipeline_start = time.time()

    def audit(msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] {msg}")

    # =====================================================================
    # Step 1: Load & Process Data
    # =====================================================================
    t1 = time.time()
    print(f"\n{'='*70}\n  STEP 1/7: LOAD & PROCESS DATA\n{'='*70}")
    actual_max_rows = args.max_rows if args.max_rows != 0 else None
    df = load_data(chunksize=args.chunksize, max_rows=actual_max_rows)
    if actual_max_rows is not None and len(df) > actual_max_rows:
        df = df.head(actual_max_rows).reset_index(drop=True)
        print(f"[pipeline] Trimmed to {actual_max_rows} rows for quick demo")
    df = clean_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    processor = DataProcessor()
    X_train, y_train = processor.fit_transform(X_train, y_train)
    X_val, y_val = processor.transform(X_val, y_val)
    X_test, y_test = processor.transform(X_test, y_test)
    processor.save(MODELS_DIR / "processor.pkl")
    audit(f"Step 1 done  | {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test rows | {time.time()-t1:.1f}s")

    # =====================================================================
    # Step 2: Train Models
    # =====================================================================
    t1 = time.time()
    print(f"\n{'='*70}\n  STEP 2/7: TRAIN MODELS\n{'='*70}")
    clean_models = train_all_models(X_train, y_train, X_val, y_val)
    audit(f"Step 2 done  | {len(clean_models)} models trained | {time.time()-t1:.1f}s")

    # =====================================================================
    # Step 3: Evaluate Clean (undefended) Performance
    # =====================================================================
    t1 = time.time()
    print(f"\n{'='*70}\n  STEP 3/7: EVALUATE CLEAN PERFORMANCE\n{'='*70}")
    import pandas as _pd
    clean_results = {}
    for name, model in clean_models.items():
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, preds, probs, prefix=name)
        clean_results[name] = metrics

        # ---- Per-class TP / TN / FP / FN ----
        cm = sk_confusion_matrix(y_test, preds)
        n = cm.shape[0]
        classes = y_test.unique()
        rows = []
        for i in range(n):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            rows.append({
                "class": str(classes[i])[:40],
                "TP": tp, "TN": tn, "FP": fp, "FN": fn,
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            })
        cm_df = _pd.DataFrame(rows)
        cm_df = cm_df.sort_values("FN", ascending=False)

        print(f"\n  === Confusion: {name} ===")
        print(f"  TP={cm.diagonal().sum()}  FP={cm.sum()-cm.diagonal().sum()}")
        # Show worst 5 classes by FN (false negatives = missed detections)
        print(f"  Worst 5 (most misses):")
        for _, r in cm_df.head(5).iterrows():
            print(f"    {r['class']:<35s} TP={r['TP']:>3d} TN={r['TN']:>4d} FP={r['FP']:>3d} FN={r['FN']:>3d}  prec={r['precision']:.0%} rec={r['recall']:.0%}")
        # Show best 5 classes by recall
        print(f"  Best 5 (most accurate):")
        for _, r in cm_df.sort_values("recall", ascending=False).head(5).iterrows():
            print(f"    {r['class']:<35s} TP={r['TP']:>3d} TN={r['TN']:>4d} FP={r['FP']:>3d} FN={r['FN']:>3d}  prec={r['precision']:.0%} rec={r['recall']:.0%}")
        # Save full per-class metrics to CSV
        cm_path = REPORTS_DIR / f"confusion_{name}.csv"
        cm_df.to_csv(cm_path, index=False)
        print(f"  Full per-class TP/TN/FP/FN saved: {cm_path}")
        print(f"  acc={metrics[f'{name}_accuracy']:.4f}, f1={metrics[f'{name}_f1_score']:.4f}\n")
    save_results(clean_results, "clean_metrics")
    audit(f"Step 3 done  | {time.time()-t1:.1f}s")

    # =====================================================================
    # Step 4: Evasion Attacks (FGSM + PGD)
    # =====================================================================
    defended_models = {}
    if not args.skip_attack:
        t1 = time.time()
        print(f"\n{'='*70}\n  STEP 4/7: EVASION ATTACKS\n{'='*70}")
        for eps in [0.05, 0.1, 0.2]:
            for name, model in clean_models.items():
                X_adv = fgsm_attack(model, X_test, y_test, epsilon=eps)
                asr = attack_success_rate(y_test, model.predict(X_adv))
                print(f"  FGSM(eps={eps}) on {name}: ASR={asr:.4f}")

                X_adv = pgd_attack(model, X_test, y_test, epsilon=eps)
                asr = attack_success_rate(y_test, model.predict(X_adv))
                print(f"  PGD(eps={eps}) on {name}: ASR={asr:.4f}")
        audit(f"Step 4 done  | {time.time()-t1:.1f}s")

    # =====================================================================
    # Step 5: Defenses
    # =====================================================================
    if not args.skip_defense:
        t1 = time.time()
        print(f"\n{'='*70}\n  STEP 5/7: DEFENSES\n{'='*70}")

        print("\n  --- Adversarial Training ---")
        for m in ["random_forest"]:
            adv_model = adversarial_training(m, X_train, y_train, X_val, y_val)
            defended_models[f"adv_train_{m}"] = adv_model

        print("\n  --- Feature Selection Defense ---")
        fs_model, kept = feature_selection_defense("xgboost", X_train, y_train, X_val, y_val)
        defended_models["feat_select_xgb"] = fs_model

        print("\n  --- Ensemble Defense ---")
        ensemble = ensemble_defense(X_train, y_train, X_val, y_val)
        defended_models["ensemble"] = ensemble

        print("\n  --- Feature Squeezing ---")
        for name, model in clean_models.items():
            X_sqz, acc = feature_squeezing_defense(model, X_test, y_test)
            print(f"  {name} after squeeze: acc={acc:.4f}")
        audit(f"Step 5 done  | {time.time()-t1:.1f}s")

    # =====================================================================
    # Step 6: Robustness Comparison & Reports
    # =====================================================================
    t1 = time.time()
    print(f"\n{'='*70}\n  STEP 6/7: REPORT GENERATION\n{'='*70}")
    if defended_models:
        for attack_name, attack_fn in [("FGSM", fgsm_attack), ("PGD", pgd_attack)]:
            comp_df = compare_attack_defense(
                clean_models, defended_models, X_test, y_test, attack_fn, attack_name
            )
            if comp_df is not None and not comp_df.empty:
                plot_robustness_comparison(comp_df, attack_name)
                generate_report_table(comp_df, attack_name)
    audit(f"Step 6 done  | {time.time()-t1:.1f}s")

    # =====================================================================
    # Step 7: Explainability (SHAP)
    # =====================================================================
    if not args.skip_explain:
        t1 = time.time()
        print(f"\n{'='*70}\n  STEP 7/7: SHAP EXPLANATIONS\n{'='*70}")
        for name, model in clean_models.items():
            explain_model(model, X_test.head(100), name)
        audit(f"Step 7 done  | {time.time()-t1:.1f}s")

    # =====================================================================
    # Done — Best algorithm summary
    # =====================================================================
    print("\n" + "=" * 60)
    print("  *** BEST ALGORITHM SUMMARY ***")
    print("=" * 60)

    # Best clean accuracy
    best_clean = max(clean_results.items(), key=lambda x: x[1].get(f"{x[0]}_accuracy", 0))
    best_name = best_clean[0]
    best_acc = best_clean[1].get(f"{best_name}_accuracy", 0)
    print(f"  Clean leader:     {best_name}  (acc={best_acc:.4f})")

    # Best defended model (average across epsilon from comparison tables)
    if defended_models:
        for attack_name in ["FGSM", "PGD"]:
            csv_path = REPORTS_DIR / f"report_{attack_name}.csv"
            if csv_path.exists():
                import pandas as pd
                report = pd.read_csv(csv_path, index_col=0)
                # Find defense with highest average accuracy across epsilons
                avg_acc = report.mean(axis=1)
                best_def = avg_acc.idxmax()
                print(f"  {attack_name} defense:    {best_def}  (avg acc={avg_acc.max():.4f})")

    elapsed = time.time() - pipeline_start
    now_end = datetime.datetime.now()
    # Save audit trail for the dashboard
    audit_data = {
        "start_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": now_end.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": round(elapsed, 1),
        "duration_min": round(elapsed / 60, 1),
        "user": user,
        "host": host,
    }
    import json as _json
    with open(REPORTS_DIR / "pipeline_audit.json", "w") as _f:
        _json.dump(audit_data, _f, indent=2)
    print("=" * 70)
    print("  Pipeline complete! Check reports/ for outputs.")
    print(f"  Finished : {now_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Run by   : {user}@{host}")
    print(f"  Audit    : reports/pipeline_audit.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
