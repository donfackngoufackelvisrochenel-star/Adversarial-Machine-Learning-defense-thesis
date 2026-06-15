"""Test script to reproduce dashboard errors."""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import DATA_RAW_DIR, MODELS_DIR, MAX_ROWS
from src.preprocessing.loader import load_data, clean_data, split_data
from src.models.trainer import train_all_models
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.defenses.feature_selection import feature_selection_defense
from src.defenses.adversarial_training import adversarial_training
from src.defenses.ensemble import ensemble_defense
from src.defenses.feature_squeezing import feature_squeezing_defense
from sklearn.metrics import accuracy_score


def main():
    # Load data same as dashboard
    print("=" * 60)
    print("1. Loading data...")
    df = load_data(DATA_RAW_DIR / "CIC_IoMT_2024.zip", chunksize=5000, max_rows=MAX_ROWS)
    df = clean_data(df)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(df)
    print(f"Features: {X_tr.shape[1]}")

    # Train models
    print("\n" + "=" * 60)
    print("2. Training models...")
    models = train_all_models(X_tr, y_tr, X_va, y_va)
    print(f"Trained models: {list(models.keys())}")

    # Test FGSM attack
    print("\n" + "=" * 60)
    print("3. Testing FGSM Attack...")
    try:
        X_adv_fgsm = fgsm_attack(models["xgboost"], X_te, y_te, epsilon=0.1)
        fgsm_asr = (models["xgboost"].predict(X_adv_fgsm) != y_te).mean()
        print(f"FGSM success rate: {fgsm_asr:.4f}")
    except Exception as e:
        print(f"ERROR in FGSM: {type(e).__name__}: {e}")

    # Test PGD attack
    print("\n" + "=" * 60)
    print("4. Testing PGD Attack...")
    try:
        X_adv_pgd = pgd_attack(models["xgboost"], X_te, y_te, epsilon=0.1, alpha=0.01, num_iter=10)
        pgd_asr = (models["xgboost"].predict(X_adv_pgd) != y_te).mean()
        print(f"PGD success rate: {pgd_asr:.4f}")
    except Exception as e:
        print(f"ERROR in PGD: {type(e).__name__}: {e}")

    # Test feature_selection defense
    print("\n" + "=" * 60)
    print("5. Testing Feature Selection Defense...")
    try:
        def_model_fs, kept = feature_selection_defense("xgboost", X_tr, y_tr, X_va, y_va)
        print(f"Kept {len(kept)} features")
        # Try evaluating on full X_te (like dashboard does at line 515)
        acc = accuracy_score(y_te, def_model_fs.predict(X_te))
        print(f"Accuracy on full X_te: {acc:.4f}")
    except Exception as e:
        print(f"ERROR in feature_selection: {type(e).__name__}: {e}")

    # Test adversarial_training defense
    print("\n" + "=" * 60)
    print("6. Testing Adversarial Training Defense...")
    try:
        def_model_at = adversarial_training("xgboost", X_tr, y_tr, X_va, y_va)
        acc = accuracy_score(y_te, def_model_at.predict(X_te))
        print(f"Accuracy on X_te: {acc:.4f}")
    except Exception as e:
        print(f"ERROR in adversarial_training: {type(e).__name__}: {e}")

    # Test ensemble defense
    print("\n" + "=" * 60)
    print("7. Testing Ensemble Defense...")
    try:
        def_model_ens = ensemble_defense(X_tr, y_tr, X_va, y_va)
        acc = accuracy_score(y_te, def_model_ens.predict(X_te))
        print(f"Accuracy on X_te: {acc:.4f}")
    except Exception as e:
        print(f"ERROR in ensemble: {type(e).__name__}: {e}")

    # Test feature squeezing defense
    print("\n" + "=" * 60)
    print("8. Testing Feature Squeezing Defense...")
    try:
        X_sqz, acc = feature_squeezing_defense(models["xgboost"], X_te, y_te)
        print(f"Accuracy after squeezing: {acc:.4f}")
    except Exception as e:
        print(f"ERROR in feature_squeezing: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("All tests completed!")


if __name__ == "__main__":
    main()
