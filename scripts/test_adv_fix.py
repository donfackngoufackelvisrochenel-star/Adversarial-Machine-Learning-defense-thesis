"""Test adversarial training fix."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import DATA_RAW_DIR, MAX_ROWS
from src.preprocessing.loader import load_data, clean_data, split_data
from src.models.trainer import train_all_models
from src.defenses.adversarial_training import adversarial_training
from sklearn.metrics import accuracy_score

df = load_data(DATA_RAW_DIR / "CIC_IoMT_2024.zip", chunksize=5000, max_rows=MAX_ROWS)
df = clean_data(df)
X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(df)
models = train_all_models(X_tr, y_tr, X_va, y_va)

print("Testing adversarial_training with xgboost...")
try:
    def_model = adversarial_training("xgboost", X_tr, y_tr, X_va, y_va)
    acc = accuracy_score(y_te, def_model.predict(X_te))
    print(f"OK - XGBoost accuracy: {acc:.4f}")
except Exception as e:
    print(f"ERROR in xgboost: {type(e).__name__}: {e}")

print("Testing adversarial_training with random_forest...")
try:
    def_model = adversarial_training("random_forest", X_tr, y_tr, X_va, y_va)
    acc = accuracy_score(y_te, def_model.predict(X_te))
    print(f"OK - RF accuracy: {acc:.4f}")
except Exception as e:
    print(f"ERROR in random_forest: {type(e).__name__}: {e}")
