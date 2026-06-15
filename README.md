# AML Defense — Evasion Attack Analysis on IoMT Data

**Topic:** Implementation and Evaluation of Defense Strategies Against Evasion Attacks in Adversarial Machine Learning on IoMT Data

## Project Structure

```
final_thesis/
├── configs/config.py               # Central configuration
├── data/
│   ├── raw/                        # Place your CSV dataset here
│   └── processed/                  # Auto-generated processed data
├── src/
│   ├── preprocessing/loader.py     # Load, clean, split data (72/8/20)
│   ├── preprocessing/processor.py  # Z-score scaling & label encoding
│   ├── features/builder.py         # Feature importance selection
│   ├── models/trainer.py           # RF, XGBoost, LightGBM training
│   ├── attacks/fgsm.py             # Fast Gradient Sign Method
│   ├── attacks/pgd.py              # Projected Gradient Descent
│   ├── defenses/adversarial_training.py
│   ├── defenses/feature_selection.py
│   ├── defenses/ensemble.py
│   ├── defenses/feature_squeezing.py
│   ├── evaluation/metrics.py       # Accuracy, ASR, robustness curves
│   ├── evaluation/comparison.py    # Defense comparison plots
│   ├── explainability/explainer.py # SHAP explanations
│   ├── utils/helpers.py            # Logging, save/load
│   ├── api/main.py                 # FastAPI + WebSocket endpoints
│   ├── api/streamer.py             # Real-time IoMT inference streamer
│   └── dashboard/streamlit_app.py  # Interactive dashboard (auth + live demo)
├── models/                         # Saved model pickles
├── reports/                        # Plots, CSVs, metrics
├── logs/                           # Pipeline logs
├── scripts/run_pipeline.py         # End-to-end pipeline
├── deployment/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/                          # Unit tests
├── report_thesis/                  # Thesis documents (LaTeX + DOCX)
└── requirements.txt
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your dataset
Put any classification CSV file in `data/raw/`. The pipeline auto-detects the label column.

### 3. Run the full pipeline
```bash
python scripts/run_pipeline.py
```
Options: `--skip-attack`, `--skip-defense`, `--skip-explain`, `--max-rows N`, `--chunksize N`

### 4. Launch the Dashboard
```bash
streamlit run src/dashboard/streamlit_app.py
```
Login: `admin` / `ciciomt2024`

### 5. Launch the API
```bash
uvicorn src.api.main:app --reload
```
Default credentials: `admin` / `ciciomt2024`

### 6. Docker
```bash
docker-compose -f deployment/docker-compose.yml up
```

## Models
- **Random Forest** (30 trees, depth 8)
- **XGBoost** (30 trees, depth 4, lr=0.1) — Best performer: 93.70% accuracy, 93.63% F1
- **LightGBM** (30 trees, depth 4, lr=0.1)

## Data Split
- **Training:** 72%
- **Validation:** 8%
- **Test:** 20%

## Evasion Attacks
- **FGSM** — Fast Gradient Sign Method (tabular adaptation via feature importance proxy)
- **PGD** — Projected Gradient Descent (iterative, 10 steps, random restarts)

Both attacks are adapted for tree-based models (which lack smooth gradients).

## Defenses
- **Adversarial Training** — Augments training with FGSM examples (most balanced robustness)
- **Feature Selection** — Removes top-K most vulnerable features (K=20)
- **Ensemble** — Soft voting across RF, XGBoost, LightGBM
- **Feature Squeezing** — Bit-depth reduction (4-bit) & median smoothing

## Dashboard Features
- **Authentication:** Login page gates all content. Logout in sidebar.
- **Data Loading:** Select dataset, configure max rows/chunk size, load & process
- **Pipeline Audit:** Displays previous run metadata (duration, host, timestamps)
- **Model Comparison:** Table + grouped bar chart of Accuracy, Precision, Recall, F1
- **Best Model Banner:** Auto-highlights the best model by F1 score
- **Per-Class Confusion Matrix:** Select model + class to view TP/TN/FP/FN
- **Attack Panel:** Run FGSM or PGD at configurable epsilon
- **Defense Panel:** Run any defense with model and hyperparameter controls
- **SHAP Explanations:** Summary bar plots for any trained model
- **Live Inference Demo:** Real-time streaming of test samples with attack toggles

## API Endpoints (FastAPI + WebSocket)

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /` | Public | API info |
| `GET /live/status` | Public | Health check |
| `GET /models` | Required | List available models |
| `POST /predict/{name}` | Required | Run inference |
| `GET /live/next/{name}` | Required | Poll next test sample + prediction |
| `WS /ws/live/{name}` | Required | WebSocket streaming (send credentials first) |
| `POST /attack/fgsm` | Required | Generate FGSM examples |
| `POST /attack/pgd` | Required | Generate PGD examples |
| `POST /upload` | Required | Upload new dataset |

Auth methods: HTTP Basic Auth or X-API-Key header.

## Live Inference Demo
The dashboard includes a real-time inference section that simulates IoMT data streaming:
- Select model (RF, XGBoost, LightGBM)
- Toggle between No Attack / FGSM / PGD
- Adjust epsilon and delay controls
- View running accuracy, latency, and prediction history
- Watch accuracy collapse under attack in real-time
- Average latency: ~2-5ms per prediction (XGBoost)

## Dataset Flexibility
Works with any binary/multi-class classification dataset in CSV format (or Parquet, TXT, ZIP, GZ). Drop your file in `data/raw/` and run.
"# Adversarial-Machine-Learning-defense-thesis" 
"# Adversarial-Machine-Learning-defense-thesis" 
