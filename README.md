# Smart Grocery Recommender

**Design and Evaluation of an Intelligent Software Architecture for Personalized Grocery Replenishment and Recommendation**

MSc Data Science & Artificial Intelligence, University of Moratuwa (CS5998 Capstone)
Rammiah J (268482A) · Supervisor: Prof. Indika Perera

## Overview
The system learns from users' grocery purchase histories to:
1. **Predict replenishment:** which previously bought items will appear in the user's next order.
2. **Recommend products:** personalized top-K suggestions using collaborative filtering.
3. **Find associations:** items frequently bought together, using FP-Growth.

These are combined into a ranked, explainable shopping list served through a FastAPI service and a Streamlit dashboard, and evaluated against rule-based baselines.

## Dataset
[Instacart Online Grocery Shopping Dataset 2017](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis) (non-commercial use).
Download it and place these CSVs in `data/raw/`:
`orders.csv`, `order_products__prior.csv`, `order_products__train.csv`, `products.csv`, `aisles.csv`, `departments.csv`.
The data is not committed to this repository.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

## Pipeline
```bash
python -m src.data.load         # CSV -> Parquet
python -m src.data.preprocess   # clean, rebuild timelines, create history/val/test splits
python -m src.features.build_features   # candidates + 25 features (train = val target, test = test target)
python -m src.models.run_baselines      # rule-based baselines
python -m src.models.train              # Logistic Regression + LightGBM, comparison, feature importance
python -m src.models.ablation           # feature-group ablation
python -m src.models.score              # batch predictions for the dashboard
streamlit run src/app/dashboard.py      # demo dashboard
```

Architecture: see [docs/architecture.md](docs/architecture.md).

## Project structure
```
data/raw, data/processed   Instacart data (git-ignored)
notebooks/                 EDA and experiments
src/data/                  loading, cleaning, splits
src/features/              user, product and user×product features
src/models/                baselines, ML models, evaluation
src/api/                   FastAPI service
reports/figures/           generated plots
docs/                      architecture diagram, milestone documents
```

## Evaluation
- Replenishment: Precision, Recall, F1, ROC-AUC, PR-AUC
- Recommendation: Precision@K, Recall@K, Hit Rate@K (K = 5, 10, 20)
- Baselines: most-frequent items, repeat last order, fixed-interval rule, global popularity
- System: latency, throughput, memory across data-volume subsets

## Reproducibility
Fixed random seed (42). Splits are made per user by order sequence: last order = test, second-last = validation, earlier orders = history.
