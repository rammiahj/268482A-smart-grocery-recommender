# Project context: Smart Grocery Recommender (MSc DS & AI Capstone)

## Who / what
- Student: Rammiah J (Index 268482A), MSc Data Science & AI, University of Moratuwa. Module CS5998 (individual capstone).
- Supervisor: Prof. Indika Perera.
- Repo: github.com/rammiahj/268482A-smart-grocery-recommender (public). Commits must be made by the student with meaningful messages.
- Title: Design and Evaluation of an Intelligent Software Architecture for Personalized Grocery Replenishment and Recommendation.

## Approved scope (Milestone 1 submitted; scope is locked, no renegotiation)
Three core capabilities:
1. Replenishment prediction: binary classification on user–item candidates ("will a previously bought item be in the user's next order?").
2. Personalized top-K recommendation: item-based collaborative filtering and implicit ALS.
3. Association rules: FP-Growth (Apriori for comparison), support/confidence/lift.
These are merged into a ranked, explainable shopping list served via FastAPI and shown in a Streamlit dashboard.
- Optional extras: K-Means customer segments, unusual-purchase alerts.
- Out of scope: a mobile app, e-commerce platform, payments, live user trials, and external LLMs (none used).

Research questions: RQ1 next-order prediction accuracy; RQ2 value of frequency/recency/interval features (ablation); RQ3 whether CF and co-purchase patterns improve top-K; RQ4 comparison with rule-based lists; RQ5 latency/throughput as data grows.

## Data
- Instacart Online Grocery Shopping Dataset 2017 (Kaggle: psparks/instacart-market-basket-analysis). Non-commercial licence.
- Files: orders, order_products__prior (~32M rows), order_products__train, products, aisles, departments.
- Location: CSVs in `data/raw/`. `data/` is git-ignored (files are too large for GitHub).
- Known limits: no calendar dates; days_since_prior_order is capped at 30 (treat as "30 or more"); no quantity field.
- Rebuild each user's relative day index by cumulatively summing days_since_prior_order.
- Split: each user's last order = test, second-last = validation, earlier orders = features/training. Avoid any future leakage.
- Scalability subsets: 10K / 50K / 100K / all ~200K users.

## Evaluation
- Classification: Precision, Recall, F1, ROC-AUC, PR-AUC (class imbalance; tune the decision threshold).
- Ranking: Precision@K, Recall@K, Hit Rate@K at K = 5, 10, 20.
- Timing: MAE/RMSE in days.
- Baselines: user's most-frequent items, repeat of last order, fixed-interval rule, global popularity.
- System: latency per user, API throughput, memory.

## Milestone 2: Technical Checkpoint (due 30 Nov 2026, 25%)
Required: preprocessing done, EDA, baseline + improved method, system architecture diagram, code link, demo video (max 6 min).
Plan:
- Week 1 (to Nov 1): repo structure, environment, load CSVs to Parquet/DuckDB, cleaning, timeline rebuild, splits.
- Week 2 (Nov 2–8): EDA notebook (orders/user, intervals, reorder rates, top products/departments, basket size).
- Week 3 (Nov 9–15): candidate generation, feature engineering, evaluation framework, rule-based baselines.
- Week 4 (Nov 16–22): Logistic Regression + LightGBM, comparison table, feature ablation, architecture diagram.
- Week 5 (Nov 23–30): minimal FastAPI/Streamlit demo, README, demo video, submit.
CF and association rules come after Milestone 2. Final submission (week 12): 25–30 page report and reproducibility checklist (requirements file, hardware, fixed seeds, splits, preprocessing, hyperparameters).

## Planned structure
```
README.md  requirements.txt  .gitignore
data/raw  data/processed          (git-ignored)
notebooks/01_eda.ipynb
src/data  src/features  src/models  src/api
reports/figures  docs/
```

## Working conventions
- Python, pandas/NumPy, DuckDB/Parquet, scikit-learn, LightGBM, mlxtend, FastAPI, Streamlit, PostgreSQL, Docker.
- Fixed random seed (42); document every split and hyperparameter for reproducibility.
- Keep code explainable: a viva can be called at any milestone, and the student must be able to justify every decision.
- The student commits and pushes; propose small, logical commits with clear messages.
- Keep the API-first design: dashboards and any future app call the API, not the models directly.
