# Wellness Tourism Package — MLOps Pipeline

Predicting whether a customer will purchase the newly introduced **Wellness Tourism Package** so the marketing team can target the right prospects before calling them. The entire flow — data validation, cleaning, model training, hyperparameter tuning, MLflow tracking and deployment — is automated end-to-end with **GitHub Actions** and served through a **Streamlit** web app.

> Built as part of the *Visit with Us* MLOps assignment.

---

## Table of Contents
1. [Architecture](#architecture)
2. [Repository Layout](#repository-layout)
3. [Tech Stack](#tech-stack)
4. [Local Setup](#local-setup)
5. [Pipeline Stages](#pipeline-stages)
6. [MLflow Experiment Tracking](#mlflow-experiment-tracking)
7. [GitHub Actions Workflow](#github-actions-workflow)
8. [Streamlit Deployment](#streamlit-deployment)
9. [Outputs](#outputs)

---

## Architecture

```
                ┌────────────────────┐
                │  tourism.csv       │
                │  (raw dataset)     │
                └─────────┬──────────┘
                          │
                          ▼
            ┌─────────────────────────────┐
   Job 1    │   register-dataset          │  validates columns,
            │   data_register.py          │  prints summary
            └─────────────┬───────────────┘
                          │  artifact: tourism.csv
                          ▼
            ┌─────────────────────────────┐
   Job 2    │   data-prep                 │  cleans + imputes,
            │   prep.py                   │  80/20 stratified split
            └─────────────┬───────────────┘
                          │  artifact: Xtrain/Xtest/ytrain/ytest
                          ▼
            ┌─────────────────────────────┐
   Job 3    │   model-traning             │  GridSearchCV + XGBoost,
            │   train.py                  │  logs every trial to MLflow
            └─────────────┬───────────────┘
                          │ commits
                          ▼
            ┌─────────────────────────────┐
            │   best_model.joblib         │  → Streamlit Community Cloud
            │   tourism_project/          │
            │     deployment/             │
            └─────────────────────────────┘
```

---

## Repository Layout

```
.
├── .github/
│   └── workflows/
│       └── pipeline.yml          # GitHub Actions workflow
├── tourism_project/
│   ├── data/
│   │   └── tourism.csv           # raw dataset (input)
│   ├── model_building/
│   │   ├── data_register.py      # Job 1: validate CSV
│   │   ├── prep.py               # Job 2: clean + split
│   │   └── train.py              # Job 3: train + tune + log
│   ├── deployment/
│   │   ├── app.py                # Streamlit front-end
│   │   ├── best_model.joblib     # trained model (output)
│   │   └── requirements.txt      # Streamlit Cloud dependencies
│   └── requirements.txt          # CI dependencies
└── README.md
```

After `data-prep` runs, the four split files are written into `tourism_project/data/` (`Xtrain.csv`, `Xtest.csv`, `ytrain.csv`, `ytest.csv`) and passed between jobs as a workflow artifact.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data | pandas, scikit-learn |
| Model | XGBoost (`XGBClassifier`) inside a `ColumnTransformer` + `Pipeline` |
| Tuning | `GridSearchCV` with 24 candidates × 3-fold CV |
| Tracking | MLflow 3.x |
| CI/CD | GitHub Actions |
| App | Streamlit 1.39 |
| Packaging | joblib |
| Auth | PyGithub (PAT stored in Colab secrets) |

---

## Local Setup

You can run the pipeline on your own machine by following the steps in `tourism_wellness_mlops.ipynb`, or with the CLI:

```bash
# 1. clone
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. install
pip install -r tourism_project/requirements.txt

# 3. drop the dataset
cp /path/to/tourism.csv tourism_project/data/tourism.csv

# 4. run each stage
python tourism_project/model_building/data_register.py
python tourism_project/model_building/prep.py
python tourism_project/model_building/train.py
```

The training step logs every hyperparameter trial to MLflow. To view the UI:

```bash
mlflow ui --backend-store-uri ./mlruns
# open http://localhost:5000
```

Or expose it via ngrok when running in Colab (see the notebook cell for the snippet).

---

## Pipeline Stages

### 1. Register Dataset — `data_register.py`
- Loads `tourism_project/data/tourism.csv`.
- Asserts that all 20 expected columns are present.
- Prints shape, target distribution and per-column null counts.

### 2. Data Preparation — `prep.py`
- Drops `CustomerID`.
- Imputes numeric NaNs with the median, categorical NaNs with the mode.
- Stratified 80/20 train/test split on `ProdTaken`.
- Writes `Xtrain.csv`, `Xtest.csv`, `ytrain.csv`, `ytest.csv` into `tourism_project/data/`.

### 3. Model Training — `train.py`
- Builds a `Pipeline(ColumnTransformer → XGBClassifier)`.
  - Numeric features → `StandardScaler`.
  - Categorical features → `OneHotEncoder(handle_unknown="ignore")`.
- Runs `GridSearchCV` over:

| Hyperparameter | Values |
|---|---|
| `n_estimators` | 100, 200 |
| `max_depth` | 3, 5, 7 |
| `learning_rate` | 0.05, 0.1 |
| `subsample` | 0.8, 1.0 |

  → 24 combinations × 3 folds = **72 fits**.
- Logs **every** combination as a nested MLflow run (`trial-0` … `trial-23`) plus a parent run (`tuning-parent`) with the final test metrics and the saved model.
- Reports accuracy, weighted precision / recall / F1 on the held-out test set.
- Saves the best estimator to `tourism_project/deployment/best_model.joblib` so the workflow can commit it back to `main`.

---

## MLflow Experiment Tracking

Every pipeline run creates an MLflow experiment named **`tourism-wellness-prediction`** with the structure:

```
tourism-wellness-prediction/
└── tuning-parent/        # final metrics + best params + model artifact
    ├── trial-0/          # params + cv_mean_f1 + rank
    ├── trial-1/
    ├── …
    └── trial-23/
```

Compare trials by sorting child runs on `metrics.cv_mean_f1`. The parent run's `best_run_id` tag points to the winning trial.

---

## GitHub Actions Workflow

`.github/workflows/pipeline.yml` defines three jobs that run sequentially:

```
push to main  ──►  register-dataset  ──►  data-prep  ──►  model-traning  ──►  commit model
                   (validate CSV)        (clean/split)   (tune + log)         (best_model.joblib)
```

- `permissions: contents: write` lets the final job push the trained model back to `main`.
- `workflow_dispatch` lets you trigger the pipeline manually from the Actions tab.
- The `[skip ci]` marker on the auto-commit prevents the workflow from re-triggering itself in an infinite loop.

Every job installs dependencies from `tourism_project/requirements.txt` and uses Python 3.11 so the same library versions are used locally and in CI.

---

## Streamlit Deployment

Once the pipeline has run at least once, `tourism_project/deployment/best_model.joblib` exists on `main`. To deploy:

1. Go to **https://share.streamlit.io** and sign in with the GitHub account that owns this repo.
2. Click **Create app** and set:
   - **Repository:** `<your-username>/<your-repo>`
   - **Branch:** `main`
   - **Main file path:** `tourism_project/deployment/app.py`
3. Open **Advanced settings → Python version** and pick **3.11** (matches training).
4. Click **Deploy**.

Streamlit installs the packages listed in `tourism_project/deployment/requirements.txt` and serves the app at a public `*.streamlit.app` URL.

The app:
- Loads the committed `best_model.joblib`.
- Collects 18 customer / interaction features through a form.
- Returns a class prediction plus a purchase probability.

---

## Outputs

| Artefact | Location |
|---|---|
| Trained model | `tourism_project/deployment/best_model.joblib` |
| Train/test splits | `tourism_project/data/{X,Xtest,y,ytrain,ytest}.csv` |
| MLflow runs | `mlruns/` (local) or MLflow UI |
| Live demo | `https://<your-app>.streamlit.app` |

---

## License

MIT — for educational use.
