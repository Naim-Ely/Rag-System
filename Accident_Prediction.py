import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix, roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel

RANDOM_STATE = 42

# =========================
# CONFIG (run-friendly)
# =========================
CSV_PATH = "US_Accidents_March23.csv"  # <-- CHANGE THIS
TARGET_COL = "Severity"

# Binary target: 1/2 = low severity, 3/4 = high severity
BINARY_TARGET = True
HIGH_SEVERITY = {3, 4}

# Runtime controls (good for i7-9700X + 32GB RAM)
NROWS = 200_000
LEARNING_CURVE_MODELS = ["LogReg", "HistGBDT", "MLP"]  # only 3 curves
N_JOBS = -1

# =========================
# 1) LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH, nrows=NROWS)

# Parse datetimes
for col in ["Start_Time", "End_Time"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Duration and time features
if "Start_Time" in df.columns and "End_Time" in df.columns:
    df["duration_min"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60.0
if "Start_Time" in df.columns:
    df["start_hour"] = df["Start_Time"].dt.hour
    df["start_dayofweek"] = df["Start_Time"].dt.dayofweek
    df["start_month"] = df["Start_Time"].dt.month

# Drop high-cardinality / text-heavy / leaky columns + all-missing columns
drop_cols = [
    "ID", "Description", "Street", "City", "County", "Zipcode",
    "Airport_Code", "Weather_Timestamp",
    "Start_Time", "End_Time",
    "End_Lat", "End_Lng"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# Target + features
y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

if BINARY_TARGET:
    y = y.apply(lambda v: 1 if v in HIGH_SEVERITY else 0)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# =========================
# 2) PREPROCESSING
# =========================
num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Sparse output (good for LogReg / RF / MLP)
pre_sparse = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ],
    remainder="drop"
)

# Dense output (required for HistGradientBoostingClassifier)
pre_dense = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ],
    remainder="drop",
    sparse_threshold=0.0  # force dense output
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# =========================
# 3) METRICS + EVAL
# =========================
def eval_model(pipe, name):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    y_proba = None
    if hasattr(pipe, "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        print(f"{k:>16}: {v:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    return metrics

# =========================
# 4) BASELINE MODEL COMPARISON
# =========================
models = {
    "LogReg": Pipeline(steps=[
        ("pre", pre_sparse),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=N_JOBS, class_weight="balanced"))
    ]),
    "RandomForest": Pipeline(steps=[
        ("pre", pre_sparse),
        ("clf", RandomForestClassifier(
            n_estimators=250, random_state=RANDOM_STATE,
            n_jobs=N_JOBS, class_weight="balanced_subsample"
        ))
    ]),
    "HistGBDT": Pipeline(steps=[
        ("pre", pre_dense),  # <-- FIX: dense required
        ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE))
    ]),
    "MLP": Pipeline(steps=[
        ("pre", pre_sparse),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=RANDOM_STATE
        ))
    ])
}

baseline_results = {}
for name, pipe in models.items():
    baseline_results[name] = eval_model(pipe, name)

baseline_df = pd.DataFrame(baseline_results).T.sort_values("macro_f1", ascending=False)
print("\n=== BASELINE SUMMARY (sorted by Macro-F1) ===")
print(baseline_df.round(4))

# =========================
# 5) REGULARIZATION + CROSS-VALIDATION (tune 2 models only)
# =========================
param_grids = {
    "LogReg": {
        "clf__C": [0.1, 1, 10],  # L2 regularization
    },
    "HistGBDT": {
        "clf__learning_rate": [0.03, 0.1],
        "clf__max_depth": [6, 10],
        "clf__max_iter": [200, 400],
    }
}

tuned_best = {}
tuned_results = {}

for name, grid in param_grids.items():
    print(f"\n--- GridSearchCV tuning: {name} ---")
    gs = GridSearchCV(
        models[name],
        param_grid=grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=N_JOBS,
        verbose=0
    )
    gs.fit(X_train, y_train)
    print("Best params:", gs.best_params_)
    tuned_best[name] = gs.best_estimator_
    tuned_results[f"{name}_TUNED"] = eval_model(gs.best_estimator_, f"{name} (TUNED)")

tuned_df = pd.DataFrame(tuned_results).T.sort_values("macro_f1", ascending=False)
print("\n=== TUNED SUMMARY (sorted by Macro-F1) ===")
print(tuned_df.round(4))

# =========================
# 6) LEARNING CURVES (Under/Overfitting)
# =========================
def plot_learning_curve(estimator, title):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=estimator,
        X=X_train, y=y_train,
        train_sizes=np.linspace(0.1, 1.0, 6),
        cv=cv,
        scoring="f1_macro",
        n_jobs=N_JOBS
    )
    plt.figure()
    plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="train")
    plt.plot(train_sizes, val_scores.mean(axis=1), marker="o", label="validation")
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Macro-F1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

for name in LEARNING_CURVE_MODELS:
    est = tuned_best.get(name, models[name])
    plot_learning_curve(est, f"Learning Curve: {name}")

# =========================
# 7) EXTRA: L1 FEATURE SELECTION (feature selection helps?)
# =========================
l1_selector = SelectFromModel(
    LogisticRegression(
        penalty="l1", solver="liblinear", C=0.5,
        class_weight="balanced", max_iter=2000
    )
)

pipe_l1 = Pipeline(steps=[
    ("pre", pre_sparse),
    ("select", l1_selector),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

_ = eval_model(pipe_l1, "LogReg + L1 Feature Selection")
