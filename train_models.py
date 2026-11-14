# %% STEP 1: Imports, settings and data
# Trains both models and exports every file the dashboard needs. Run: python train_models.py
import io
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
GAP = pd.Timedelta(days=60)      # gap between periods = label window
REVIEW_CAPACITY = 0.10           # service team can review the top 10% risky customers


def load_csv(path, **kwargs):
    """These files have every line wrapped in quotes. Remove them, then read normally."""
    with open(path, encoding="utf-8-sig") as f:
        lines = [line.strip() for line in f]
    lines = [l[1:-1] if l.startswith('"') and l.endswith('"') else l for l in lines]
    return pd.read_csv(io.StringIO("\n".join(lines)), **kwargs)


df = load_csv("data/synthetic_customers.csv",
              parse_dates=["snapshot_date", "label_available_date"])

# Cleaning rule: impossible values become missing (the imputer fills them later)
df.loc[df["days_since_last_transaction"] > 365, "days_since_last_transaction"] = np.nan

# %% STEP 2: Time-aware split (train 60% | gap | validation 20% | gap | test 20%)
dates = pd.Series(np.sort(df["snapshot_date"].unique()))
val_start = dates[int(len(dates) * 0.6)]
test_start = dates[int(len(dates) * 0.8)]

train = df[df["snapshot_date"] < val_start - GAP]            # labels finish before validation starts
val = df[(df["snapshot_date"] >= val_start) & (df["snapshot_date"] < test_start - GAP)]
test = df[df["snapshot_date"] >= test_start]                 # untouched until STEP 8

for name, part in [("train", train), ("validation", val), ("test", test)]:
    print(f"{name:<11} rows={len(part):>6}  {part.snapshot_date.min().date()} to "
          f"{part.snapshot_date.max().date()}  churn={part.churn_next_60d.mean():.1%}")
print("Validation customers also in train: "
      f"{val['customer_id'].isin(train['customer_id']).mean():.1%}")

# %% STEP 3: Features, targets and preprocessing
# NOT features: customer_id, snapshot_date, label_available_date, both labels
NUM = ["avg_balance_inr", "previous_month_balance_inr", "monthly_credits_inr",
       "monthly_debits_inr", "transaction_count_30d", "digital_sessions_30d",
       "days_since_last_transaction", "complaint_count_90d", "tenure_months"]
CAT = ["account_type", "branch_code"]
FEATURES = NUM + CAT
REG_TARGET, CLF_TARGET = "next_month_avg_balance_inr", "churn_next_60d"

X_train, X_val, X_test = train[FEATURES], val[FEATURES], test[FEATURES]
y_train, y_val, y_test = train[CLF_TARGET], val[CLF_TARGET], test[CLF_TARGET]


def make_preprocessor(scale):
    """Median-impute numbers (+ standardize if scale=True), one-hot encode categories.
    Fitted on training data only, because it lives inside the Pipeline."""
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer([("num", Pipeline(steps), NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


# %% STEP 4: Balance forecasting - baseline vs Linear Regression
def regression_scores(y_true, y_pred):
    return {"MAE (INR)": mean_absolute_error(y_true, y_pred),
            "RMSE (INR)": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred)}


linreg = Pipeline([("prep", make_preprocessor(scale=True)),
                   ("model", LinearRegression())]).fit(X_train, train[REG_TARGET])
reg_pred = linreg.predict(X_val)

print(pd.DataFrame({
    "Baseline (last month balance)": regression_scores(val[REG_TARGET], val["avg_balance_inr"]),
    "Linear Regression": regression_scores(val[REG_TARGET], reg_pred),
}).T.round(2))
print("Negative predictions to investigate:", int((reg_pred < 0).sum()))

# %% STEP 5: Churn - tune Logistic Regression, Random Forest and KNN on validation data
CANDIDATES = {
    "Logistic Regression": dict(
        build=lambda **p: LogisticRegression(class_weight="balanced", max_iter=1000, **p),
        grid={"C": [0.1, 1, 10]}, scale=True),
    "Random Forest": dict(
        build=lambda **p: RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                 random_state=SEED, n_jobs=-1, **p),
        grid={"max_depth": [5, 10, None], "min_samples_leaf": [5, 20]}, scale=False),
    "KNN": dict(
        build=lambda **p: KNeighborsClassifier(**p),
        grid={"n_neighbors": [5, 11, 21], "weights": ["uniform", "distance"]}, scale=True),
}

results = {}
for name, cfg in CANDIDATES.items():
    for params in ParameterGrid(cfg["grid"]):
        pipe = Pipeline([("prep", make_preprocessor(cfg["scale"])),
                         ("model", cfg["build"](**params))]).fit(X_train, y_train)
        start = time.time()
        proba = pipe.predict_proba(X_val)[:, 1]
        seconds = time.time() - start
        score = average_precision_score(y_val, proba)          # PR-AUC
        if name not in results or score > results[name]["pr_auc"]:
            results[name] = dict(pipe=pipe, proba=proba, params=params,
                                 pr_auc=score, seconds=seconds)
    print(f"{name}: best params {results[name]['params']}  PR-AUC {results[name]['pr_auc']:.3f}")

majority = DummyClassifier(strategy="prior").fit(X_train, y_train)
print("Majority-class baseline PR-AUC:",
      round(average_precision_score(y_val, majority.predict_proba(X_val)[:, 1]), 3))

# %% STEP 6: Compare models, choose the best one and its threshold
rows = []
for name, r in results.items():
    cutoff = np.quantile(r["proba"], 1 - REVIEW_CAPACITY)       # flag top 10% of customers
    flagged = (r["proba"] >= cutoff).astype(int)
    rows.append({"model": name, "PR-AUC": r["pr_auc"],
                 "ROC-AUC": roc_auc_score(y_val, r["proba"]),
                 "precision": precision_score(y_val, flagged),
                 "recall": recall_score(y_val, flagged),
                 "scoring seconds": r["seconds"]})
comparison = pd.DataFrame(rows).set_index("model").sort_values("PR-AUC", ascending=False)
print(comparison.round(3))

best_name = comparison.index[0]
best_proba = results[best_name]["proba"]
threshold = float(np.quantile(best_proba, 1 - REVIEW_CAPACITY))
print(f"\nChosen: {best_name} | threshold = {threshold:.3f} (top {REVIEW_CAPACITY:.0%} reviewed)")

# Calibration: are the probabilities reliable risk estimates?
observed, predicted = calibration_curve(y_val, best_proba, n_bins=5, strategy="quantile")
print("Calibration (predicted vs actual churn):")
print(pd.DataFrame({"predicted": predicted, "actual": observed}).round(3))
print("Brier score:", round(brier_score_loss(y_val, best_proba), 4))

# %% STEP 7: Interpret the models
lr = results["Logistic Regression"]["pipe"]
coefs = pd.Series(lr.named_steps["model"].coef_[0],
                  index=lr.named_steps["prep"].get_feature_names_out()).sort_values()
print("Logistic Regression - pushes churn DOWN:\n", coefs.head(5).round(2))
print("Logistic Regression - pushes churn UP:\n", coefs.tail(5).round(2))

sample = val.sample(5000, random_state=SEED)                    # sample keeps it fast
imp = permutation_importance(results["Random Forest"]["pipe"], sample[FEATURES],
                             sample[CLF_TARGET], scoring="average_precision",
                             n_repeats=3, random_state=SEED, n_jobs=-1)
print("Random Forest permutation importance:\n",
      pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False).round(4))

# %% STEP 8: Final test evaluation (run ONCE, choices are frozen)
print("Test class counts:", y_test.value_counts().to_dict())

test_proba = results[best_name]["pipe"].predict_proba(X_test)[:, 1]
test_flag = (test_proba >= threshold).astype(int)
print(f"\nCHURN ({best_name})")
print("PR-AUC   :", round(average_precision_score(y_test, test_proba), 3),
      "| majority baseline:", round(y_test.mean(), 3))
print("ROC-AUC  :", round(roc_auc_score(y_test, test_proba), 3))
print("Precision:", round(precision_score(y_test, test_flag), 3))
print("Recall   :", round(recall_score(y_test, test_flag), 3))
print("F1       :", round(f1_score(y_test, test_flag), 3))
print("Confusion matrix [[TN, FP], [FN, TP]]:\n", confusion_matrix(y_test, test_flag))

print("\nBALANCE FORECAST")
print(pd.DataFrame({
    "Baseline (last month balance)": regression_scores(test[REG_TARGET], test["avg_balance_inr"]),
    "Linear Regression": regression_scores(test[REG_TARGET], linreg.predict(X_test)),
}).T.round(2))

# %% STEP 9: Save pipelines and results
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
joblib.dump(linreg, "models/balance_model.joblib")
joblib.dump(results[best_name]["pipe"], "models/churn_model.joblib")
with open("models/churn_settings.json", "w") as f:
    json.dump({"classifier": best_name, "threshold": threshold, "features": FEATURES}, f, indent=2)
comparison.round(4).to_csv("reports/churn_model_comparison.csv")
print("Saved models/ and reports/")

# %% STEP 10: Export extra reports for the dashboard
cm = confusion_matrix(y_test, test_flag)
pd.DataFrame(cm, index=["Actual: stays", "Actual: churns"],
             columns=["Predicted: stays", "Predicted: churns"]).to_csv("reports/test_confusion_matrix.csv")

balance_pred = linreg.predict(X_test)
pd.DataFrame({"customer_id": test["customer_id"], "snapshot_date": test["snapshot_date"],
              "actual_balance_inr": test[REG_TARGET],
              "baseline_balance_inr": test["avg_balance_inr"],
              "predicted_balance_inr": balance_pred.round(2),
              "residual_inr": (test[REG_TARGET] - balance_pred).round(2)}
             ).to_csv("reports/balance_test_predictions.csv", index=False)

pd.DataFrame({"Baseline (last month balance)": regression_scores(test[REG_TARGET], test["avg_balance_inr"]),
              "Linear Regression": regression_scores(test[REG_TARGET], balance_pred)}
             ).T.round(2).rename_axis("model").to_csv("reports/balance_test_results.csv")

with open("reports/final_test_metrics.json", "w") as f:
    json.dump({"classifier": best_name, "threshold": threshold, "test_rows": int(len(test)),
               "test_pr_auc": average_precision_score(y_test, test_proba),
               "test_roc_auc": roc_auc_score(y_test, test_proba),
               "test_precision": precision_score(y_test, test_flag),
               "test_recall": recall_score(y_test, test_flag),
               "test_f1": f1_score(y_test, test_flag)}, f, indent=2)
print("Dashboard reports saved in reports/")
