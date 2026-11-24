
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. PATHS AND SETTINGS
# ---------------------------------------------------------
ROOT = Path("/content")
CLEAN_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF_DATE = pd.Timestamp("2026-03-31")

NUMERIC_FEATURES = [
    "tenure_months",
    "age",
    "add_on_count",
    "monthly_recharge_inr",
    "hours_watched_per_week",
    "hd_hours_per_week",
    "sports_hours_per_week",
    "channels_watched",
    "support_calls_90d",
    "n_success_6m",
    "n_failed_6m",
    "recency_days",
]

CATEGORICAL_FEATURES = [
    "city",
    "plan_id",
    "payment_mode",
    "auto_renew",
]

LEAKAGE_FEATURES = [
    "closure_request_logged",
    "next_cycle_recharge_inr",
]

TARGET = "churned"


# ---------------------------------------------------------
# 2. LOAD CLEAN DATA
# ---------------------------------------------------------

subscribers = pd.read_csv(CLEAN_DIR / "subscribers.csv")
recharges = pd.read_csv(CLEAN_DIR / "recharges.csv")

print("\n========== PART B: CHURN PREDICTION ==========")
print("Subscribers:", subscribers.shape)
print("Recharges:", recharges.shape)

required_subscriber_columns = (
    NUMERIC_FEATURES[:9]
    + CATEGORICAL_FEATURES
    + [TARGET, "subscriber_id"]
)

missing_columns = [
    col for col in required_subscriber_columns
    if col not in subscribers.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in subscribers.csv: {missing_columns}"
    )

# Convert target into 0 and 1 if it is stored as text.
if subscribers[TARGET].dtype == "object":
    target_map = {
        "yes": 1,
        "true": 1,
        "1": 1,
        "churned": 1,
        "no": 0,
        "false": 0,
        "0": 0,
        "stayed": 0,
        "stay": 0,
    }

    subscribers[TARGET] = (
        subscribers[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(target_map)
    )

subscribers[TARGET] = pd.to_numeric(
    subscribers[TARGET], errors="coerce"
)

if subscribers[TARGET].isna().any():
    raise ValueError("The churn target contains missing or unknown values.")

subscribers[TARGET] = subscribers[TARGET].astype(int)

if not set(subscribers[TARGET].unique()).issubset({0, 1}):
    raise ValueError("The churn target must contain only 0 and 1.")


# ---------------------------------------------------------
# 3. RECHARGE FEATURE ENGINEERING
# ---------------------------------------------------------

# Identify the recharge date column.
date_candidates = ["recharge_date", "date", "transaction_date"]
date_col = next(
    (col for col in date_candidates if col in recharges.columns),
    None
)

# Identify the recharge status column.
status_candidates = ["status", "recharge_status", "transaction_status"]
status_col = next(
    (col for col in status_candidates if col in recharges.columns),
    None
)

if date_col is None or status_col is None:
    raise ValueError(
        "Could not identify recharge date/status columns. "
        f"Available columns: {list(recharges.columns)}"
    )

if "subscriber_id" not in recharges.columns:
    raise ValueError("recharges.csv must contain subscriber_id.")

recharges[date_col] = pd.to_datetime(
    recharges[date_col], errors="coerce"
)

recharges[status_col] = (
    recharges[status_col]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Use only transactions available on or before the cutoff date.
eligible = recharges[
    recharges[date_col].notna()
    & (recharges[date_col] <= CUTOFF_DATE)
].copy()

six_month_start = CUTOFF_DATE - pd.DateOffset(months=6)

six_month_rows = eligible[
    eligible[date_col] > six_month_start
].copy()

successful_6m = six_month_rows[
    six_month_rows[status_col] == "success"
]

failed_6m = six_month_rows[
    six_month_rows[status_col] == "failed"
]

success_counts = (
    successful_6m.groupby("subscriber_id")
    .size()
    .rename("n_success_6m")
)

failed_counts = (
    failed_6m.groupby("subscriber_id")
    .size()
    .rename("n_failed_6m")
)

# Last successful recharge up to the cutoff date.
all_successful = eligible[
    eligible[status_col] == "success"
]

last_success = (
    all_successful.groupby("subscriber_id")[date_col]
    .max()
)

recency = (
    (CUTOFF_DATE - last_success)
    .dt.days
    .rename("recency_days")
)

recharge_features = pd.concat(
    [success_counts, failed_counts, recency],
    axis=1
).reset_index()

subscribers = subscribers.merge(
    recharge_features,
    on="subscriber_id",
    how="left"
)

# Subscribers with no recharge rows have zero counts.
subscribers["n_success_6m"] = (
    subscribers["n_success_6m"].fillna(0)
)

subscribers["n_failed_6m"] = (
    subscribers["n_failed_6m"].fillna(0)
)

# Recency remains missing when no successful recharge exists.
subscribers["n_success_6m"] = (
    subscribers["n_success_6m"].astype(int)
)

subscribers["n_failed_6m"] = (
    subscribers["n_failed_6m"].astype(int)
)

print("\nRecharge feature checks:")
print("Missing recency_days:", subscribers["recency_days"].isna().sum())
print("Total successful recharges in 6 months:",
      subscribers["n_success_6m"].sum())
print("Total failed recharges in 6 months:",
      subscribers["n_failed_6m"].sum())


# ---------------------------------------------------------
# 4. CLASS BALANCE AND MODELLING TABLE
# ---------------------------------------------------------

print("\n========== CLASS BALANCE ==========")

class_counts = subscribers[TARGET].value_counts().sort_index()
class_percentages = (
    subscribers[TARGET].value_counts(normalize=True).sort_index() * 100
)

print("Stayed (0):", class_counts.get(0, 0))
print("Churned (1):", class_counts.get(1, 0))
print("Churn percentage:", round(class_percentages.get(1, 0), 2), "%")

X = subscribers[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
y = subscribers[TARGET].copy()
subscriber_ids = subscribers["subscriber_id"].copy()


# ---------------------------------------------------------
# 5. TRAIN / VALIDATION / TEST SPLIT
# ---------------------------------------------------------

# First hold out 20% as the test set.
X_train_val, X_test, y_train_val, y_test, id_train_val, id_test = (
    train_test_split(
        X,
        y,
        subscriber_ids,
        test_size=0.20,
        stratify=y,
        random_state=42
    )
)

# Split the remaining 80% into 75% training and 25% validation.
X_train, X_val, y_train, y_val, id_train, id_val = (
    train_test_split(
        X_train_val,
        y_train_val,
        id_train_val,
        test_size=0.25,
        stratify=y_train_val,
        random_state=42
    )
)

print("\n========== DATA SPLIT ==========")
print("Training:", X_train.shape, "| churners:", int(y_train.sum()))
print("Validation:", X_val.shape, "| churners:", int(y_val.sum()))
print("Test:", X_test.shape, "| churners:", int(y_test.sum()))

print(
    "\nTraining is used to fit preprocessing and models; "
    "validation is used for model/threshold selection; "
    "test is used once for final evaluation."
)


# ---------------------------------------------------------
# 6. PREPROCESSING PIPELINE
# ---------------------------------------------------------

def make_preprocessor(numeric_columns, categorical_columns):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])


preprocessor = make_preprocessor(
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)

# Fit only on training data.
X_train_transformed = preprocessor.fit_transform(X_train)
X_val_transformed = preprocessor.transform(X_val)

print("\n========== PREPROCESSING ==========")
print("Output feature count:", X_train_transformed.shape[1])
print("Missing values after transform:",
      np.isnan(X_train_transformed).sum())

numeric_imputer = (
    preprocessor.named_transformers_["numeric"]
    .named_steps["imputer"]
)

numeric_scaler = (
    preprocessor.named_transformers_["numeric"]
    .named_steps["scaler"]
)

age_index = NUMERIC_FEATURES.index("age")
tenure_index = NUMERIC_FEATURES.index("tenure_months")

print("Learned median age:", numeric_imputer.statistics_[age_index])
print("Scaler mean of tenure_months:", numeric_scaler.mean_[tenure_index])
print("Full-data mean of tenure_months:",
      subscribers["tenure_months"].mean())


# ---------------------------------------------------------
# 7. LEAKAGE EXPERIMENT
# ---------------------------------------------------------

print("\n========== DATA LEAKAGE EXPERIMENT ==========")

for col in LEAKAGE_FEATURES:
    if col not in subscribers.columns:
        raise ValueError(
            f"Expected leakage column '{col}' is missing from subscribers.csv."
        )

leakage_numeric = NUMERIC_FEATURES + ["next_cycle_recharge_inr"]
leakage_categorical = CATEGORICAL_FEATURES + ["closure_request_logged"]

X_leaky = subscribers[
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + LEAKAGE_FEATURES
].copy()

X_leak_train_val, X_leak_test, y_leak_train_val, y_leak_test = (
    train_test_split(
        X_leaky,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )
)

X_leak_train, X_leak_val, y_leak_train, y_leak_val = (
    train_test_split(
        X_leak_train_val,
        y_leak_train_val,
        test_size=0.25,
        stratify=y_leak_train_val,
        random_state=42
    )
)

leakage_preprocessor = make_preprocessor(
    leakage_numeric,
    leakage_categorical
)

leaky_pipeline = Pipeline([
    ("preprocessor", leakage_preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

leaky_pipeline.fit(X_leak_train, y_leak_train)
leaky_predictions = leaky_pipeline.predict(X_leak_val)

normal_pipeline = Pipeline([
    ("preprocessor", make_preprocessor(
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES
    )),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

normal_pipeline.fit(X_train, y_train)
normal_predictions = normal_pipeline.predict(X_val)

print("WITH leakage:")
print("Accuracy:", round(accuracy_score(y_leak_val, leaky_predictions), 3))
print("Churn F1:", round(f1_score(y_leak_val, leaky_predictions), 3))

print("\nWITHOUT leakage:")
print("Accuracy:", round(accuracy_score(y_val, normal_predictions), 3))
print("Churn F1:", round(f1_score(y_val, normal_predictions), 3))

print(
    "\nLeakage columns are excluded from all final models because "
    "they are not available when a real prediction is made. "
    "Their use makes validation results unrealistically high and "
    "cannot represent performance on future subscribers."
)


# ---------------------------------------------------------
# 8. BASELINE AND SIMPLE MODELS
# ---------------------------------------------------------

print("\n========== BASELINE AND SIMPLE MODELS ==========")

# Majority-class baseline: always predicts that subscriber stays.
baseline_predictions = np.zeros(len(y_val), dtype=int)

print("\nMajority baseline:")
print("Accuracy:", round(accuracy_score(y_val, baseline_predictions), 3))
print("Churn F1:", round(f1_score(y_val, baseline_predictions), 3))

logistic_basic = Pipeline([
    ("preprocessor", make_preprocessor(
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES
    )),
    ("model", LogisticRegression(max_iter=1000))
])

logistic_basic.fit(X_train, y_train)
logistic_basic_pred = logistic_basic.predict(X_val)

print("\nBasic Logistic Regression:")
print("Accuracy:", round(accuracy_score(y_val, logistic_basic_pred), 3))
print("Precision:", round(precision_score(y_val, logistic_basic_pred), 3))
print("Recall:", round(recall_score(y_val, logistic_basic_pred), 3))
print("F1:", round(f1_score(y_val, logistic_basic_pred), 3))

tree_basic = Pipeline([
    ("preprocessor", make_preprocessor(
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES
    )),
    ("model", DecisionTreeClassifier(random_state=42))
])

tree_basic.fit(X_train, y_train)
tree_train_pred = tree_basic.predict(X_train)
tree_val_pred = tree_basic.predict(X_val)

print("\nUnrestricted Decision Tree:")
print("Training accuracy:", round(accuracy_score(y_train, tree_train_pred), 3))
print("Validation F1:", round(f1_score(y_val, tree_val_pred), 3))

forest_basic = Pipeline([
    ("preprocessor", make_preprocessor(
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES
    )),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

forest_basic.fit(X_train, y_train)
forest_train_pred = forest_basic.predict(X_train)
forest_val_pred = forest_basic.predict(X_val)

print("\nDefault Random Forest:")
print("Training accuracy:", round(accuracy_score(y_train, forest_train_pred), 3))
print("Validation F1:", round(f1_score(y_val, forest_val_pred), 3))


# ---------------------------------------------------------
# 9. DECISION TREE DEPTH EXPERIMENT
# ---------------------------------------------------------

print("\n========== DECISION TREE DEPTH EXPERIMENT ==========")

depth_results = []

for depth in range(1, 11):
    depth_pipeline = Pipeline([
        ("preprocessor", make_preprocessor(
            NUMERIC_FEATURES,
            CATEGORICAL_FEATURES
        )),
        ("model", DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        ))
    ])

    depth_pipeline.fit(X_train, y_train)

    train_predictions = depth_pipeline.predict(X_train)
    validation_predictions = depth_pipeline.predict(X_val)

    depth_results.append({
        "max_depth": depth,
        "training_accuracy": accuracy_score(y_train, train_predictions),
        "validation_f1": f1_score(y_val, validation_predictions),
    })

depth_df = pd.DataFrame(depth_results)
depth_df.to_csv(OUTPUT_DIR / "decision_tree_depth.csv", index=False)

print(depth_df.round(3).to_string(index=False))

print(
    "\nAs tree depth increases, training accuracy can improve while "
    "validation F1 eventually declines. This widening gap indicates "
    "overfitting: a greedy tree can learn details and noise in its "
    "training data that do not generalise to unseen subscribers."
)


# ---------------------------------------------------------
# 10. GRID SEARCH: LOGISTIC REGRESSION
# ---------------------------------------------------------

print("\n========== LOGISTIC REGRESSION GRID SEARCH ==========")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

logistic_pipeline = Pipeline([
    ("preprocessor", make_preprocessor(
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES
    )),
    ("model", LogisticRegression(max_iter=1000))
])

logistic_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

logistic_search = GridSearchCV(
    logistic_pipeline,
    logistic_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    return_train_score=True
)

logistic_search.fit(X_train, y_train)

logistic_best = logistic_search.best_estimator_

logistic_best_index = logistic_search.best_index_
logistic_cv_mean = logistic_search.cv_results_["mean_test_score"][
    logistic_best_index
]
logistic_cv_std = logistic_search.cv_results_["std_test_score"][
    logistic_best_index
]

print("Best parameters:", logistic_search.best_params_)
print("Best mean CV F1:", round(logistic_cv_mean, 3))
print("CV F1 standard deviation:", round(logistic_cv_std, 3))
print("Combinations:", len(logistic_grid["model__C"]))
print("Model fits:", len(logistic_grid["model__C"]) * 5)


# ---------------------------------------------------------
# 11. GRID SEARCH: RANDOM FOREST
# ---------------------------------------------------------

print("\n========== RANDOM FOREST GRID SEARCH ==========")

forest_pipeline = Pipeline([
    ("preprocessor", make_preprocessor(
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES
    )),
    ("model", RandomForestClassifier(random_state=42))
])

forest_grid = {
    "model__n_estimators": [50, 100, 150],
    "model__max_depth": [3, 4, 5, 6, 8],
    "model__max_features": [3, 4, 5],
}

forest_search = GridSearchCV(
    forest_pipeline,
    forest_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    return_train_score=True
)

forest_search.fit(X_train, y_train)

forest_best = forest_search.best_estimator_

forest_best_index = forest_search.best_index_
forest_cv_mean = forest_search.cv_results_["mean_test_score"][
    forest_best_index
]
forest_cv_std = forest_search.cv_results_["std_test_score"][
    forest_best_index
]

forest_combinations = (
    len(forest_grid["model__n_estimators"])
    * len(forest_grid["model__max_depth"])
    * len(forest_grid["model__max_features"])
)

print("Best parameters:", forest_search.best_params_)
print("Best mean CV F1:", round(forest_cv_mean, 3))
print("CV F1 standard deviation:", round(forest_cv_std, 3))
print("Combinations:", forest_combinations)
print("Model fits:", forest_combinations * 5)

print(
    "\nThe model with the smaller cross-validation standard deviation "
    "is more consistent across folds. Consistency matters because its "
    "performance is less dependent on one particular training subset."
)


# ---------------------------------------------------------
# 12. THRESHOLD SELECTION USING VALIDATION COST
# ---------------------------------------------------------

print("\n========== VALIDATION THRESHOLD SELECTION ==========")

def calculate_cost(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    cost = (1200 * fn) + (300 * fp)

    return {
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "cost": cost,
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
    }


logistic_val_prob = logistic_best.predict_proba(X_val)[:, 1]
forest_val_prob = forest_best.predict_proba(X_val)[:, 1]

thresholds = np.round(np.arange(0.10, 0.901, 0.05), 2)

logistic_threshold_results = [
    calculate_cost(y_val, logistic_val_prob, t)
    for t in thresholds
]

forest_threshold_results = [
    calculate_cost(y_val, forest_val_prob, t)
    for t in thresholds
]

logistic_threshold_df = pd.DataFrame(logistic_threshold_results)
forest_threshold_df = pd.DataFrame(forest_threshold_results)

logistic_threshold_df.to_csv(
    OUTPUT_DIR / "logistic_thresholds.csv", index=False
)
forest_threshold_df.to_csv(
    OUTPUT_DIR / "forest_thresholds.csv", index=False
)

# Select the cheapest threshold from validation only.
best_logistic_threshold_row = (
    logistic_threshold_df.sort_values(
        ["cost", "threshold"]
    ).iloc[0]
)

best_forest_threshold_row = (
    forest_threshold_df.sort_values(
        ["cost", "threshold"]
    ).iloc[0]
)

logistic_threshold = float(
    best_logistic_threshold_row["threshold"]
)

forest_threshold = float(
    best_forest_threshold_row["threshold"]
)

print("Logistic best threshold:", logistic_threshold)
print("Logistic validation cost:",
      int(best_logistic_threshold_row["cost"]))

print("Random Forest best threshold:", forest_threshold)
print("Random Forest validation cost:",
      int(best_forest_threshold_row["cost"]))


# ---------------------------------------------------------
# 13. FINAL TEST EVALUATION
# ---------------------------------------------------------

print("\n========== FINAL TEST EVALUATION ==========")

def evaluate_test_model(model, threshold, X_test_data, y_test_data):
    probabilities = model.predict_proba(X_test_data)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test_data,
        predictions,
        labels=[0, 1]
    ).ravel()

    total_cost = (1200 * fn) + (300 * fp)

    result = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": precision_score(
            y_test_data, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test_data, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test_data, predictions, zero_division=0
        ),
        "cost": int(total_cost),
        "probabilities": probabilities,
        "predictions": predictions,
    }

    return result


logistic_test_result = evaluate_test_model(
    logistic_best,
    logistic_threshold,
    X_test,
    y_test
)

forest_test_result = evaluate_test_model(
    forest_best,
    forest_threshold,
    X_test,
    y_test
)

def print_test_result(name, result):
    print(f"\n{name}")
    print("TN / FP / FN / TP:",
          result["tn"], "/", result["fp"], "/",
          result["fn"], "/", result["tp"])
    print("Precision:", round(result["precision"], 3))
    print("Recall:", round(result["recall"], 3))
    print("F1:", round(result["f1"], 3))
    print("Total cost: INR", result["cost"])


print_test_result("LOGISTIC REGRESSION", logistic_test_result)
print_test_result("RANDOM FOREST", forest_test_result)

test_churners = int(y_test.sum())
no_action_cost = 1200 * test_churners
offer_everybody_cost = 300 * (len(y_test) - test_churners)

print("\nReference policy comparison:")
print("No action cost: INR", no_action_cost)
print("Offer everybody cost: INR", offer_everybody_cost)


# ---------------------------------------------------------
# 14. SELECT MODEL USING VALIDATION COST
# ---------------------------------------------------------

# Selection is based on validation cost, not test performance.
if best_forest_threshold_row["cost"] < best_logistic_threshold_row["cost"]:
    selected_name = "Random Forest"
    selected_model = forest_best
    selected_threshold = forest_threshold
    selected_test_result = forest_test_result
else:
    selected_name = "Logistic Regression"
    selected_model = logistic_best
    selected_threshold = logistic_threshold
    selected_test_result = logistic_test_result

print("\n========== SELECTED MODEL ==========")
print("Selected model:", selected_name)
print("Selected threshold:", selected_threshold)
print("Selected test cost: INR", selected_test_result["cost"])
print("Selected test recall:", round(selected_test_result["recall"], 3))
print("Selected test F1:", round(selected_test_result["f1"], 3))

print(
    "\nThe selected model is chosen using validation cost, not test "
    "performance. The test set is used only for the final evaluation "
    "after model and threshold selection."
)

if selected_test_result["cost"] > 74160:
    print("WARNING: Test cost is above the required INR 74,160.")
if selected_test_result["recall"] < 0.55:
    print("WARNING: Test recall is below the required 0.55.")
if selected_test_result["f1"] < 0.50:
    print("WARNING: Test F1 is below the required 0.50.")


# ---------------------------------------------------------
# 15. FEATURE IMPORTANCE AND LOGISTIC COEFFICIENTS
# ---------------------------------------------------------

print("\n========== FEATURE IMPORTANCE ==========")

forest_preprocessor = forest_best.named_steps["preprocessor"]
forest_model = forest_best.named_steps["model"]

feature_names = forest_preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": forest_model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nTop five Random Forest features:")
print(importance_df.head(5).round(4).to_string(index=False))

importance_df.to_csv(
    OUTPUT_DIR / "random_forest_feature_importance.csv",
    index=False
)

print("\n========== LOGISTIC REGRESSION COEFFICIENTS ==========")

logistic_preprocessor = logistic_best.named_steps["preprocessor"]
logistic_model = logistic_best.named_steps["model"]

logistic_feature_names = logistic_preprocessor.get_feature_names_out()

coefficient_df = pd.DataFrame({
    "feature": logistic_feature_names,
    "coefficient": logistic_model.coef_[0],
})

requested_features = [
    "support_calls_90d",
    "tenure_months",
    "n_failed_6m",
]

for feature in requested_features:
    matching = coefficient_df[
        coefficient_df["feature"].str.endswith(feature)
    ]

    if not matching.empty:
        print(matching.to_string(index=False))

coefficient_df.to_csv(
    OUTPUT_DIR / "logistic_regression_coefficients.csv",
    index=False
)


# ---------------------------------------------------------
# 16. SAVE MODEL AND VERIFY RELOADING
# ---------------------------------------------------------

model_path = OUTPUT_DIR / "churn_model.joblib"

joblib.dump(
    {
        "model": selected_model,
        "threshold": selected_threshold,
        "model_name": selected_name,
    },
    model_path
)

loaded_package = joblib.load(model_path)
loaded_model = loaded_package["model"]
loaded_threshold = loaded_package["threshold"]

original_probabilities = selected_model.predict_proba(X_test)[:, 1]
loaded_probabilities = loaded_model.predict_proba(X_test)[:, 1]

original_predictions = (
    original_probabilities >= selected_threshold
).astype(int)

loaded_predictions = (
    loaded_probabilities >= loaded_threshold
).astype(int)

print("\nModel reload verification:")
print(
    "Predictions identical:",
    np.array_equal(original_predictions, loaded_predictions)
)


# ---------------------------------------------------------
# 17. SAVE TEST SCORES
# ---------------------------------------------------------

test_scores = pd.DataFrame({
    "subscriber_id": id_test.values,
    "churn_probability": original_probabilities,
    "churned": y_test.values,
})

test_scores = test_scores.sort_values("subscriber_id")

test_scores.to_csv(
    OUTPUT_DIR / "test_scores.csv",
    index=False
)

print("\nSaved:", model_path)
print("Saved:", OUTPUT_DIR / "test_scores.csv")
print("Test score rows:", len(test_scores))

print("\n========== PART B COMPLETED ==========")
