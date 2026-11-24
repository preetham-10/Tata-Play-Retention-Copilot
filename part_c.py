
# ============================================================
# PART C: SEGMENTS, SPEND MODEL AND REVENUE AT RISK
# Tata Play Subscriber Retention & Support Copilot
# ============================================================

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# ------------------------------------------------------------
# 1. PATHS AND INPUT FILES
# ------------------------------------------------------------

ROOT = Path("/content")
CLEAN_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBSCRIBER_FILE = CLEAN_DIR / "subscribers.csv"
TEST_SCORE_FILE = OUTPUT_DIR / "test_scores.csv"

if not SUBSCRIBER_FILE.exists():
    raise FileNotFoundError(f"Missing file: {SUBSCRIBER_FILE}")

if not TEST_SCORE_FILE.exists():
    raise FileNotFoundError(
        f"Missing file: {TEST_SCORE_FILE}. Run Part B successfully first."
    )

subscribers = pd.read_csv(SUBSCRIBER_FILE)
test_scores = pd.read_csv(TEST_SCORE_FILE)

print("Subscribers:", subscribers.shape)
print("Part B test scores:", test_scores.shape)

# Required columns
required_columns = [
    "subscriber_id",
    "tenure_months",
    "hours_watched_per_week",
    "monthly_recharge_inr",
    "churned",
    "next_cycle_recharge_inr",
    "hd_hours_per_week",
    "sports_hours_per_week",
    "channels_watched",
    "support_calls_90d"
]

missing_columns = [
    col for col in required_columns if col not in subscribers.columns
]

if missing_columns:
    raise ValueError(f"Missing subscriber columns: {missing_columns}")

# Convert relevant columns to numeric
for col in required_columns:
    if col != "subscriber_id":
        subscribers[col] = pd.to_numeric(
            subscribers[col], errors="coerce"
        )

# ------------------------------------------------------------
# 2. K-MEANS CLUSTERING
# ------------------------------------------------------------

print("\n========== K-MEANS CLUSTERING ==========")

cluster_features = [
    "tenure_months",
    "hours_watched_per_week",
    "monthly_recharge_inr"
]

cluster_data = subscribers[cluster_features].copy()

if cluster_data.isna().any().any():
    raise ValueError(
        "Missing values found in K-means features. "
        "Check the cleaned subscriber data before clustering."
    )

scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_data)

# Evaluate K from 2 to 8
kmeans_results = []

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    kmeans_results.append({
        "K": k,
        "Inertia": inertia,
        "Silhouette Score": silhouette
    })

kmeans_table = pd.DataFrame(kmeans_results)

print("\nK-means evaluation:")
print(kmeans_table.round(3).to_string(index=False))

# Choose K=4 using elbow, silhouette and business interpretability
BEST_K = 4

kmeans = KMeans(
    n_clusters=BEST_K,
    n_init=10,
    random_state=42
)

subscribers["segment_id"] = kmeans.fit_predict(X_scaled)

# ------------------------------------------------------------
# 3. PROFILE AND NAME THE SEGMENTS
# ------------------------------------------------------------

print("\n========== SEGMENT PROFILES ==========")

profile = subscribers.groupby("segment_id").agg(
    subscriber_count=("subscriber_id", "count"),
    mean_tenure=("tenure_months", "mean"),
    mean_weekly_hours=("hours_watched_per_week", "mean"),
    mean_monthly_recharge=("monthly_recharge_inr", "mean"),
    churn_rate=("churned", "mean")
).reset_index()

profile["churn_rate"] = profile["churn_rate"] * 100

print(profile.round(2).to_string(index=False))

# Name segments using their measured profiles.
# Churn is used only to describe/name clusters, not to build them.
at_risk_id = int(profile.loc[profile["churn_rate"].idxmax(), "segment_id"])

remaining = profile[profile["segment_id"] != at_risk_id]
premium_id = int(
    remaining.loc[
        remaining["mean_monthly_recharge"].idxmax(),
        "segment_id"
    ]
)

other_ids = [
    int(x) for x in profile["segment_id"]
    if int(x) not in [at_risk_id, premium_id]
]

# Give the remaining two clusters descriptive names by tenure.
other_profiles = (
    profile[profile["segment_id"].isin(other_ids)]
    .sort_values("mean_tenure")
)

segment_names = {
    at_risk_id: "At-Risk",
    premium_id: "Premium"
}

if len(other_profiles) >= 2:
    segment_names[int(other_profiles.iloc[0]["segment_id"])] = "New & Budget"
    segment_names[int(other_profiles.iloc[1]["segment_id"])] = "Established"

subscribers["segment_name"] = subscribers["segment_id"].map(segment_names)

profile["segment_name"] = profile["segment_id"].map(segment_names)

print("\nNamed segment profile:")
print(
    profile[
        [
            "segment_id",
            "segment_name",
            "subscriber_count",
            "mean_tenure",
            "mean_weekly_hours",
            "mean_monthly_recharge",
            "churn_rate"
        ]
    ].round(2).to_string(index=False)
)

# ------------------------------------------------------------
# 4. ASSIGN A NEW SUBSCRIBER BY HAND
# ------------------------------------------------------------

print("\n========== NEW SUBSCRIBER ASSIGNMENT ==========")

new_subscriber = pd.DataFrame([{
    "tenure_months": 4,
    "hours_watched_per_week": 15,
    "monthly_recharge_inr": 520
}])

new_scaled = scaler.transform(new_subscriber[cluster_features])

distances = np.linalg.norm(
    kmeans.cluster_centers_ - new_scaled,
    axis=1
)

nearest_cluster = int(np.argmin(distances))
predicted_cluster = int(kmeans.predict(new_scaled)[0])

print("Distances to centroids:")
for cluster_id, distance in enumerate(distances):
    print(
        f"Cluster {cluster_id} ({segment_names[cluster_id]}): "
        f"{distance:.4f}"
    )

print("Nearest cluster by hand:", nearest_cluster)
print("Nearest segment:", segment_names[nearest_cluster])
print("KMeans predict result:", predicted_cluster)

assert nearest_cluster == predicted_cluster, (
    "Manual nearest-centroid assignment does not match KMeans.predict."
)

# Save segments for all subscribers
segments_output = subscribers[
    ["subscriber_id", "segment_id", "segment_name"]
].copy()

segments_output.to_csv(
    OUTPUT_DIR / "segments.csv",
    index=False
)

print("\nSaved outputs/segments.csv")
print("Segment rows:", len(segments_output))

# ------------------------------------------------------------
# 5. CORRELATION MATRIX AND HIGH CORRELATION PAIRS
# ------------------------------------------------------------

print("\n========== CORRELATION ANALYSIS ==========")

correlation_features = [
    "hours_watched_per_week",
    "hd_hours_per_week",
    "sports_hours_per_week",
    "channels_watched"
]

correlation_matrix = subscribers[
    correlation_features
].corr()

print("\nCorrelation matrix:")
print(correlation_matrix.round(3))

print("\nPairs with absolute correlation above 0.9:")

for i in range(len(correlation_features)):
    for j in range(i + 1, len(correlation_features)):
        feature_a = correlation_features[i]
        feature_b = correlation_features[j]
        corr_value = correlation_matrix.loc[feature_a, feature_b]

        if abs(corr_value) > 0.9:
            print(
                f"{feature_a} - {feature_b}: {corr_value:.3f}"
            )

# ------------------------------------------------------------
# 6. PREPARE THE SPEND PREDICTION DATA
# ------------------------------------------------------------

print("\n========== SPEND PREDICTION ==========")

spend_features = [
    "monthly_recharge_inr",
    "tenure_months",
    "hours_watched_per_week",
    "hd_hours_per_week",
    "sports_hours_per_week",
    "channels_watched",
    "support_calls_90d"
]

# Only subscribers who did not churn are used for training/testing.
spend_data = subscribers[
    subscribers["churned"] == 0
].copy()

spend_data = spend_data.dropna(
    subset=spend_features + ["next_cycle_recharge_inr"]
)

X = spend_data[spend_features]
y = spend_data["next_cycle_recharge_inr"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Spend-model training rows:", len(X_train))
print("Spend-model testing rows:", len(X_test))

# ------------------------------------------------------------
# 7. LINEAR REGRESSION, RIDGE AND LASSO
# ------------------------------------------------------------

alpha_values = [
    0.001, 0.01, 0.1, 0.5, 1, 2, 3, 10, 30, 100
]

linear_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

lasso_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Lasso(max_iter=10000))
])

# Linear Regression
linear_pipeline.fit(X_train, y_train)

# Ridge Grid Search
ridge_search = GridSearchCV(
    ridge_pipeline,
    {"model__alpha": alpha_values},
    cv=5,
    scoring="r2",
    n_jobs=-1
)

ridge_search.fit(X_train, y_train)
best_ridge = ridge_search.best_estimator_

# Lasso Grid Search
lasso_search = GridSearchCV(
    lasso_pipeline,
    {"model__alpha": alpha_values},
    cv=5,
    scoring="r2",
    n_jobs=-1
)

lasso_search.fit(X_train, y_train)
best_lasso = lasso_search.best_estimator_

# ------------------------------------------------------------
# 8. MODEL EVALUATION
# ------------------------------------------------------------

def evaluate_regression(model, X_test, y_test):
    """Calculate R-squared, MAE and RMSE for a regression model."""
    predictions = model.predict(X_test)

    return {
        "Test R2": r2_score(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions))
    }


linear_metrics = evaluate_regression(
    linear_pipeline, X_test, y_test
)

ridge_metrics = evaluate_regression(
    best_ridge, X_test, y_test
)

lasso_metrics = evaluate_regression(
    best_lasso, X_test, y_test
)

linear_cv = np.mean(
    # Cross-validation is run only on the training rows.
    __import__("sklearn.model_selection", fromlist=["cross_val_score"])
    .cross_val_score(
        linear_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="r2"
    )
)

model_results = pd.DataFrame([
    {
        "Model": "Linear Regression",
        "Best Alpha": "Not applicable",
        "CV R2": linear_cv,
        **linear_metrics
    },
    {
        "Model": "Ridge",
        "Best Alpha": ridge_search.best_params_["model__alpha"],
        "CV R2": ridge_search.best_score_,
        **ridge_metrics
    },
    {
        "Model": "Lasso",
        "Best Alpha": lasso_search.best_params_["model__alpha"],
        "CV R2": lasso_search.best_score_,
        **lasso_metrics
    }
])

print("\nRegression model comparison:")
print(model_results.round(4).to_string(index=False))

print("\nBest Ridge alpha:", ridge_search.best_params_["model__alpha"])
print("Best Lasso alpha:", lasso_search.best_params_["model__alpha"])

# Save best Ridge model
joblib.dump(best_ridge, OUTPUT_DIR / "spend_model.joblib")
print("\nSaved outputs/spend_model.joblib")

# ------------------------------------------------------------
# 9. LASSO ZERO COEFFICIENTS
# ------------------------------------------------------------

print("\n========== LASSO FEATURE SELECTION ==========")

lasso_coefficients = best_lasso.named_steps["model"].coef_

lasso_coefficient_table = pd.DataFrame({
    "Feature": spend_features,
    "Coefficient": lasso_coefficients
})

print(lasso_coefficient_table.to_string(index=False))

zero_features = lasso_coefficient_table.loc[
    lasso_coefficient_table["Coefficient"] == 0,
    "Feature"
].tolist()

print("\nFeatures with exactly zero Lasso coefficients:")
print(zero_features)

# ------------------------------------------------------------
# 10. STABILITY EXPERIMENT
# ------------------------------------------------------------

print("\n========== STABILITY EXPERIMENT ==========")

linear_hours_coefficients = []
ridge_hours_coefficients = []

for random_state in range(5):
    X_sample = X_train.sample(
        frac=0.8,
        random_state=random_state
    )

    y_sample = y_train.loc[X_sample.index]

    # Linear Regression
    linear_sample = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    linear_sample.fit(X_sample, y_sample)

    linear_hours_coefficients.append(
        linear_sample.named_steps["model"].coef_[
            spend_features.index("hours_watched_per_week")
        ]
    )

    # Ridge with alpha=30
    ridge_sample = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=30))
    ])

    ridge_sample.fit(X_sample, y_sample)

    ridge_hours_coefficients.append(
        ridge_sample.named_steps["model"].coef_[
            spend_features.index("hours_watched_per_week")
        ]
    )

linear_stability = np.std(
    linear_hours_coefficients,
    ddof=0
)

ridge_stability = np.std(
    ridge_hours_coefficients,
    ddof=0
)

print("Linear Regression coefficients:", linear_hours_coefficients)
print("Ridge coefficients:", ridge_hours_coefficients)
print("Linear Regression coefficient std:", round(linear_stability, 4))
print("Ridge coefficient std:", round(ridge_stability, 4))

# ------------------------------------------------------------
# 11. REVENUE AT RISK
# ------------------------------------------------------------

print("\n========== REVENUE AT RISK ==========")

# Attach segment information to Part B test scores.
test_data = test_scores.merge(
    subscribers[
        [
            "subscriber_id",
            "segment_id",
            "segment_name",
            "next_cycle_recharge_inr"
        ]
    ],
    on="subscriber_id",
    how="left",
    validate="one_to_one"
)

if test_data["segment_name"].isna().any():
    raise ValueError("Some Part B test subscribers have no segment assignment.")

# Predict next-cycle recharge for every Part B test subscriber,
# including subscribers whose actual outcome is churned.
X_priority = subscribers.set_index("subscriber_id").loc[
    test_data["subscriber_id"],
    spend_features
]

if X_priority.isna().any().any():
    raise ValueError(
        "Missing spend-model input values in Part B test subscribers."
    )

predicted_recharge = best_ridge.predict(X_priority)

test_data["predicted_next_recharge_inr"] = predicted_recharge

test_data["revenue_at_risk_inr"] = (
    test_data["churn_probability"]
    * test_data["predicted_next_recharge_inr"]
)

# Select top 15 subscribers by revenue at risk.
priority_list = test_data.sort_values(
    "revenue_at_risk_inr",
    ascending=False
).head(15).copy()

priority_columns = [
    "subscriber_id",
    "churn_probability",
    "predicted_next_recharge_inr",
    "revenue_at_risk_inr",
    "segment_name",
    "churned"
]

priority_list = priority_list[priority_columns]

priority_list.to_csv(
    OUTPUT_DIR / "retention_priority.csv",
    index=False
)

print("\nTop 15 revenue-at-risk subscribers:")
print(priority_list.round(2).to_string(index=False))

total_revenue_at_risk = priority_list[
    "revenue_at_risk_inr"
].sum()

actual_churners = int(priority_list["churned"].sum())

print("\nTotal revenue at risk for top 15: INR",
      round(total_revenue_at_risk, 2))

print("Number of actual churners in top 15:", actual_churners)

# ------------------------------------------------------------
# 12. FINAL OUTPUT CHECK
# ------------------------------------------------------------

print("\n========== PART C COMPLETED ==========")

print("Created files:")
for filename in [
    "segments.csv",
    "spend_model.joblib",
    "retention_priority.csv"
]:
    path = OUTPUT_DIR / filename
    print(filename, "->", "EXISTS" if path.exists() else "MISSING")

print("\nPart C completed successfully!")
