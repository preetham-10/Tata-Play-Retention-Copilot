"""
Part A - Data Foundation
Tata Play Subscriber Retention & Support Copilot

Tasks:
1. Audit raw datasets
2. Clean subscribers
3. Clean recharges
4. Save clean CSV files
5. Create SQLite database
6. Execute four SQL queries
"""

import os
import sqlite3
import pandas as pd
from google.colab import files
uploaded = files.upload()
import zipfile
zip_file = next(name for name in uploaded if name.endswith(".zip"))

with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall("/content")

print("Files extracted successfully!")
import os

for root, dirs, files_list in os.walk("/content"):
    if "subscribers.csv" in files_list:
        print(os.path.join(root, "subscribers.csv"))

# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"
OUTPUT_DIR = "outputs"
SQL_DIR = "sql"

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SQL_DIR, exist_ok=True)

SUBSCRIBERS_FILE = os.path.join(RAW_DIR, "subscribers.csv")
RECHARGES_FILE = os.path.join(RAW_DIR, "recharges.csv")
PLANS_FILE = os.path.join(RAW_DIR, "plans.csv")

DB_FILE = "data/tataplay.db"


# --------------------------------------------------
# 2. LOAD RAW DATA
# --------------------------------------------------

print("\n========== PART A: DATA FOUNDATION ==========")

subscribers = pd.read_csv(SUBSCRIBERS_FILE)
recharges = pd.read_csv(RECHARGES_FILE)
plans = pd.read_csv(PLANS_FILE)

print("\nRaw datasets loaded successfully.")


# --------------------------------------------------
# 3. AUDIT RAW DATA
# --------------------------------------------------

def audit_dataset(df, name):
    print(f"\n========== {name.upper()} AUDIT ==========")

    print("\nShape:")
    print(df.shape)

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    missing = pd.DataFrame({
        "Missing_Count": df.isna().sum(),
        "Missing_Percentage": (
            df.isna().sum() / len(df) * 100
        ).round(2)
    })

    print(missing)

    print("\nExact duplicate rows:")
    print(df.duplicated().sum())


audit_dataset(subscribers, "Subscribers")
audit_dataset(recharges, "Recharges")


# Additional subscriber audit figures

print("\n========== SUBSCRIBER AUDIT CHECKS ==========")

print("Total subscriber rows:", len(subscribers))
print("Distinct subscriber IDs:", subscribers["subscriber_id"].nunique())
print("Duplicate subscriber rows:", subscribers.duplicated().sum())

print(
    "Blank age:",
    subscribers["age"].isna().sum()
)

age_numeric = pd.to_numeric(
    subscribers["age"], errors="coerce"
)

print(
    "Age values outside 10-100:",
    ((age_numeric < 10) | (age_numeric > 100)).sum()
)

print(
    "Blank payment mode:",
    subscribers["payment_mode"].isna().sum()
)

print(
    "Distinct raw city spellings:",
    subscribers["city"].nunique()
)

print(
    "Distinct raw auto-renew spellings:",
    subscribers["auto_renew"].nunique()
)

print(
    "Blank next-cycle recharge:",
    subscribers["next_cycle_recharge_inr"].isna().sum()
)


# Additional recharge audit figures

print("\n========== RECHARGE AUDIT CHECKS ==========")

print("Total recharge rows:", len(recharges))
print("Duplicate recharge rows:", recharges.duplicated().sum())
print("Distinct recharge IDs:", recharges["recharge_id"].nunique())

print(
    "Distinct raw status spellings:",
    recharges["status"].nunique()
)

amount_text = recharges["amount_inr"].astype(str).str.strip()

not_plain_number = pd.to_numeric(
    amount_text, errors="coerce"
).isna()

print(
    "Amounts that are not plain numbers:",
    not_plain_number.sum()
)

print(
    "Recharge rows with unknown subscriber IDs:",
    (~recharges["subscriber_id"].isin(
        subscribers["subscriber_id"]
    )).sum()
)


# --------------------------------------------------
# 4. CLEAN SUBSCRIBERS
# --------------------------------------------------

print("\n========== CLEANING SUBSCRIBERS ==========")

# Remove exact duplicate rows
subscribers = subscribers.drop_duplicates().copy()

print("Rows after removing duplicates:", len(subscribers))


# Standardise city names

city_mapping = {
    "chennai": "Chennai",
    "madras": "Chennai",
    "mumbai": "Mumbai",
    "bombay": "Mumbai",
    "bengaluru": "Bengaluru",
    "bangalore": "Bengaluru",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "hyderabad": "Hyderabad",
    "hyd": "Hyderabad",
    "pune": "Pune",
    "kolkata": "Kolkata",
    "calcutta": "Kolkata"
}

subscribers["city"] = (
    subscribers["city"]
    .astype("string")
    .str.strip()
    .str.lower()
    .map(city_mapping)
)


# Standardise auto-renew values

subscribers["auto_renew"] = (
    subscribers["auto_renew"]
    .astype("string")
    .str.strip()
    .str.lower()
    .map({
        "yes": "Yes",
        "y": "Yes",
        "no": "No",
        "n": "No"
    })
)


# Convert age to numeric

subscribers["age"] = pd.to_numeric(
    subscribers["age"],
    errors="coerce"
)

# Replace impossible ages with missing values
invalid_age = (
    (subscribers["age"] < 10) |
    (subscribers["age"] > 100)
)

subscribers.loc[invalid_age, "age"] = pd.NA


# Convert numeric columns to numeric types

numeric_columns = [
    "tenure_months",
    "add_on_count",
    "monthly_recharge_inr",
    "hours_watched_per_week",
    "hd_hours_per_week",
    "sports_hours_per_week",
    "channels_watched",
    "support_calls_90d",
    "churned"
]

for column in numeric_columns:
    subscribers[column] = pd.to_numeric(
        subscribers[column],
        errors="coerce"
    )


# Display cleaned subscriber results

print("\nCity distribution:")
print(subscribers["city"].value_counts())

print("\nAuto-renew distribution:")
print(subscribers["auto_renew"].value_counts())

print("\nMissing age after cleaning:")
print(subscribers["age"].isna().sum())

print("\nMissing payment mode:")
print(subscribers["payment_mode"].isna().sum())

print("\nChurn distribution:")
print(subscribers["churned"].value_counts())

churn_percentage = (
    subscribers["churned"].mean() * 100
)

print(f"Churn percentage: {churn_percentage:.2f}%")


# --------------------------------------------------
# 5. CLEAN RECHARGES
# --------------------------------------------------

print("\n========== CLEANING RECHARGES ==========")

# Remove exact duplicate rows
recharges = recharges.drop_duplicates().copy()

print("After removing duplicates:", len(recharges))


# Remove recharge rows for unknown subscribers

valid_subscriber_ids = set(
    subscribers["subscriber_id"]
)

recharges = recharges[
    recharges["subscriber_id"].isin(valid_subscriber_ids)
].copy()

print("After removing unknown subscribers:", len(recharges))


# Standardise recharge status

recharges["status"] = (
    recharges["status"]
    .astype("string")
    .str.strip()
    .str.lower()
)


# Convert recharge amounts to numbers.
# Values written with an INR prefix are rebuilt from
# the subscriber's monthly recharge amount, as required.

raw_amount = recharges["amount_inr"].astype("string").str.strip()

has_inr_prefix = raw_amount.str.match(
    r"(?i)^INR\s*",
    na=False
)

amount_numeric = pd.to_numeric(
    raw_amount,
    errors="coerce"
)

# INR-prefixed text is not treated as a plain numeric amount.
amount_numeric.loc[has_inr_prefix] = pd.NA

# Also rebuild any other values that cannot be converted.
amount_numeric = amount_numeric.astype("float64")

recharges["amount_inr"] = amount_numeric


# Get each subscriber's monthly recharge amount

monthly_amounts = subscribers.set_index(
    "subscriber_id"
)["monthly_recharge_inr"]


# Rebuild invalid recharge amounts

rebuild_mask = recharges["amount_inr"].isna()

recharges.loc[rebuild_mask, "amount_inr"] = (
    recharges.loc[rebuild_mask, "subscriber_id"]
    .map(monthly_amounts)
)

print(
    "Recharge amounts rebuilt:",
    int(rebuild_mask.sum())
)


# Parse recharge dates

recharges["recharge_date"] = pd.to_datetime(
    recharges["recharge_date"],
    errors="coerce"
)

# Store dates in ISO format in the clean CSV
recharges["recharge_date"] = (
    recharges["recharge_date"].dt.strftime("%Y-%m-%d")
)


# Display recharge cleaning results

print("\nRecharge status counts:")
print(recharges["status"].value_counts())

successful_recharges = recharges[
    recharges["status"] == "success"
]

print(
    "\nTotal successful recharge amount:",
    successful_recharges["amount_inr"].sum()
)


# --------------------------------------------------
# 6. SAVE CLEAN CSV FILES
# --------------------------------------------------

print("\n========== SAVING CLEAN DATA ==========")

subscribers.to_csv(
    os.path.join(CLEAN_DIR, "subscribers.csv"),
    index=False
)

recharges.to_csv(
    os.path.join(CLEAN_DIR, "recharges.csv"),
    index=False
)

plans.to_csv(
    os.path.join(CLEAN_DIR, "plans.csv"),
    index=False
)

print("Clean CSV files saved successfully.")

print("Clean subscribers:", len(subscribers))
print("Clean recharges:", len(recharges))
print("Clean plans:", len(plans))


# --------------------------------------------------
# 7. CREATE SQLITE DATABASE
# --------------------------------------------------

print("\n========== CREATING SQLITE DATABASE ==========")

os.makedirs("data", exist_ok=True)

connection = sqlite3.connect(DB_FILE)

subscribers.to_sql(
    "subscribers",
    connection,
    if_exists="replace",
    index=False
)

recharges.to_sql(
    "recharges",
    connection,
    if_exists="replace",
    index=False
)

plans.to_sql(
    "plans",
    connection,
    if_exists="replace",
    index=False
)

print("SQLite database created:", DB_FILE)


# --------------------------------------------------
# 8. SQL QUERIES
# --------------------------------------------------

queries = {

    # Q1: Revenue by plan

    "Q1 Revenue by Plan": """
        SELECT
            p.plan_name,
            SUM(r.amount_inr) AS total_revenue,
            ROUND(AVG(r.amount_inr), 2) AS average_revenue,
            COUNT(DISTINCT s.subscriber_id) AS subscriber_count

        FROM recharges r

        INNER JOIN subscribers s
            ON r.subscriber_id = s.subscriber_id

        INNER JOIN plans p
            ON s.plan_id = p.plan_id

        WHERE LOWER(r.status) = 'success'

        GROUP BY p.plan_name

        ORDER BY total_revenue DESC;
    """,

    # Q2: Subscribers who never successfully recharged

    "Q2 Never Successfully Recharged": """
        SELECT
            s.subscriber_id,
            s.city,
            s.tenure_months

        FROM subscribers s

        LEFT JOIN recharges r
            ON s.subscriber_id = r.subscriber_id
            AND LOWER(r.status) = 'success'

        WHERE r.recharge_id IS NULL

        ORDER BY s.subscriber_id;
    """,

    # Q3: Ranking within each city

    "Q3 Ranking Within City": """
        WITH subscriber_revenue AS (
            SELECT
                s.subscriber_id,
                s.city,
                SUM(r.amount_inr) AS total_success

            FROM subscribers s

            INNER JOIN recharges r
                ON s.subscriber_id = r.subscriber_id

            WHERE LOWER(r.status) = 'success'

            GROUP BY s.subscriber_id, s.city
        ),

        ranked AS (
            SELECT
                subscriber_id,
                city,
                total_success,

                RANK() OVER (
                    PARTITION BY city
                    ORDER BY total_success DESC
                ) AS rank_value,

                DENSE_RANK() OVER (
                    PARTITION BY city
                    ORDER BY total_success DESC
                ) AS dense_rank_value,

                ROW_NUMBER() OVER (
                    PARTITION BY city
                    ORDER BY total_success DESC, subscriber_id
                ) AS row_number_value

            FROM subscriber_revenue
        )

        SELECT *
        FROM ranked
        ORDER BY city, row_number_value;
    """,

    # Q4: Running total for TP00010

    "Q4 Running Total": """
        SELECT
            subscriber_id,
            recharge_id,
            recharge_date,
            amount_inr,

            SUM(amount_inr) OVER (
                ORDER BY recharge_date, recharge_id
                ROWS BETWEEN UNBOUNDED PRECEDING
                AND CURRENT ROW
            ) AS running_total

        FROM recharges

        WHERE subscriber_id = 'TP00010'
          AND LOWER(status) = 'success'

        ORDER BY recharge_date, recharge_id;
    """
}


# --------------------------------------------------
# 9. SAVE SQL QUERIES TO FILE
# --------------------------------------------------

sql_file = os.path.join(SQL_DIR, "queries.sql")

with open(sql_file, "w", encoding="utf-8") as file:
    for query_name, query in queries.items():
        file.write(f"\n-- {query_name}\n")
        file.write(query)
        file.write("\n")

print("\nSQL queries saved to:", sql_file)


# --------------------------------------------------
# 10. EXECUTE SQL QUERIES
# --------------------------------------------------

print("\n========== EXECUTING SQL QUERIES ==========")

for query_name, query in queries.items():

    print(f"\n========== {query_name} ==========")

    result = pd.read_sql_query(
        query,
        connection
    )

    print(result.to_string(index=False))

    # Save individual results for inspection
    filename = (
        query_name.lower()
        .replace(" ", "_")
        .replace(":", "")
        + ".csv"
    )

    result.to_csv(
        os.path.join(OUTPUT_DIR, filename),
        index=False
    )


# --------------------------------------------------
# 11. Q1 CROSS-CHECK USING PANDAS GROUPBY
# --------------------------------------------------

print("\n========== PANDAS GROUPBY CROSS-CHECK ==========")

successful = recharges[
    recharges["status"] == "success"
].copy()

pandas_check = (
    successful
    .merge(
        subscribers[["subscriber_id", "plan_id"]],
        on="subscriber_id",
        how="inner"
    )
    .merge(
        plans[["plan_id", "plan_name"]],
        on="plan_id",
        how="inner"
    )
    .groupby("plan_name")
    .agg(
        total_revenue=("amount_inr", "sum"),
        average_revenue=("amount_inr", "mean"),
        subscriber_count=("subscriber_id", "nunique")
    )
    .reset_index()
)

pandas_check["average_revenue"] = (
    pandas_check["average_revenue"].round(2)
)

pandas_check = pandas_check.sort_values(
    "total_revenue",
    ascending=False
)

print(pandas_check.to_string(index=False))


# --------------------------------------------------
# 12. Q3 RANK = 1 COUNT PER CITY
# --------------------------------------------------

print("\n========== RANK 1 COUNT PER CITY ==========")

ranked_result = pd.read_sql_query(
    queries["Q3 Ranking Within City"],
    connection
)

rank_one_counts = (
    ranked_result[
        ranked_result["rank_value"] == 1
    ]
    .groupby("city")
    .size()
    .reset_index(name="subscribers_with_rank_1")
)

print(rank_one_counts.to_string(index=False))


# Kolkata first four rows

print("\n========== KOLKATA TOP FOUR ==========")

kolkata_top_four = ranked_result[
    ranked_result["city"] == "Kolkata"
].head(4)

print(kolkata_top_four.to_string(index=False))


# --------------------------------------------------
# 13. CLOSE DATABASE
# --------------------------------------------------

connection.close()

print("\n========== PART A COMPLETED ==========")
print("Clean CSV files, SQLite database and SQL queries created.")