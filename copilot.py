
import os
import sys
import glob
import json
import pandas as pd

sys.path.insert(0, "/content")
from llm_client import call_llm

ROOT = "/content"
DATA = f"{ROOT}/data"
OUTPUT = f"{ROOT}/outputs"

os.makedirs(OUTPUT, exist_ok=True)


def find_column(df, names):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    raise ValueError(
        f"Missing column {names}. Available: {list(df.columns)}"
    )


# Load existing project outputs
subscribers = pd.read_csv(f"{DATA}/clean/subscribers.csv")
priority = pd.read_csv(f"{OUTPUT}/retention_priority.csv")
recharges = pd.read_csv(f"{DATA}/clean/recharges.csv")

sid_sub = find_column(subscribers, ["subscriber_id"])
sid_priority = find_column(priority, ["subscriber_id"])
sid_recharge = find_column(recharges, ["subscriber_id"])

tenure_col = find_column(subscribers, ["tenure_months"])
prob_col = find_column(
    priority,
    ["churn_probability", "predicted_churn_probability", "probability"]
)

status_col = find_column(recharges, ["status", "recharge_status"])

# Count failed recharges
failed = recharges[
    recharges[status_col].astype(str).str.lower().isin(
        ["failed", "failure"]
    )
]

failed_counts = failed.groupby(sid_recharge).size().to_dict()

# Select top 15 subscribers
priority = priority.sort_values(prob_col, ascending=False)
top15 = priority.head(15)

subscriber_lookup = subscribers.set_index(sid_sub).to_dict("index")

# Load policy files
policy_paths = sorted(glob.glob(f"{DATA}/policies/*.txt"))

if not policy_paths:
    raise FileNotFoundError("Policy files not found.")

policy_context = ""

for path in policy_paths:
    with open(path, "r", encoding="utf-8") as f:
        policy_context += (
            f"\n[{os.path.basename(path)}]\n{f.read()}\n"
        )

system_prompt = """
RETENTION BRIEF.
Use only the supplied subscriber facts and POLICY CONTEXT.
Return only a valid JSON object with:
subscriber_id, risk_band, recommended_action, policy_source.
Do not invent policy details.
"""

briefs = []

for _, row in top15.iterrows():

    sid = str(row[sid_priority])
    subscriber = subscriber_lookup.get(sid, {})

    tenure = int(float(subscriber.get(tenure_col, 0)))
    failed_count = int(failed_counts.get(sid, 0))

    probability = float(row[prob_col])

    risk_band = "High" if probability >= 0.5 else "Low"

    user_message = (
        f"subscriber_id: {sid}\n"
        f"risk_band: {risk_band}\n"
        f"tenure_months: {tenure}\n"
        f"failed_recharges_6m: {failed_count}\n"
        f"POLICY CONTEXT:\n{policy_context}"
    )

    answer = call_llm(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0,
        max_tokens=250
    )

    try:
        result = json.loads(answer)
    except json.JSONDecodeError:
        result = {
            "subscriber_id": sid,
            "risk_band": risk_band,
            "recommended_action": "Manual review required.",
            "policy_source": "none",
            "raw_response": answer
        }

    result["tenure_months"] = tenure
    result["failed_recharges_6m"] = failed_count
    result["churn_probability"] = probability

    briefs.append(result)


with open(
    f"{OUTPUT}/retention_briefs.jsonl",
    "w",
    encoding="utf-8"
) as f:
    for brief in briefs:
        f.write(json.dumps(brief, ensure_ascii=False) + "\n")

print("Retention briefs generated successfully!")
print("Total subscribers:", len(briefs))
print(json.dumps(briefs[0], indent=2) if briefs else "No records generated.")
