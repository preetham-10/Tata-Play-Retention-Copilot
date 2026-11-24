
import os
import re
import json
import glob
import sys
import pandas as pd
import chromadb

from typing import Optional, Literal
from pydantic import BaseModel, Field, ValidationError
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

sys.path.insert(0, "/content")
from llm_client import call_llm

ROOT = "/content"
DATA = f"{ROOT}/data"
OUTPUT = f"{ROOT}/outputs"
POLICY_DIR = f"{DATA}/policies"

os.makedirs(OUTPUT, exist_ok=True)


# ---------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------

def read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def find_column(df, possible_names, required=True):
    lookup = {str(c).strip().lower(): c for c in df.columns}

    for name in possible_names:
        if name.lower() in lookup:
            return lookup[name.lower()]

    if required:
        raise ValueError(
            f"Could not find any of {possible_names}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def response_text(messages, max_tokens=300):
    return call_llm(messages, temperature=0, max_tokens=max_tokens)


# ---------------------------------------------------------
# 1. LOAD TICKETS AND FEW-SHOT EXAMPLES
# ---------------------------------------------------------

tickets = read_csv(f"{DATA}/raw/tickets.csv")
examples = read_csv(f"{DATA}/raw/few_shot_examples.csv")

ticket_text_col = find_column(tickets, ["text", "ticket_text", "message"])
ticket_id_col = find_column(tickets, ["ticket_id"], required=False)
subscriber_col = find_column(tickets, ["subscriber_id"], required=False)
true_label_col = find_column(
    tickets, ["true_category", "category", "label"], required=False
)

example_text_col = find_column(
    examples, ["text", "ticket_text", "ticket", "message"]
)
example_label_col = find_column(
    examples, ["label", "category", "true_category"]
)

few_shot_text = "\n\n".join(
    f"Ticket: {row[example_text_col]}\nLabel: {row[example_label_col]}"
    for _, row in examples.iterrows()
)


# ---------------------------------------------------------
# 2. TICKET CLASSIFICATION
# ---------------------------------------------------------

def classify_ticket(ticket, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(ticket)}
    ]
    return response_text(messages, max_tokens=100).strip()


zero_shot_system = """
TICKET CLASSIFIER.
You classify Tata Play customer support tickets.
Valid labels are: billing, signal, plan_change, cancellation.
Respond with only the label. Do not explain.
"""

few_shot_system = f"""
TICKET CLASSIFIER.
You classify Tata Play customer support tickets.
Valid labels are: billing, signal, plan_change, cancellation.

Use these labelled examples:
{few_shot_text}

Respond with only the label. Do not explain.
"""

weak_system = """
TICKET CLASSIFIER.
Classify the customer's support ticket.
"""


classification_rows = []

for index, row in tickets.iterrows():
    ticket = str(row[ticket_text_col]).strip()

    ticket_id = (
        row[ticket_id_col]
        if ticket_id_col
        else f"TICKET_{index + 1:03d}"
    )

    subscriber_id = (
        row[subscriber_col]
        if subscriber_col
        else None
    )

    true_category = (
        row[true_label_col]
        if true_label_col
        else None
    )

    if ticket in ("", "...", "nan", "None"):
        classification_rows.append({
            "ticket_id": ticket_id,
            "subscriber_id": subscriber_id,
            "ticket_text": ticket,
            "true_category": true_category,
            "zero_shot_prediction": None,
            "few_shot_prediction": None,
            "weak_prompt_response": None,
            "manual_review": True
        })
        continue

    zero = classify_ticket(ticket, zero_shot_system)
    few = classify_ticket(ticket, few_shot_system)
    weak = classify_ticket(ticket, weak_system)

    valid_labels = {
        "billing", "signal", "plan_change", "cancellation"
    }

    zero = zero.lower().strip()
    few = few.lower().strip()

    classification_rows.append({
        "ticket_id": ticket_id,
        "subscriber_id": subscriber_id,
        "ticket_text": ticket,
        "true_category": true_category,
        "zero_shot_prediction": zero if zero in valid_labels else None,
        "few_shot_prediction": few if few in valid_labels else None,
        "weak_prompt_response": weak,
        "manual_review": False
    })

classification_df = pd.DataFrame(classification_rows)
classification_df.to_csv(
    f"{OUTPUT}/ticket_classification.csv", index=False
)

print("Ticket classification completed.")
print(classification_df.head())


# ---------------------------------------------------------
# 3. STRUCTURED TICKET EXTRACTION WITH PYDANTIC
# ---------------------------------------------------------

class TicketRecord(BaseModel):
    subscriber_id: Optional[str] = None
    category: Literal[
        "billing", "signal", "plan_change", "cancellation"
    ]
    urgency: int = Field(ge=1, le=5)
    wants_refund: bool


extractor_system = """
TICKET EXTRACTOR.
Extract the ticket information and return ONLY a valid JSON object.

Required fields:
subscriber_id: subscriber ID such as TP00001, or null
category: billing, signal, plan_change, or cancellation
urgency: integer from 1 to 5
wants_refund: true or false

Do not add explanations or Markdown.
"""


def extract_ticket(ticket, max_retries=3):
    messages = [
        {"role": "system", "content": extractor_system},
        {"role": "user", "content": str(ticket)}
    ]

    last_error = None

    for attempt in range(max_retries):
        raw = response_text(messages, max_tokens=200)

        try:
            # Validate JSON structure and field constraints.
            record = TicketRecord.model_validate_json(raw)
            return record.model_dump(), raw, attempt + 1

        except Exception as exc:
            last_error = str(exc)

            messages.append({
                "role": "user",
                "content": (
                    "VALIDATION ERROR: "
                    + last_error
                    + ". Return a corrected JSON object only. "
                    "Use valid category values and urgency from 1 to 5."
                )
            })

    return None, str(last_error), max_retries


extraction_rows = []

for index, row in tickets.iterrows():
    ticket = str(row[ticket_text_col]).strip()

    ticket_id = (
        row[ticket_id_col]
        if ticket_id_col
        else f"TICKET_{index + 1:03d}"
    )

    if ticket in ("", "...", "nan", "None"):
        extraction_rows.append({
            "ticket_id": ticket_id,
            "ticket_text": ticket,
            "status": "manual_review",
            "attempts": 0,
            "extraction": None,
            "error": "Unreadable or empty ticket"
        })
        continue

    record, raw, attempts = extract_ticket(ticket)

    extraction_rows.append({
        "ticket_id": ticket_id,
        "ticket_text": ticket,
        "status": "success" if record else "manual_review",
        "attempts": attempts,
        "extraction": record,
        "error": None if record else raw
    })


with open(f"{OUTPUT}/ticket_extractions.jsonl", "w", encoding="utf-8") as file:
    for item in extraction_rows:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Ticket extraction completed.")
print("Successful extractions:", sum(
    x["status"] == "success" for x in extraction_rows
))
print("Manual review:", sum(
    x["status"] == "manual_review" for x in extraction_rows
))


# ---------------------------------------------------------
# 4. LOAD POLICY DOCUMENTS
# ---------------------------------------------------------

policy_paths = sorted(glob.glob(f"{POLICY_DIR}/*.txt"))

if not policy_paths:
    raise FileNotFoundError(
        f"No policy text files found in {POLICY_DIR}"
    )

policy_documents = []

for path in policy_paths:
    with open(path, "r", encoding="utf-8") as file:
        content = file.read()

    policy_documents.append({
        "filename": os.path.basename(path),
        "content": content
    })


# ---------------------------------------------------------
# 5. CREATE CHROMA VECTOR DATABASE
# ---------------------------------------------------------

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

chroma_client = chromadb.Client()

try:
    chroma_client.delete_collection("tataplay_policy_collection")
except Exception:
    pass

collection = chroma_client.create_collection(
    name="tataplay_policy_collection",
    embedding_function=embedding_function
)

collection.add(
    ids=[f"policy_{i}" for i in range(len(policy_documents))],
    documents=[
        f"[{doc['filename']}]\n{doc['content']}"
        for doc in policy_documents
    ],
    metadatas=[
        {"filename": doc["filename"]}
        for doc in policy_documents
    ]
)


def retrieve_policy_context(question, top_k=5):
    count = min(top_k, len(policy_documents))

    result = collection.query(
        query_texts=[str(question)],
        n_results=count
    )

    docs = result["documents"][0]
    metadata = result["metadatas"][0]

    context = "\n\n".join(docs)
    filenames = [m["filename"] for m in metadata]

    return context, filenames


# ---------------------------------------------------------
# 6. POLICY QUESTION ANSWERING / RAG EVALUATION
# ---------------------------------------------------------

policy_eval = read_csv(f"{DATA}/raw/policy_eval.csv")

question_col = find_column(
    policy_eval, ["question", "query", "user_question"]
)

expected_file_col = find_column(
    policy_eval,
    ["expected_policy_file", "policy_file", "expected_file", "source_file"],
    required=False
)

expected_fact_col = find_column(
    policy_eval,
    ["expected_fact", "expected_answer", "expected_text", "answer"],
    required=False
)

rag_rows = []

policy_system = """
POLICY ASSISTANT.
Answer using only the retrieved context.
Do not use outside knowledge or invent policy details.
If the answer is not available in the context, clearly say:
I don't have enough information to answer that.
"""


for _, row in policy_eval.iterrows():
    question = str(row[question_col])

    expected_file = (
        row[expected_file_col]
        if expected_file_col else None
    )

    expected_fact = (
        row[expected_fact_col]
        if expected_fact_col else None
    )

    context, retrieved_files = retrieve_policy_context(question)

    user_message = (
        "CONTEXT:\n"
        + context
        + "\n\nQUESTION:\n"
        + question
    )

    answer = response_text([
        {"role": "system", "content": policy_system},
        {"role": "user", "content": user_message}
    ], max_tokens=250)

    expected_fact_text = (
        str(expected_fact)
        if pd.notna(expected_fact) and expected_fact is not None
        else ""
    )

    contains_fact = (
        expected_fact_text.strip().lower() in answer.lower()
        if expected_fact_text.strip()
        else None
    )

    rag_rows.append({
        "question": question,
        "expected_policy_file": expected_file,
        "retrieved_policy_files": "; ".join(retrieved_files),
        "answer": answer,
        "expected_fact": expected_fact,
        "answer_contains_expected_fact": contains_fact,
        "abstained": "don't have enough information" in answer.lower()
    })


rag_df = pd.DataFrame(rag_rows)
rag_df.to_csv(f"{OUTPUT}/rag_eval.csv", index=False)

print("RAG evaluation completed.")
print(rag_df.head())
