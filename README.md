# Tata Play Retention & Support Copilot

## 1. Overview

The **Tata Play Retention & Support Copilot** is a data-driven customer retention and support automation project designed to analyze subscriber behaviour, predict customer churn, identify revenue at risk, and assist customer support teams.

The project combines data cleaning, SQL analytics, machine learning, customer segmentation, ticket classification, information extraction, and Retrieval-Augmented Generation (RAG).

### Key Objectives

- Analyze subscriber and recharge data.
- Identify customers who are likely to discontinue their subscriptions.
- Predict customer churn using machine learning.
- Segment customers based on their behaviour and spending.
- Estimate revenue at risk.
- Automate customer support ticket classification and information extraction.
- Generate retention recommendations using an AI-powered support copilot.

### Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- SQLite and SQL
- Joblib
- LangChain
- ChromaDB
- Sentence Transformers
- Pydantic
- Google Colab
- GitHub

---

## 2. Reproduce

### Requirements

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### Project Structure

```text
Tata-Play-Retention-Copilot/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── generate_data.py
├── part_a.py
├── part_b.py
├── part_c.py
├── part_d.py
│
├── copilot.py
├── mock_llm.py
├── llm_client.py
│
├── data/
│   ├── raw/
│   └── clean/
│
├── outputs/
│
└── sql/
    └── queries.sql
```

### Execution Order

Execute the project in the following order:

1. Generate or prepare the datasets.
2. Run Part A for data cleaning and SQL analysis.
3. Run Part B for churn prediction.
4. Run Part C for customer segmentation and revenue analysis.
5. Run Part D for ticket classification and support automation.
6. Run the Copilot module to generate retention briefs.

The project was developed and executed using Google Colab.

---

## 3. Part A – Data Cleaning and SQL Analysis

Part A establishes the data foundation for the project.

### Tasks Performed

- Audited raw subscriber and recharge datasets.
- Cleaned subscriber records.
- Cleaned recharge records.
- Handled missing and duplicate values.
- Prepared clean CSV datasets.
- Created an SQLite database.
- Executed SQL queries for customer analysis.

### Results

| Metric | Result |
|---|---:|
| Raw subscribers | 2,415 |
| Clean subscribers | 2,400 |
| Raw recharge records | 13,055 |
| Clean recharge records | 12,985 |
| Churned subscribers | 514 |
| Churn rate | 21.42% |
| Successful recharge revenue | ₹64
