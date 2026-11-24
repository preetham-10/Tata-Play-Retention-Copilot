# Tata Play Retention & Support Copilot

## 1. Project Introduction

The Tata Play Retention & Support Copilot is an AI-based system designed to help telecom companies understand customer problems, predict customer churn, and improve customer retention.

The system analyzes subscriber information, recharge history, and customer support tickets. It uses machine learning to identify customers who may leave the service and AI-based techniques to classify support tickets and retrieve relevant company policies.

It also generates personalized retention briefs with recommended actions to help customer support teams retain subscribers.

## 2. Project Objectives

- Analyze subscriber and recharge data.
- Predict customers who are likely to discontinue the service.
- Automatically classify customer support tickets.
- Extract important information from support tickets.
- Retrieve relevant company policies using RAG.
- Generate personalized retention recommendations.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas and NumPy | Data processing |
| SQL and SQLite | Database management and queries |
| Scikit-learn | Machine learning |
| Sentence Transformers | Text embeddings |
| ChromaDB | Vector database |
| RAG | Policy retrieval and answering |
| Google Colab | Project execution |
| GitHub | Project source code and documentation |

## 4. Project Modules

### Part A – Data Processing and SQL Analysis

- Loads subscriber, plan, recharge and ticket datasets.
- Cleans and preprocesses the data.
- Creates a SQLite database.
- Performs SQL queries to analyze revenue, recharge history and subscriber information.

### Part B – Machine Learning

- Uses subscriber-related features to predict churn.
- Trains machine learning models.
- Evaluates model performance.
- Identifies customers with a higher risk of leaving.

### Part C – Ticket Classification and Extraction

- Classifies customer tickets into categories such as billing, cancellation and signal.
- Uses zero-shot and few-shot classification.
- Extracts important information from tickets.
- Identifies tickets that require manual review.

### Part D – Retrieval-Augmented Generation (RAG)

- Retrieves relevant company policy documents.
- Uses semantic search to find relevant information.
- Generates answers based on retrieved policies.
- Evaluates whether answers contain expected facts.

### Copilot – Retention Brief Generation

- Identifies high-risk subscribers.
- Generates personalized retention briefs.
- Recommends actions based on subscriber information and company policies.

## 5. System Workflow

1. Collect subscriber, recharge, plan and ticket data.
2. Clean and preprocess the datasets.
3. Perform data analysis using Python and SQL.
4. Train machine learning models for churn prediction.
5. Classify customer support tickets.
6. Retrieve relevant policies using RAG.
7. Generate retention briefs and recommended actions.

## 6. Expected Outcomes

- Identification of customers at risk of leaving.
- Faster understanding of customer complaints.
- Relevant policy-based responses.
- Personalized customer retention recommendations.
- Better support-team decision-making.

## 7. Conclusion

The Tata Play Retention & Support Copilot combines data analytics, machine learning, ticket classification and Retrieval-Augmented Generation into one integrated system.

It helps customer support teams identify at-risk subscribers, understand customer complaints, retrieve relevant policy information and take proactive steps to improve customer retention.

## 8. Project Execution

The project was tested in Google Colab using the public GitHub repository.

The following components were executed successfully:

- Part A – Data Processing and SQL Analysis
- Part B – Machine Learning
- Part C – Ticket Classification and Extraction
- Part D – RAG Evaluation
- Copilot – Retention Brief Generation

## 9. Repository

GitHub: https://github.com/preetham-10/Tata-Play-Retention-Copilot

## 10. Disclaimer

This project is an academic prototype. Its predictions and recommendations are intended for demonstration and decision-support purposes and should be validated before real-world deployment.
