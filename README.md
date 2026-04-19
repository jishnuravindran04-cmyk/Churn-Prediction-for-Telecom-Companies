# Churn-Prediction-for-Telecom-Companies
Predictive Analytics Project - Telecom Churn Prediction using XGBoost, Random Forest, and SHAP

Telecom companies face their biggest problem with customer churn. This project establishes an end to end Machine Learning pipeline which detects risk customers with high accuracy. 

There are 10 stages in this project:
1. Problem Definition.
2. Data Understanding.
3. Data Preprocessing.
4. EDA.
5. Feature Engineering.
6. Model Building.
7. Evaluation.
8. Interpretability.
9. Deployment.
10. Documentation.

##  Project Structure

```text
.
├── data/                  # Raw and Cleaned datasets (Git-ignored)
├── notebooks/
│   └── main_code.ipynb    # Main implementation file
├── results/               # Saved EDA plots and Model metrics
├── .gitignore             # Prevents clutter (venv, csv, etc.)
└── README.md              # Project documentation

# Data Preprocessing

Data Preprocessing is the process of transforming raw, "messy" data into a clean format that a machine learning model can understand. In your project, this involved converting the TotalCharges column to a numeric format and filling missing values for new customers with 0.0. It also included Feature Engineering, where you created new variables like ServiceCount and ContractRisk to give the model better predictive power
