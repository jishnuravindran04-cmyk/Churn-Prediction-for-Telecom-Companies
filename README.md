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
```text
| Property               | Value                                                                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset Name**       | IBM Watson Telco Churn                                                                                                           |
| **Source**             | [Kaggle – IBM Watson Telco Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?utm_source=chatgpt.com) |
| **Rows**               | 7,043 customers                                                                                                                  |
| **Columns**            | 21 raw features → 32 features after encoding                                                                                     |
| **Target Variable**    | **Churn** — Yes (1) / No (0)                                                                                                     |
| **Class Distribution** | 73.5% No Churn · 26.5% Churned                                                                                                   |

```
##  Project Structure

```text
.
├── data/                  # Raw and Cleaned datasets (Git-ignored)
├── notebooks/
│   └── main_code.ipynb    # Main implementation file
├── results/               # Saved EDA plots and Model metrics
├── .gitignore             # Prevents clutter (venv, csv, etc.)
└── README.md              # Project documentation
```
                                                                                                   |
### Data Preprocessing
Data Preprocessing is the essential cleaning and refinement phase of the project. During this stage, I transformed raw, inconsistent data into a machine-learning-ready format by converting `TotalCharges` into a numeric type and handling missing values for new customers. Additionally, I performed feature engineering by creating new metrics like `ServiceCount` and `ContractRisk`, which allowed the model to capture deeper business insights that weren't present in the initial dataset.

### Exploratory Data Analysis - EDA
EDA functioned as the investigative heart of the project, where I used visual storytelling to uncover the "why" behind customer behavior. By plotting churn rates against various categories, I identified that customers on month-to-month contracts and those with high monthly charges are at the highest risk of leaving. This phase was crucial for validating our features and ensuring that the patterns discovered visually aligned with the logical expectations of the telecom industry.



### SMOTE 
SMOTE (Synthetic Minority Over-sampling Technique) was implemented to solve the critical problem of class imbalance, where "staying" customers significantly outnumbered "churning" customers. Without this, the model would likely ignore churners entirely to maintain a high (but misleading) accuracy score. By generating synthetic data points for the minority churn class, I balanced the training set to a 50/50 split, forcing the model to become more sensitive and accurate in predicting when a customer is actually about to leave.
