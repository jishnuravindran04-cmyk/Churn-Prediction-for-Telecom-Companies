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

| Property               | Value                                                                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset Name**       | IBM Watson Telco Churn                                                                                                           |
| **Source**             | [Kaggle – IBM Watson Telco Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?utm_source=chatgpt.com) |
| **Rows**               | 7,043 customers                                                                                                                  |
| **Columns**            | 21 raw features → 32 features after encoding                                                                                     |
| **Target Variable**    | **Churn** — Yes (1) / No (0)                                                                                                     |
| **Class Distribution** | 73.5% No Churn · 26.5% Churned                                                                                                   |

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
### Data Preprocessing
Data Preprocessing is the essential cleaning and refinement phase of the project. During this stage, I transformed raw, inconsistent data into a machine-learning-ready format by converting `TotalCharges` into a numeric type and handling missing values for new customers. Additionally, I performed feature engineering by creating new metrics like `ServiceCount` and `ContractRisk`, which allowed the model to capture deeper business insights that weren't present in the initial dataset.

### Exploratory Data Analysis - EDA
EDA functioned as the investigative heart of the project, where I used visual storytelling to uncover the "why" behind customer behavior. By plotting churn rates against various categories, I identified that customers on month-to-month contracts and those with high monthly charges are at the highest risk of leaving. This phase was crucial for validating our features and ensuring that the patterns discovered visually aligned with the logical expectations of the telecom industry.



### SMOTE 
SMOTE (Synthetic Minority Over-sampling Technique) was implemented to solve the critical problem of class imbalance, where "staying" customers significantly outnumbered "churning" customers. Without this, the model would likely ignore churners entirely to maintain a high (but misleading) accuracy score. By generating synthetic data points for the minority churn class, I balanced the training set to a 50/50 split, forcing the model to become more sensitive and accurate in predicting when a customer is actually about to leave.

## Feature Engineering & Selection

### Business-Logic Features Engineered

| Feature | Formula / Logic | Purpose |
|---|---|---|
| ChargesPerMonth | `TotalCharges / (tenure + 1)` | Captures average monthly spending rate |
| ServiceCount | Sum of 7 subscribed services | Measures customer embeddedness (0–7) |
| ContractRisk | Month-to-month = 2, One year = 1, Two year = 0 | Encodes churn risk based on contract duration |

### Feature Selection

Feature selection was performed using **SHAP feature importance** from a preliminary **XGBoost** model.

# Stage 6 · Model Building & Training

### Train-Test Split & SMOTE

SMOTE was applied **only on the training set** to prevent data leakage.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

sm = SMOTE(random_state=42)

X_train_res, y_train_res = sm.fit_resample(
    X_train,
    y_train
)
| Model | Purpose |
|---|---|
| Logistic Regression | Interpretable baseline |
| Random Forest | Ensemble model with `class_weight='balanced'` |
| XGBoost | Primary optimized model with tuned `scale_pos_weight`, `max_depth`, and `learning_rate` |

## Model Evaluation & Comparison

| Model | Accuracy | Precision | Recall (Churn) | F1 Score (Churn) | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 80.2% | 65.1% | 55.3% | 59.8% | 0.843 |
| Random Forest | 79.8% | 63.4% | 57.6% | 60.4% | 0.836 |
| XGBoost | **81.5%** | **67.3%** | **62.1%** | **64.6%** | **0.861** |

### Best Model Selection

XGBoost achieved the best overall performance, especially in:

- Recall
- ROC-AUC
- Churn detection capability

### Threshold Optimization

The decision threshold was reduced from:

```python
0.50 → 0.35

# Deployment (Streamlit)

### Deployment Platform

The trained XGBoost model was deployed on:

```python
Streamlit Community Cloud

### App Features

- **Single Customer Prediction** — Interactive form with sliders and dropdowns that outputs:
  - Churn probability
  - Risk label
  - SHAP waterfall explanation
  - Personalized retention recommendation

- **Batch Prediction** — Upload a CSV file of customers to:
  - Generate churn probability scores
  - Download prediction results
  - View a leaderboard of highest-risk customers

- **Model Insights Tab** — Displays:
  - Global SHAP summary plot
  - Feature importance chart
  - EDA visualizations
