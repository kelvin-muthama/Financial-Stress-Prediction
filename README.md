# Financial Stress Prediction

**Predicting Customer Liquidity Stress Using Mobile Money Transaction Data**

## Overview

Financial stability is an important aspect of individual well-being. However, identifying customers who may experience financial difficulties before those challenges become severe remains a significant challenge for financial institutions.

The **Financial Stress Prediction** project applies machine learning techniques to mobile money transaction data to identify early warning signs of financial hardship.

Using six months of customer transaction history, demographic information, platform activity, and mobile money balance records, the project aims to predict the likelihood of a customer experiencing **liquidity stress within the next 30 days**.

By identifying potentially vulnerable customers early, financial institutions can develop proactive interventions, improve customer support, and promote better financial inclusion.

## Problem Statement

The project is formulated as a supervised binary classification problem.

The objective is to predict the target variable:

`liquidity_stress_next_30d`

Where:

- `1` – The customer is expected to experience liquidity stress within the next 30 days.
- `0` – The customer is not expected to experience liquidity stress within the next 30 days.

The model generates a probability score representing the likelihood of each customer experiencing financial stress.

### Evaluation Metrics

Model performance is evaluated using a weighted combination of Log Loss and ROC-AUC.

| Evaluation Metric | Weight | Description |
|---|---:|---|
| Log Loss | 60% | Measures the quality and accuracy of predicted probabilities. |
| ROC-AUC | 40% | Measures the model's ability to distinguish between stressed and non-stressed customers. |

The evaluation approach emphasizes both probability calibration and classification discrimination.

**Model optimization objective:** Minimize Log Loss while maximizing ROC-AUC.

## Dataset Description

The dataset contains customer-level information and six months of historical mobile money transaction activity.

The data is organized into the following major categories:

### 1. Customer Profiles

Static customer characteristics that provide demographic and behavioural context.

Examples include:

- Customer age
- Geographical region
- Customer segment
- Earning pattern
- Other available demographic attributes

### 2. Platform Activity

Measures of customer engagement and interaction with the mobile money platform.

Activity rate provides additional context for understanding transaction frequency and customer usage patterns.

### 3. Transactional Behaviour

Six months of transaction history across seven major transaction categories:

| Transaction Category | Description |
|---|---|
| Paybill | Payments made through bill payment services. |
| MerchantPay | Payments made to merchants and businesses. |
| Transfer_from_bank | Funds transferred from bank accounts. |
| MM_Send | Money sent to other mobile money users. |
| Received | Money received from other mobile money users. |
| Deposit | Cash deposited into mobile money accounts. |
| Withdraw | Cash withdrawn from mobile money accounts. |

### 4. Balance History

Historical average daily mobile money balances for each of the six months.

Balance information provides insights into customers' liquidity positions, savings behaviour, and potential financial vulnerability.

## Methodology

The project follows an end-to-end machine learning workflow consisting of data preprocessing, exploratory data analysis, feature engineering, model development, evaluation, and prediction.

### 1. Data Preprocessing and Exploration

The initial stage focuses on understanding the structure, quality, and distribution of the available data.

Key activities include:

- Loading and inspecting training and testing datasets.
- Examining data types and column structures.
- Identifying missing values and duplicate records.
- Investigating outliers and unusual transaction patterns.
- Analyzing the distribution of the target variable.
- Comparing customer characteristics across the target classes.
- Identifying potential data leakage and inconsistencies.

### 2. Feature Engineering

Feature engineering transforms raw transaction records into meaningful financial and behavioural indicators.

The dataset follows a structured column naming convention:

`{month}_{group}_{metric}`

This structure allows transaction-related columns to be programmatically identified and grouped.

The following feature engineering strategies are explored:

**A. Aggregate Features**

Statistical summaries across the six-month observation period:

- Total transaction amounts.
- Average monthly transaction amounts.
- Transaction standard deviations.
- Maximum and minimum transaction values.
- Total transaction counts.
- Average daily balances.

**B. Temporal Features**

Features that capture changes in customer behaviour over time:

- Month-over-month changes.
- Percentage changes in transaction activity.
- Six-month transaction trends.
- Balance deterioration patterns.
- Increasing or decreasing withdrawal activity.
- Changes in customer transaction frequency.

**C. Financial Behavioural Ratios**

Financial indicators derived from relationships between transaction categories:

- Total inflow-to-outflow ratio.
- Send-to-receive ratio.
- Deposit-to-withdrawal ratio.
- Balance-to-withdrawal ratio.
- Transaction activity ratios.
- Net cash flow.

**D. Counterparty Diversity**

Measures of customer transaction diversity, including:

- Number of unique companies.
- Number of unique merchants.
- Number of unique recipients.
- Number of unique transaction counterparties.

These features help capture changes in customer financial behaviour that may not be visible through transaction volumes alone.

### 3. Exploratory Data Analysis

Exploratory analysis is used to identify patterns and relationships that may contribute to liquidity stress.

Key areas of investigation include:

- Distribution of liquidity stress across customers.
- Transaction activity among stressed and non-stressed customers.
- Monthly balance movements.
- Differences in customer inflows and outflows.
- Withdrawal and deposit behaviour.
- Regional and demographic patterns.
- Correlations between engineered features and the target variable.

Visualization tools such as Matplotlib, Seaborn, and Plotly support the analytical process.

### 4. Machine Learning Model Development

The problem is approached using supervised binary classification algorithms.

Several models are considered to establish performance benchmarks and explore different learning capabilities.

**Baseline Model**

Logistic Regression provides an interpretable baseline against which more complex models can be evaluated.

**Gradient Boosting Models**

The primary modelling candidates include:

- LightGBM
- XGBoost

These algorithms are particularly useful for structured tabular datasets and can capture nonlinear relationships and interactions between financial behavioural features.

Additional models, such as CatBoost, may also be explored where appropriate.

### 5. Model Validation

Stratified K-Fold Cross-Validation is used to evaluate model performance while maintaining approximately consistent target-class proportions across validation folds.

The validation process includes:

- Training models across multiple folds.
- Generating out-of-fold predictions.
- Calculating Log Loss and ROC-AUC.
- Comparing model performance.
- Monitoring overfitting.
- Evaluating probability calibration.

Out-of-fold predictions are retained for model comparison and ensemble development.

### 6. Model Ensembling and Optimization

To improve predictive performance, the project explores combining predictions from multiple classification models.

The ensemble approach involves:

1. Generating out-of-fold predictions from individual models.
2. Comparing their predictive performance.
3. Experimenting with weighted probability blending.
4. Optimizing ensemble weights against the competition evaluation metric.
5. Generating final predictions using the selected models.

Probability calibration may also be investigated to improve Log Loss.

## Technology Stack

The project uses Python and its machine learning ecosystem.

| Technology | Application |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Plotly | Interactive visualizations |
| Scikit-learn | Preprocessing, validation, and machine learning |
| LightGBM | Gradient boosting classification |
| XGBoost | Gradient boosting classification |
| CatBoost | Additional gradient boosting experiments |
| Optuna | Hyperparameter optimization |
| SHAP | Model interpretability |

## Repository Structure

```text
Financial-Stress-Prediction/
│
├── DATA/
│   ├── Train.csv
│   └── Test.csv
│
├── StarterNotebook.ipynb
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
└── models/
    └── (Saved model artifacts)
```

*The repository structure may evolve as the project develops.*

## Getting Started

### Prerequisites

Ensure that the following are installed:

- Python 3.8 or later.
- pip package manager.
- Git.
- Jupyter Notebook or JupyterLab.

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/kelvin-muthama/Financial-Stress-Prediction.git
```

**2. Navigate to the project directory**

```bash
cd Financial-Stress-Prediction
```

**3. Create a virtual environment (recommended)**

```bash
python -m venv venv
```

Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

### Running the Project

1. Place the `Train.csv` and `Test.csv` datasets inside the `DATA/` directory.
2. Launch Jupyter Notebook:

```bash
jupyter notebook
```

3. Open `StarterNotebook.ipynb`.
4. Execute the notebook cells sequentially to explore the data, engineer features, train models, and generate predictions.

### Model Deployment

An optional Flask application (`app.py`) is included in the project structure for exposing the trained model through a prediction API.

The deployment workflow involves:

- Loading the trained model.
- Accepting customer transaction features.
- Applying the same preprocessing and feature engineering pipeline.
- Generating liquidity stress probabilities.
- Returning prediction results through an API endpoint.

The deployment component is optional and depends on the availability of a trained model artifact and the completed API implementation.

## Expected Analytical Insights

The project investigates several financial behavioural indicators that may be associated with liquidity stress.

These include:

**1. Cash Flow Imbalance**

Customers whose outgoing transactions consistently exceed incoming funds may have a higher likelihood of experiencing liquidity difficulties.

**2. Declining Mobile Money Balances**

A sustained reduction in average daily balances may indicate deteriorating liquidity.

**3. Reduced Incoming Transactions**

Declining peer-to-peer receipts, bank transfers, or deposits may indicate reduced access to incoming funds.

**4. Withdrawal Dependency**

High withdrawal activity relative to deposits may indicate increased demand for cash liquidity.

**5. Changes in Transaction Behaviour**

Sudden reductions in activity, unusual spending patterns, or changes in transaction frequency may provide additional early warning signals.

These are hypotheses to investigate through the data, not established findings. Their predictive importance will be assessed using validation results and model interpretation techniques.

## Future Improvements

Potential extensions to the project include:

- Advanced temporal feature engineering.
- Rolling-window statistical features.
- Automated feature selection.
- Hyperparameter optimization using Optuna.
- Additional experiments with CatBoost and neural networks.
- Advanced probability calibration.
- Stacking and other ensemble learning techniques.
- SHAP-based model interpretation.
- Interactive financial analytics dashboards.
- Flask API deployment.
- Monitoring model performance and prediction drift.
- Fairness evaluation across customer demographic and regional groups.

## Ethical Considerations

Financial stress prediction involves sensitive customer financial information. Responsible development and deployment are therefore essential.

The project recognizes the importance of:

- Protecting customer privacy and confidentiality.
- Applying appropriate data security measures.
- Evaluating potential demographic and regional biases.
- Ensuring predictions are not treated as definitive financial diagnoses.
- Using predictions to support responsible customer assistance rather than unfair exclusion.
- Maintaining transparency and accountability in automated decision-making.

Any real-world deployment should undergo appropriate privacy, security, fairness, and regulatory assessments.

## License

This project is intended for educational, research, and demonstration purposes.

The dataset remains subject to its original provider's terms and conditions. Users should ensure they have appropriate permission before redistributing or using any underlying data.

## Acknowledgments

- Data provider for supplying the mobile money transaction dataset.
- The open-source Python and machine learning communities.
- Contributors and researchers working to advance data-driven financial inclusion.

---

**Developed by:** Kelvin Mutuku Muthama

**GitHub:** [kelvin-muthama](https://github.com/kelvin-muthama)

**Project Repository:** [Financial-Stress-Prediction](https://github.com/kelvin-muthama/Financial-Stress-Prediction)

*Turning financial transaction data into actionable insights through machine learning.*
