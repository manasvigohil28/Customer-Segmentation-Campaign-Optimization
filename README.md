# Intelligent Customer Segmentation and Marketing Campaign Optimization System

**Data Science (BE05016021) – PBL Activity · L.D. College of Engineering, Ahmedabad**  
**Author:** Manasvi

Companies often run marketing campaigns without knowing which customers are likely to respond, which wastes marketing budget. This project analyses customer demographic, behavioural and purchase data to:

1. **Segment customers** into meaningful groups (K-Means clustering)
2. **Predict which customers will respond** to a campaign (Logistic Regression, Decision Tree, Random Forest)
3. **Estimate customer spending** (Linear Regression)
4. Present everything in an **interactive Streamlit dashboard** for business stakeholders

---

## Key Results

| Area | Result |
|---|---|
| Data cleaning | 2,240 → **2,054** customers (duplicates, outliers and invalid values removed) |
| Estimation | Mean 2-year spending **606 (95% CI 580–633)**; response rate **15.2% (95% CI 13.7%–16.8%)** |
| Linear Regression | Predicts spending with **R² = 0.71** (income ↑, children ↓ are the main drivers) |
| Segmentation | 4 segments – **Premium Spenders** and **Established Families** are 50% of customers but generate **~90% of revenue** |
| Classification | **Logistic Regression** best: **ROC-AUC 0.885**, recall **81%** |
| Business impact | Contacting only the **top 30%** of customers captures **~80% of responders** with **70% fewer contacts**, turning a campaign loss (−2,719) into a profit (+979) |

---

## Project Structure

```
LDCE_DataScience_Project/
├── app.py                     # Streamlit dashboard (5 pages)
├── utils.py                   # Shared cleaning & feature engineering
├── requirements.txt
├── data/
│   ├── marketing_campaign.csv     # Original Kaggle dataset (tab-separated)
│   ├── customers_clean.csv        # After cleaning + feature engineering
│   ├── customers_segmented.csv    # + K-Means segment
│   └── customers_scored.csv       # + response probability
├── notebooks/
│   └── analysis.ipynb         # Complete analysis (Steps 1–6)
├── models/                    # Saved models (.pkl) and metrics.json
├── figures/                   # All charts used in the report
└── report/screenshots/        # Dashboard screenshots
```

## Methodology

| Step | Course topic | What was done |
|---|---|---|
| 1. Data cleaning | Data preprocessing | Missing values, duplicates, outliers, category grouping, 10 new features |
| 2. EDA | Descriptive analytics | 10 analyses of demographics, spending, channels and campaign response |
| 3. Sampling & Estimation | Sampling and estimation | Confidence intervals, random vs stratified sampling, bootstrap, sample-size planning, t-tests and chi-square tests |
| 4. Linear Regression | Linear regression | Simple and multiple regression to predict Total_Spent, OLS significance tests, diagnostics |
| 5. Segmentation | Clustering | K-Means (k = 4 by elbow + silhouette), PCA visualisation, segment profiling and strategy |
| 6. Classification | Logistic regression, classification | Logistic Regression, Decision Tree, Random Forest with GridSearchCV, ROC-AUC, confusion matrix, feature importance, gains/lift analysis |
| 7. Dashboard | Application development | Streamlit app: overview, segments, model performance, customer prediction, campaign targeting simulator |

## How to Run

```bash
pip install -r requirements.txt
```

Run the analysis notebook (creates the cleaned data, figures and models):

```bash
jupyter notebook notebooks/analysis.ipynb
```

Launch the dashboard:

```bash
streamlit run app.py
```

The dashboard opens at http://localhost:8501. Pages can also be opened directly, e.g. `http://localhost:8501/?page=targeting`.

## Dashboard Pages

- **Overview** – KPIs, revenue by product, purchase channels, campaign acceptance
- **Customer Segments** – segment cards, PCA plot, response by segment, segment explorer with marketing strategy
- **Model Performance** – model comparison, ROC curves, confusion matrices, feature importance, regression results
- **Predict a Customer** – enter a customer's details to get segment, response probability, expected spending and a campaign decision
- **Campaign Targeting** – choose the % of customers to contact, see responders captured and campaign profit, download the target list (CSV)

## Dataset

**Customer Personality Analysis** – Kaggle  
https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis  
2,240 customers × 29 attributes (demographics, product spending, purchase channels, campaign responses).

## Tech Stack

Python · pandas · NumPy · scikit-learn · SciPy · statsmodels · Matplotlib · Seaborn · Plotly · Streamlit
