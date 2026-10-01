"""Shared data cleaning + feature engineering used by both the notebook and the Streamlit app."""
import pandas as pd
import numpy as np

DATA_PATH = "data/marketing_campaign.csv"

MNT_COLS = ["MntWines", "MntFruits", "MntMeatProducts",
            "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
PURCHASE_COLS = ["NumDealsPurchases", "NumWebPurchases",
                 "NumCatalogPurchases", "NumStorePurchases"]
CMP_COLS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
            "AcceptedCmp4", "AcceptedCmp5"]

# Features used for K-Means segmentation
CLUSTER_FEATURES = ["Income", "Total_Spent", "Age", "Children",
                    "Total_Purchases", "Recency", "NumWebVisitsMonth"]

# Features used for Linear Regression (target = Total_Spent; no spending columns -> no leakage)
LINREG_FEATURES = ["Income", "Age", "Children", "Partner", "Edu_Code",
                   "Recency", "Customer_Days", "NumWebVisitsMonth"]

# Features used for Response classification
CLF_FEATURES = ["Income", "Age", "Children", "Partner", "Edu_Code", "Recency",
                "Customer_Days", "Total_Spent", "Total_Purchases",
                "NumWebVisitsMonth", "NumDealsPurchases", "NumWebPurchases",
                "NumCatalogPurchases", "NumStorePurchases", "Total_Accepted_Cmp",
                "Complain"] + MNT_COLS

EDU_MAP = {"Basic": "Undergraduate", "2n Cycle": "Undergraduate",
           "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"}
EDU_CODE = {"Undergraduate": 0, "Graduate": 1, "Postgraduate": 2}


def load_raw(path=DATA_PATH):
    # Kaggle file is TAB separated
    return pd.read_csv(path, sep="\t")


def clean(df):
    df = df.copy()

    # 1. Missing income -> median
    df["Income"] = df["Income"].fillna(df["Income"].median())

    # 2. Remove duplicate customers (identical in every column except ID), then drop constant / ID columns
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "ID"])
    df = df.drop(columns=["Z_CostContact", "Z_Revenue", "ID"], errors="ignore")

    # 3. Outliers
    df = df[(df["Year_Birth"] >= 1940) & (df["Income"] < 600000)]

    # 4. Group categories
    df["Education"] = df["Education"].map(EDU_MAP).fillna("Graduate")
    df["Marital_Status"] = np.where(df["Marital_Status"].isin(["Married", "Together"]),
                                    "Partner", "Single")

    # 5. Dates
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)

    # 6. Feature engineering
    df["Age"] = 2014 - df["Year_Birth"]
    df["Total_Spent"] = df[MNT_COLS].sum(axis=1)
    df["Children"] = df["Kidhome"] + df["Teenhome"]
    df["Partner"] = (df["Marital_Status"] == "Partner").astype(int)
    df["Family_Size"] = 1 + df["Partner"] + df["Children"]
    df["Is_Parent"] = (df["Children"] > 0).astype(int)
    df["Total_Purchases"] = df[PURCHASE_COLS].sum(axis=1)
    df["Total_Accepted_Cmp"] = df[CMP_COLS].sum(axis=1)
    df["Customer_Days"] = (df["Dt_Customer"].max() - df["Dt_Customer"]).dt.days
    df["Edu_Code"] = df["Education"].map(EDU_CODE)

    return df.reset_index(drop=True)


def load_clean(path=DATA_PATH):
    return clean(load_raw(path))


# Business names for clusters are assigned in the notebook after profiling,
# based on cluster means of Income / Total_Spent / Children / web visits.
def name_clusters(profile):
    """profile: DataFrame of cluster means indexed by cluster id. Returns {id: name}."""
    # Rank clusters from highest to lowest average spending
    order = profile["Total_Spent"].sort_values(ascending=False).index
    labels = ["Premium Spenders", "Established Families", "Average Shoppers", "Budget Families"]
    if len(order) != 4:
        return {c: f"Segment {i + 1}" for i, c in enumerate(order)}
    return {int(c): labels[i] for i, c in enumerate(order)}


STRATEGY = {
    "Premium Spenders": "Luxury & wine offers, loyalty / VIP program, catalog campaigns.",
    "Established Families": "Family bundles, store promotions, teen-focused offers.",
    "Average Shoppers": "Cross-sell bundles, web retargeting, deal-based reactivation.",
    "Budget Families": "Discount coupons, deals, kids' product bundles, web promotions.",
}
