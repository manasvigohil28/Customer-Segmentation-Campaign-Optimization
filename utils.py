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


# Business names for clusters are assigned after profiling, using the cluster means.
def name_clusters(profile):
    """profile: DataFrame of cluster means indexed by cluster id (k = 4). Returns {id: name}.
    Highest spender -> Premium Spenders, second -> Established Families;
    of the two low-spending clusters, the one with more children -> Budget Families,
    the other (younger, more web visits) -> Young Starters."""
    order = list(profile["Total_Spent"].sort_values(ascending=False).index)
    if len(order) != 4:
        return {int(c): f"Segment {i + 1}" for i, c in enumerate(order)}
    low = profile.loc[order[2:], "Children"].sort_values(ascending=False).index
    return {int(order[0]): "Premium Spenders", int(order[1]): "Established Families",
            int(low[0]): "Budget Families", int(low[1]): "Young Starters"}


STRATEGY = {
    "Premium Spenders": "Luxury wine & meat offers, VIP / loyalty program, catalog campaigns. Top priority for every campaign.",
    "Established Families": "Value bundles and deals (they use discounts often), store and catalog promotions, upgrade offers to move them to premium.",
    "Budget Families": "Discount coupons, family and kids' product bundles, in-store promotions. Keep contact cost low.",
    "Young Starters": "Web and mobile promotions, first-purchase coupons, retargeting ads (they visit the website often but buy little).",
}


def customer_features(inp):
    """Build one engineered feature row from raw inputs entered in the dashboard.
    inp keys: Age, Education, Marital_Status, Income, Kidhome, Teenhome, Recency, Customer_Days,
    the 6 Mnt* columns, the 4 Num*Purchases columns, NumWebVisitsMonth, Total_Accepted_Cmp, Complain."""
    row = dict(inp)
    row["Children"] = inp["Kidhome"] + inp["Teenhome"]
    row["Partner"] = 1 if inp["Marital_Status"] == "Partner" else 0
    row["Edu_Code"] = EDU_CODE[inp["Education"]]
    row["Total_Spent"] = sum(inp[c] for c in MNT_COLS)
    row["Total_Purchases"] = sum(inp[c] for c in PURCHASE_COLS)
    return pd.DataFrame([row])
