"""
Intelligent Customer Segmentation and Marketing Campaign Optimization System
Interactive dashboard – Data Science PBL, L.D. College of Engineering
Author: Manasvi

Run:  streamlit run app.py
"""
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import utils

st.set_page_config(page_title="Customer Intelligence Dashboard", page_icon="📊", layout="wide")

SEG_ORDER = ["Premium Spenders", "Established Families", "Budget Families", "Young Starters"]
SEG_COLORS = dict(zip(SEG_ORDER, ["#1b9e77", "#7570b3", "#d95f02", "#e7298a"]))
MODEL_COLORS = {"Logistic Regression": "#1b9e77", "Decision Tree": "#d95f02", "Random Forest": "#7570b3"}

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.kpi {background: #f3f6f9; border-left: 5px solid #1b9e77; border-radius: 8px; padding: 14px 18px; height: 100%;}
.kpi .label {font-size: 0.85rem; color: #52606d; margin-bottom: 4px;}
.kpi .value {font-size: 1.7rem; font-weight: 700; color: #1f2933;}
.kpi .sub {font-size: 0.8rem; color: #7b8794;}
.seg-card {border-radius: 10px; padding: 14px 16px; color: white; margin-bottom: 8px;}
.seg-card h4 {margin: 0 0 6px 0; color: white;}
.seg-card p {margin: 2px 0; font-size: 0.9rem;}
.result {border-radius: 10px; padding: 18px; background: #f3f6f9; text-align: center;}
.result .big {font-size: 2.2rem; font-weight: 700;}
</style>
""", unsafe_allow_html=True)


def kpi(col, label, value, sub="", color="#1b9e77"):
    col.markdown(f"""<div class="kpi" style="border-left-color:{color}">
        <div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>""",
                 unsafe_allow_html=True)


# ---------------------------------------------------------------- data & models
@st.cache_data
def load_data():
    return pd.read_csv("data/customers_scored.csv", parse_dates=["Dt_Customer"])


@st.cache_resource
def load_models():
    m = {name: joblib.load(f"models/{name}.pkl") for name in
         ["response_model", "logistic_regression", "decision_tree", "random_forest", "linreg",
          "kmeans", "kmeans_scaler", "pca", "cluster_names", "clf_features"]}
    with open("models/metrics.json") as f:
        m["metrics"] = json.load(f)
    return m


df = load_data()
M = load_models()


def clf_matrix(data):
    seg = pd.get_dummies(data["Segment"], prefix="Seg", dtype=int)
    X = pd.concat([data[utils.CLF_FEATURES], seg], axis=1)
    return X.reindex(columns=M["clf_features"], fill_value=0)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("📊 Customer Intelligence")
PAGES = ["🏠 Overview", "👥 Customer Segments", "🤖 Model Performance", "🔮 Predict a Customer", "🎯 Campaign Targeting"]
PAGE_KEYS = ["overview", "segments", "models", "predict", "targeting"]
# A page can also be opened directly with a link, e.g. http://localhost:8501/?page=targeting
start = PAGE_KEYS.index(st.query_params.get("page")) if st.query_params.get("page") in PAGE_KEYS else 0
page = st.sidebar.radio("Navigate", PAGES, index=start)
st.sidebar.markdown("---")
seg_filter = st.sidebar.multiselect("Filter segments", SEG_ORDER, default=SEG_ORDER)
st.sidebar.caption("Dataset: Customer Personality Analysis (Kaggle) · 2,054 customers after cleaning")
st.sidebar.caption("Data Science PBL · L.D. College of Engineering · Manasvi")

view = df[df["Segment"].isin(seg_filter)] if seg_filter else df

# ================================================================= OVERVIEW
if page == "🏠 Overview":
    st.title("Intelligent Customer Segmentation & Campaign Optimization")
    st.caption("Understand customers, find the right segments and target the people most likely to respond.")

    c = st.columns(5)
    kpi(c[0], "Customers", f"{len(view):,}", f"{len(view) / len(df):.0%} of total")
    kpi(c[1], "Average income", f"{view['Income'].mean():,.0f}", "per year", "#7570b3")
    kpi(c[2], "Average spending", f"{view['Total_Spent'].mean():,.0f}", "last 2 years", "#d95f02")
    kpi(c[3], "Total revenue", f"{view['Total_Spent'].sum() / 1e6:,.2f} M", "last 2 years", "#e7298a")
    kpi(c[4], "Response rate", f"{view['Response'].mean():.1%}", "last campaign", "#66a61e")
    st.write("")

    col1, col2 = st.columns(2)
    prod = view[utils.MNT_COLS].sum()
    prod.index = ["Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold"]
    fig = px.pie(values=prod.values, names=prod.index, hole=0.45, title="Revenue by Product Category",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    col1.plotly_chart(fig, width="stretch")

    ch = view[["NumStorePurchases", "NumWebPurchases", "NumCatalogPurchases", "NumDealsPurchases"]].sum()
    fig = px.bar(x=["Store", "Web", "Catalog", "Deals"], y=ch.values, title="Purchases by Channel",
                 labels={"x": "", "y": "Number of purchases"}, color=["Store", "Web", "Catalog", "Deals"],
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(showlegend=False)
    col2.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    fig = px.scatter(view, x="Income", y="Total_Spent", color="Segment", color_discrete_map=SEG_COLORS,
                     category_orders={"Segment": SEG_ORDER}, opacity=0.6, title="Income vs Total Spending",
                     hover_data=["Age", "Children", "Recency"])
    col1.plotly_chart(fig, width="stretch")

    cmp = view[utils.CMP_COLS + ["Response"]].mean() * 100
    fig = px.bar(x=["Cmp 1", "Cmp 2", "Cmp 3", "Cmp 4", "Cmp 5", "Last"], y=cmp.values, text=cmp.round(1).astype(str) + "%",
                 title="Acceptance Rate of Each Campaign", labels={"x": "", "y": "% accepted"})
    fig.update_traces(marker_color=["#8da0cb"] * 5 + ["#e7298a"])
    col2.plotly_chart(fig, width="stretch")

    st.subheader("Key insights")
    st.markdown("""
- **Wine and meat** generate about **78%** of revenue.
- Customers **without children spend ~6× more** than parents, and income is strongly linked to spending (r = 0.78).
- Only **15%** of customers responded to the last campaign – **85% of contacts were wasted**.
- Past responders, recent buyers and long-standing customers are the most likely to respond again.
""")

# ================================================================= SEGMENTS
elif page == "👥 Customer Segments":
    st.title("Customer Segments (K-Means, k = 4)")
    st.caption("Customers grouped by income, spending, age, children, purchases, recency and website visits.")

    cols = st.columns(4)
    for col, seg in zip(cols, SEG_ORDER):
        s = df[df["Segment"] == seg]
        col.markdown(f"""<div class="seg-card" style="background:{SEG_COLORS[seg]}">
            <h4>{seg}</h4>
            <p>👥 {len(s):,} customers ({len(s) / len(df):.0%})</p>
            <p>💰 Avg income: {s['Income'].mean():,.0f}</p>
            <p>🛒 Avg spending: {s['Total_Spent'].mean():,.0f}</p>
            <p>📈 Revenue share: {s['Total_Spent'].sum() / df['Total_Spent'].sum():.0%}</p>
            <p>✅ Response rate: {s['Response'].mean():.1%}</p></div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    fig = px.scatter(view, x="PC1", y="PC2", color="Segment", color_discrete_map=SEG_COLORS,
                     category_orders={"Segment": SEG_ORDER}, opacity=0.65,
                     hover_data=["Income", "Total_Spent", "Age", "Children"],
                     title="Segments in 2D (PCA projection)")
    col1.plotly_chart(fig, width="stretch")

    resp = df.groupby("Segment")["Response"].mean().reindex(SEG_ORDER) * 100
    fig = px.bar(x=resp.index, y=resp.values, color=resp.index, color_discrete_map=SEG_COLORS,
                 text=resp.round(1).astype(str) + "%", title="Response Rate by Segment",
                 labels={"x": "", "y": "Response rate (%)"})
    fig.add_hline(y=df["Response"].mean() * 100, line_dash="dash", annotation_text="Overall")
    fig.update_layout(showlegend=False)
    col2.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    prod = df.groupby("Segment")[utils.MNT_COLS].mean().reindex(SEG_ORDER)
    prod.columns = ["Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold"]
    fig = px.bar(prod, barmode="stack", title="Average Spending by Product",
                 color_discrete_sequence=px.colors.qualitative.Set2, labels={"value": "Amount", "variable": "Product"})
    col1.plotly_chart(fig, width="stretch")
    ch = df.groupby("Segment")[["NumStorePurchases", "NumWebPurchases", "NumCatalogPurchases",
                                "NumDealsPurchases"]].mean().reindex(SEG_ORDER)
    ch.columns = ["Store", "Web", "Catalog", "Deals"]
    fig = px.bar(ch, barmode="group", title="Average Purchases by Channel",
                 color_discrete_sequence=px.colors.qualitative.Set2, labels={"value": "Purchases", "variable": "Channel"})
    col2.plotly_chart(fig, width="stretch")

    st.subheader("Explore a segment")
    seg = st.selectbox("Choose a segment", SEG_ORDER)
    s = df[df["Segment"] == seg]
    st.info(f"**Recommended strategy:** {utils.STRATEGY[seg]}")
    prof_cols = ["Income", "Total_Spent", "Age", "Children", "Total_Purchases", "NumWebVisitsMonth",
                 "Recency", "Customer_Days", "Total_Accepted_Cmp"]
    comp = pd.DataFrame({seg: s[prof_cols].mean(), "All customers": df[prof_cols].mean()})
    comp["Difference %"] = (comp[seg] / comp["All customers"] - 1) * 100
    st.dataframe(comp.round(1), width="stretch")

# ================================================================= MODELS
elif page == "🤖 Model Performance":
    st.title("Model Performance")
    met = M["metrics"]
    tab1, tab2 = st.tabs(["Response classification", "Spending regression"])

    with tab1:
        st.markdown(f"**Best model:** {met['best_model']} (chosen by 5-fold cross-validated ROC-AUC on the training set). "
                    "Evaluated on a stratified 20% test set (411 customers).")
        res = pd.DataFrame(met["classification"]).set_index("Model")
        st.dataframe(res.style.format("{:.3f}").highlight_max(axis=0, color="#c7eadf"), width="stretch")

        # Re-create the same stratified test split to draw ROC curves and confusion matrices
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score
        X_all = clf_matrix(df)
        _, X_te, _, y_te = train_test_split(X_all, df["Response"], test_size=0.2, stratify=df["Response"], random_state=42)

        col1, col2 = st.columns(2)
        fig = go.Figure()
        for name, key in [("Logistic Regression", "logistic_regression"), ("Decision Tree", "decision_tree"),
                          ("Random Forest", "random_forest")]:
            p = M[key].predict_proba(X_te)[:, 1]
            fpr, tpr, _ = roc_curve(y_te, p)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC {roc_auc_score(y_te, p):.3f})",
                                     line=dict(color=MODEL_COLORS[name], width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash", color="grey")))
        fig.update_layout(title="ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                          legend=dict(x=0.35, y=0.08))
        col1.plotly_chart(fig, width="stretch")

        chosen = col2.selectbox("Confusion matrix for", list(MODEL_COLORS))
        key = chosen.lower().replace(" ", "_")
        cm = confusion_matrix(y_te, M[key].predict(X_te))
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Predicted No", "Predicted Yes"], y=["Actual No", "Actual Yes"], title=f"Confusion Matrix – {chosen}")
        fig.update_layout(coloraxis_showscale=False)
        col2.plotly_chart(fig, width="stretch")

        col1, col2 = st.columns(2)
        imp = pd.Series(M["random_forest"].feature_importances_, index=M["clf_features"]).sort_values().tail(12)
        fig = px.bar(x=imp.values, y=imp.index, orientation="h", title="Random Forest – Feature Importance",
                     labels={"x": "Importance", "y": ""}, color_discrete_sequence=["#7570b3"])
        col1.plotly_chart(fig, width="stretch")
        coef = pd.Series(M["logistic_regression"].named_steps["model"].coef_[0], index=M["clf_features"])
        coef = coef.reindex(coef.abs().sort_values().index).tail(12)
        fig = px.bar(x=coef.values, y=coef.index, orientation="h", title="Logistic Regression – Coefficients",
                     labels={"x": "← lowers response | raises response →", "y": ""},
                     color=np.where(coef.values > 0, "Raises", "Lowers"),
                     color_discrete_map={"Raises": "#1b9e77", "Lowers": "#d95f02"})
        fig.update_layout(showlegend=False)
        col2.plotly_chart(fig, width="stretch")

    with tab2:
        lr = met["linear_regression"]
        c = st.columns(3)
        kpi(c[0], "Test R²", f"{lr['test_r2']:.3f}", "share of spending variation explained")
        kpi(c[1], "RMSE", f"{lr['rmse']:,.0f}", "typical prediction error", "#7570b3")
        kpi(c[2], "MAE", f"{lr['mae']:,.0f}", "average absolute error", "#d95f02")
        st.write("")
        coef = pd.Series(M["linreg"].coef_, index=utils.LINREG_FEATURES)
        st.markdown("**Multiple Linear Regression:** Total_Spent = β₀ + Σ βᵢ·xᵢ")
        st.dataframe(pd.DataFrame({"Coefficient": coef.round(3),
                                   "Meaning": ["+1 unit income", "+1 year of age", "+1 child/teen", "lives with partner",
                                               "+1 education level", "+1 day since last purchase",
                                               "+1 day as customer", "+1 website visit/month"]}), width="stretch")
        pred = np.clip(M["linreg"].predict(df[utils.LINREG_FEATURES]), 0, None)
        fig = px.scatter(x=df["Total_Spent"], y=pred, opacity=0.5, labels={"x": "Actual spending", "y": "Predicted spending"},
                         title="Actual vs Predicted Spending (all customers)", color_discrete_sequence=["#8da0cb"])
        fig.add_trace(go.Scatter(x=[0, 2600], y=[0, 2600], mode="lines", name="Perfect", line=dict(dash="dash", color="red")))
        st.plotly_chart(fig, width="stretch")

# ================================================================= PREDICT
elif page == "🔮 Predict a Customer":
    st.title("Predict a Customer")
    st.caption("Enter a customer's details to get their segment, expected spending and probability of responding.")

    preset = st.selectbox("Start from a typical customer of segment", SEG_ORDER)
    base = df[df["Segment"] == preset].median(numeric_only=True)
    base_edu = df.loc[df["Segment"] == preset, "Education"].mode()[0]
    base_mar = df.loc[df["Segment"] == preset, "Marital_Status"].mode()[0]

    with st.form("predict"):
        st.markdown("**Profile**")
        c = st.columns(4)
        age = c[0].number_input("Age", 18, 90, int(base["Age"]), key=f"age_{preset}")
        edu = c[1].selectbox("Education", ["Undergraduate", "Graduate", "Postgraduate"],
                             index=["Undergraduate", "Graduate", "Postgraduate"].index(base_edu), key=f"edu_{preset}")
        mar = c[2].selectbox("Marital status", ["Partner", "Single"], index=["Partner", "Single"].index(base_mar), key=f"mar_{preset}")
        income = c[3].number_input("Yearly income", 0, 200000, int(base["Income"]), step=1000, key=f"inc_{preset}")
        c = st.columns(4)
        kid = c[0].number_input("Kids at home", 0, 3, int(base["Kidhome"]), key=f"kid_{preset}")
        teen = c[1].number_input("Teens at home", 0, 3, int(base["Teenhome"]), key=f"teen_{preset}")
        recency = c[2].number_input("Days since last purchase", 0, 100, int(base["Recency"]), key=f"rec_{preset}")
        days = c[3].number_input("Days as customer", 0, 700, int(base["Customer_Days"]), key=f"days_{preset}")

        st.markdown("**Spending in last 2 years**")
        c = st.columns(6)
        mnt = {}
        for col, name, label in zip(c, utils.MNT_COLS, ["Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold"]):
            mnt[name] = col.number_input(label, 0, 3000, int(base[name]), key=f"{name}_{preset}")

        st.markdown("**Purchases & engagement**")
        c = st.columns(4)
        deals = c[0].number_input("Deal purchases", 0, 20, int(base["NumDealsPurchases"]), key=f"deal_{preset}")
        web = c[1].number_input("Web purchases", 0, 30, int(base["NumWebPurchases"]), key=f"web_{preset}")
        cat = c[2].number_input("Catalog purchases", 0, 30, int(base["NumCatalogPurchases"]), key=f"cat_{preset}")
        store = c[3].number_input("Store purchases", 0, 15, int(base["NumStorePurchases"]), key=f"store_{preset}")
        c = st.columns(4)
        visits = c[0].number_input("Website visits / month", 0, 20, int(base["NumWebVisitsMonth"]), key=f"vis_{preset}")
        acc = c[1].number_input("Previous campaigns accepted (0–5)", 0, 5, int(base["Total_Accepted_Cmp"]), key=f"acc_{preset}")
        complain = c[2].selectbox("Complained in last 2 years?", ["No", "Yes"], key=f"comp_{preset}")
        submitted = st.form_submit_button("🔮 Predict", type="primary")

    if submitted:
        row = utils.customer_features({
            "Age": age, "Education": edu, "Marital_Status": mar, "Income": income, "Kidhome": kid,
            "Teenhome": teen, "Recency": recency, "Customer_Days": days, **mnt,
            "NumDealsPurchases": deals, "NumWebPurchases": web, "NumCatalogPurchases": cat,
            "NumStorePurchases": store, "NumWebVisitsMonth": visits, "Total_Accepted_Cmp": acc,
            "Complain": 1 if complain == "Yes" else 0})

        cluster = int(M["kmeans"].predict(M["kmeans_scaler"].transform(row[utils.CLUSTER_FEATURES]))[0])
        row["Segment"] = M["cluster_names"][cluster]
        prob = float(M["response_model"].predict_proba(clf_matrix(row))[0, 1])
        exp_spend = float(np.clip(M["linreg"].predict(row[utils.LINREG_FEATURES])[0], 0, None))

        if prob >= 0.6:
            level, color, action = "High", "#1b9e77", "Include in the next campaign with a personalised premium offer."
        elif prob >= 0.4:
            level, color, action = "Medium", "#e6ab02", "Include via a low-cost channel (email / web) with a targeted discount."
        else:
            level, color, action = "Low", "#d95f02", "Do not include in paid campaigns; keep in general newsletters only."

        st.markdown("---")
        c = st.columns(3)
        c[0].markdown(f"""<div class="result"><div>Customer segment</div>
            <div class="big" style="color:{SEG_COLORS[row['Segment'][0]]}">{row['Segment'][0]}</div></div>""",
                      unsafe_allow_html=True)
        c[1].markdown(f"""<div class="result"><div>Response probability</div>
            <div class="big" style="color:{color}">{prob:.0%}</div><div>{level} chance</div></div>""",
                      unsafe_allow_html=True)
        c[2].markdown(f"""<div class="result"><div>Expected 2-year spending (profile-based)</div>
            <div class="big">{exp_spend:,.0f}</div><div>Linear Regression estimate</div></div>""",
                      unsafe_allow_html=True)
        st.write("")
        {"High": st.success, "Medium": st.warning, "Low": st.error}[level](f"**Campaign decision:** {action}")
        st.info(f"**Segment strategy:** {utils.STRATEGY[row['Segment'][0]]}")

# ================================================================= TARGETING
elif page == "🎯 Campaign Targeting":
    st.title("Campaign Targeting Simulator")
    st.caption("Rank customers by predicted response probability and decide how many to contact. "
               "Probabilities are out-of-fold predictions (each customer scored by a model that did not train on them).")

    c = st.columns(3)
    pct = c[0].slider("Contact the top % of customers", 5, 100, 30, step=5)
    cost = c[1].number_input("Cost per contact", 0.0, 100.0, 3.0, step=0.5)
    revenue = c[2].number_input("Revenue per response", 0.0, 1000.0, 11.0, step=1.0)

    pool = view.sort_values("Response_Prob", ascending=False).reset_index(drop=True)
    n = int(round(len(pool) * pct / 100))
    target = pool.head(n)
    total_resp = pool["Response"].sum()
    found = target["Response"].sum()
    rate_all = pool["Response"].mean()

    profit_model = found * revenue - n * cost
    profit_all = total_resp * revenue - len(pool) * cost

    c = st.columns(5)
    kpi(c[0], "Customers contacted", f"{n:,}", f"of {len(pool):,}")
    kpi(c[1], "Responders captured", f"{found / total_resp:.0%}" if total_resp else "–", f"{int(found)} of {int(total_resp)}", "#7570b3")
    kpi(c[2], "Response rate", f"{found / n:.1%}" if n else "–", f"vs {rate_all:.1%} contacting all", "#d95f02")
    kpi(c[3], "Contacts saved", f"{1 - n / len(pool):.0%}", "compared with contacting everyone", "#e7298a")
    kpi(c[4], "Campaign profit", f"{profit_model:,.0f}", f"vs {profit_all:,.0f} contacting all", "#66a61e")
    st.write("")

    col1, col2 = st.columns([3, 2])
    cum = pool["Response"].cumsum() / max(total_resp, 1) * 100
    x = (np.arange(1, len(pool) + 1) / len(pool)) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=cum, name="Model targeting", line=dict(color="#1b9e77", width=3)))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], name="Random targeting", line=dict(dash="dash", color="grey")))
    fig.add_vline(x=pct, line_dash="dot", line_color="#d95f02", annotation_text=f"Top {pct}%")
    fig.update_layout(title="Cumulative Gains", xaxis_title="% of customers contacted", yaxis_title="% of responders captured")
    col1.plotly_chart(fig, width="stretch")

    seg_mix = target["Segment"].value_counts().reindex(SEG_ORDER).fillna(0)
    fig = px.pie(values=seg_mix.values, names=seg_mix.index, color=seg_mix.index, color_discrete_map=SEG_COLORS,
                 hole=0.45, title="Segments in the Target List")
    col2.plotly_chart(fig, width="stretch")

    st.subheader(f"Target list – top {pct}% ({n:,} customers)")
    out = target[["Segment", "Response_Prob", "Age", "Education", "Marital_Status", "Income", "Children",
                  "Total_Spent", "Recency", "Total_Accepted_Cmp"]].copy()
    out.insert(0, "Rank", np.arange(1, len(out) + 1))
    out["Recommended action"] = out["Segment"].map(utils.STRATEGY)
    st.dataframe(out.style.format({"Response_Prob": "{:.1%}", "Income": "{:,.0f}", "Total_Spent": "{:,.0f}"}),
                 width="stretch", height=350, hide_index=True)
    st.download_button("⬇️ Download target list (CSV)", out.to_csv(index=False).encode("utf-8"),
                       file_name=f"campaign_target_top{pct}pct.csv", mime="text/csv")
    st.caption("Default cost (3 per contact) and revenue (11 per response) match the Z_CostContact and Z_Revenue "
               "values given in the original dataset.")
