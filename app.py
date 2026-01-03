# app.py - Phase 1+2: Portfolio Dashboard + Multi-Scenario + Ensemble ML
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

from financial_ratios import add_financial_ratios

st.set_page_config(
    page_title="CreditLens Pro - Sowjenya J", 
    page_icon="💳",
    layout="wide"
)
st.title("💳 CreditLens Pro")
st.markdown("_Next-gen underwriting: AI PD + DSCR rules + Portfolio analytics_")
st.markdown("_Developed by **Sowjenya J**_")

uploaded_file = st.file_uploader("📁 Upload borrower portfolio CSV", type="csv")

# Multi-scenario selector
st.sidebar.header("🎯 Scenarios")
scenario = st.sidebar.selectbox(
    "Stress Scenario", 
    ["Baseline", "Interest Shock +25%", "Recession -20% Cashflow", "High Inflation"]
)

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = add_financial_ratios(data)

    feature_cols = [
        "Age", "Income", "Debt_to_Income", "Credit_History_Years",
        "DSCR", "Current_Ratio", "Debt_to_Equity", "ICR"
    ]

    if all(col in data.columns for col in feature_cols) and "Default" in data.columns:
        # Ensemble ML Models
        X = data[feature_cols].fillna(data[feature_cols].mean())
        y = data["Default"]

        models = {
            'Logistic': LogisticRegression(max_iter=1000),
            'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42)
        }
        
        pd_predictions = {}
        for name, model in models.items():
            model.fit(X, y)
            pd_predictions[name] = model.predict_proba(X)[:, 1]
        
        # Ensemble PD (average)
        data["PD"] = np.mean(list(pd_predictions.values()), axis=0)

        # Risk grades & decisions (your existing logic)
        data["RiskGrade"] = data["PD"].apply(
            lambda x: "AAA" if x<=0.02 else "AA" if x<=0.05 else "A" if x<=0.10 
            else "BBB" if x<=0.20 else "BB" if x<=0.35 else "B" if x<=0.50 else "C"
        )
        
        data["Decision"] = data.apply(lambda row: 
            "Auto Reject" if row["DSCR"]<1.0 or row["ICR"]<1.5 or row["PD"]>0.5 
            else "Approve" if row["DSCR"]>=1.5 and row["ICR"]>=2.0 and row["PD"]<=0.20 
            else "Review", axis=1
        )

        # Apply scenario
        stressed_data = data.copy()
        if scenario == "Interest Shock +25%":
            stressed_data["Interest_Expense"] *= 1.25
        elif scenario == "Recession -20% Cashflow":
            stressed_data["Operating_Cash_Flow"] *= 0.8
        elif scenario == "High Inflation":
            stressed_data["Operating_Cash_Flow"] *= 0.9
            stressed_data["Interest_Expense"] *= 1.15
        
        stressed_data = add_financial_ratios(stressed_data)
        stressed_data["PD"] = np.mean([m.predict_proba(stressed_data[feature_cols].fillna(X.mean()))[:,1] 
                                       for m in models.values()], axis=0)

        # Portfolio Dashboard (Phase 1)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Portfolio Size", len(data))
        col2.metric("🎯 Avg PD", f"{data['PD'].mean():.1%}")
        col3.metric("✅ Approve Rate", f"{(data['Decision']=='Approve').mean():.1%}")
        col4.metric("🚨 High Risk (C)", f"{(data['PD']>0.5).sum()}")

        # Results Tables
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("📈 Baseline Results")
            baseline_df = data[feature_cols + ["PD", "RiskGrade", "Decision"]]
            st.dataframe(baseline_df, use_container_width=True)
            st.download_button("📥 Download Baseline CSV", 
                             convert_df_to_csv(baseline_df),
                             "baseline_results.csv")
        
        with col_right:
            st.subheader(f"🎯 {scenario} Stress Test")
            stress_df = stressed_data[feature_cols + ["PD", "RiskGrade", "Decision"]]
            st.dataframe(stress_df, use_container_width=True)
            st.download_button("📥 Download Stress CSV", 
                             convert_df_to_csv(stress_df),
                             f"{scenario.lower().replace(' ','_')}_results.csv")

        # Risk Grade Chart (Phase 2 visual)
        fig = px.histogram(data, x="RiskGrade", color="Decision", 
                          title="Portfolio Risk Distribution")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("❌ Missing columns. Need: " + ", ".join(feature_cols) + ", Default")
else:
    st.info("👆 Upload CSV to analyze portfolio")
