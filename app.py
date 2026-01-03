# app.py
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

from financial_ratios import add_financial_ratios

st.set_page_config(page_title="Credit Underwriting Simulator", layout="wide")
st.title("Credit Underwriting Simulator")
st.markdown("_Developed by **Sowjenya J**_")

st.write("Upload borrower financial data to compute ratios, PD, risk grades and approval decisions.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

# --- Risk engine helpers -----------------------------------------------------
def map_pd_to_grade(pd_value: float) -> str:
    if pd_value <= 0.02:
        return "AAA"
    elif pd_value <= 0.05:
        return "AA"
    elif pd_value <= 0.10:
        return "A"
    elif pd_value <= 0.20:
        return "BBB"
    elif pd_value <= 0.35:
        return "BB"
    elif pd_value <= 0.50:
        return "B"
    else:
        return "C"

def approval_decision(row) -> str:
    dscr = row.get("DSCR", None)
    icr = row.get("ICR", None)
    cr = row.get("Current_Ratio", None)
    pd_val = row.get("PD", None)

    if any(x is None for x in [dscr, icr, cr, pd_val]):
        return "Review"

    if (dscr is not None and dscr < 1.0) or \
       (icr is not None and icr < 1.5) or \
       (cr is not None and cr < 1.0) or \
       (pd_val is not None and pd_val > 0.5):
        return "Auto Reject"

    if dscr >= 1.5 and icr >= 2.0 and cr >= 1.2 and pd_val <= 0.20:
        return "Approve"

    return "Review"

def explain_grade_and_decision(row) -> str:
    grade = row.get("RiskGrade", "")
    decision = row.get("Decision", "")
    dscr = row.get("DSCR", float("nan"))
    icr = row.get("ICR", float("nan"))
    cr = row.get("Current_Ratio", float("nan"))
    pd_val = row.get("PD", float("nan"))

    reasons = []
    if pd_val <= 0.05:
        reasons.append("very low default probability")
    elif pd_val <= 0.20:
        reasons.append("moderate default probability")
    else:
        reasons.append("high default probability")

    if dscr >= 1.5:
        reasons.append("strong DSCR")
    elif dscr < 1.0:
        reasons.append("weak DSCR")

    if icr >= 2.0:
        reasons.append("strong ICR")
    elif icr < 1.5:
        reasons.append("weak ICR")

    if cr >= 1.2:
        reasons.append("good liquidity")
    elif cr < 1.0:
        reasons.append("weak liquidity")

    text = f"Grade {grade}, decision: {decision} – " + ", ".join(reasons)
    return text

# --- Sidebar: Stress test ----------------------------------------------------
st.sidebar.header("Stress Testing – Interest Rate Shock")
rate_shock_pct = st.sidebar.slider(
    "Increase interest expense by (%)", min_value=0, max_value=200, value=0, step=10
)

# --- CSV to download helper --------------------------------------------------
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# --- Main flow ---------------------------------------------------------------
if uploaded_file is not None:
    # 1. Load CSV
    data = pd.read_csv(uploaded_file)

    # 2. Compute ratios (also cleans numeric amount columns)
    data = add_financial_ratios(data)

    # 3. Feature columns for PD model (match CSV and ratios)
    feature_cols = [
        "Age",
        "Income",
        "Debt_to_Income",
        "Credit_History_Years",
        "DSCR",
        "Current_Ratio",
        "Debt_to_Equity",
        "ICR",
    ]

    missing_feats = [c for c in feature_cols if c not in data.columns]
    if missing_feats:
        st.error(f"Missing required columns for PD model: {missing_feats}")
    else:
        if "Default" not in data.columns:
            st.error("No 'Default' column in data – cannot train PD model.")
        else:
            base_data = data.copy()

            # 4. Train Logistic Regression
            X = data[feature_cols]
            y = data["Default"]

            model_df = pd.concat([X, y], axis=1).dropna()
            X_model = model_df[feature_cols]
            y_model = model_df["Default"]

            if y_model.nunique() < 2:
                st.error("Default column has only one class – cannot train logistic regression.")
            else:
                model = LogisticRegression(max_iter=1000)
                model.fit(X_model, y_model)

                # 5. Baseline PD
                X_full = data[feature_cols].copy()
                X_full = X_full.fillna(X_model.mean())
                pd_probs = model.predict_proba(X_full)[:, 1]
                data["PD"] = pd_probs

                data["RiskGrade"] = data["PD"].apply(map_pd_to_grade)
                data["Decision"] = data.apply(approval_decision, axis=1)
                data["Explanation"] = data.apply(explain_grade_and_decision, axis=1)

                st.subheader("Baseline – Ratios, PD, Risk Grade & Decision")
                st.dataframe(
                    data[
                        feature_cols
                        + ["PD", "RiskGrade", "Decision", "Explanation"]
                    ]
                )

                # ---- Download baseline results as CSV ----
                baseline_df = data[feature_cols + ["PD", "RiskGrade", "Decision", "Explanation"]]
                csv_baseline = convert_df_to_csv(baseline_df)

                st.download_button(
                    label="📥 Download baseline results as CSV",
                    data=csv_baseline,
                    file_name="baseline_credit_underwriting_results.csv",
                    mime="text/csv",
                )

                # 6. Stress test – increase interest expense
                st.subheader("Stress Testing – Interest Rate Shock")

                stressed = base_data.copy()
                if "Interest_Expense" in stressed.columns:
                    stressed["Interest_Expense"] = (
                        stressed["Interest_Expense"] * (1 + rate_shock_pct / 100.0)
                    )
                    stressed = add_financial_ratios(stressed)

                    X_stress = stressed[feature_cols].copy()
                    X_stress = X_stress.fillna(X_model.mean())
                    stressed_pd = model.predict_proba(X_stress)[:, 1]
                    stressed["PD"] = stressed_pd
                    stressed["RiskGrade"] = stressed["PD"].apply(map_pd_to_grade)
                    stressed["Decision"] = stressed.apply(approval_decision, axis=1)
                    stressed["Explanation"] = stressed.apply(explain_grade_and_decision, axis=1)

                    st.write(f"Interest expense increased by **{rate_shock_pct}%**.")
                    st.dataframe(
                        stressed[
                            feature_cols
                            + ["PD", "RiskGrade", "Decision", "Explanation"]
                        ]
                    )

                    # ---- Download stressed results as CSV ----
                    stressed_df = stressed[feature_cols + ["PD", "RiskGrade", "Decision", "Explanation"]]
                    csv_stressed = convert_df_to_csv(stressed_df)

                    st.download_button(
                        label="📥 Download stressed results as CSV",
                        data=csv_stressed,
                        file_name=f"stressed_results_{rate_shock_pct}pct.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("Interest_Expense column not found – cannot run interest rate shock.")
else:
    st.info("Please upload a CSV file with borrower and financial data.")
