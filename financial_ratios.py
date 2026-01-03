# financial_ratios.py
import pandas as pd

# Columns with amounts in rupee-style "3,00,000"
AMOUNT_COLS = [
    "Income",
    "Operating_Cash_Flow",
    "Total_Debt",
    "Current_Assets",
    "Current_Liabilities",
    "EBIT",
    "Interest_Expense",
    "Equity",
]

def clean_amount_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove commas and convert rupee-style strings to float.
    """
    for col in AMOUNT_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def add_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds DSCR, Current_Ratio, Debt_to_Equity, ICR to df.
    Expects amount columns to exist (see AMOUNT_COLS).
    """
    df = clean_amount_columns(df)

    # Avoid division by zero
    for col in ["Total_Debt", "Current_Liabilities", "Equity", "Interest_Expense"]:
        if col in df.columns:
            df[col] = df[col].replace(0, pd.NA)

    df["DSCR"] = df["Operating_Cash_Flow"] / df["Total_Debt"]
    df["Current_Ratio"] = df["Current_Assets"] / df["Current_Liabilities"]
    df["Debt_to_Equity"] = df["Total_Debt"] / df["Equity"]
    df["ICR"] = df["EBIT"] / df["Interest_Expense"]

    return df
