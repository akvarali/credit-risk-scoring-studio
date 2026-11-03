"""
Shared feature engineering used by data_generator.py, train.py and predict.py.
Keeping it in one place guarantees training and prediction see identical features.
"""

import pandas as pd

ANNUAL_INTEREST_RATE = 0.12  # assumed lending rate used to estimate the new loan's monthly installment

CATEGORICAL_FEATURES = ["employment_type", "home_ownership", "loan_purpose"]
NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "employment_years",
    "savings_balance",
    "credit_history_years",
    "num_open_accounts",
    "credit_utilization",
    "num_late_payments",
    "num_credit_inquiries",
    "previous_default",
    "loan_amount",
    "loan_term_months",
    # engineered ratios
    "debt_to_income",
    "loan_to_income",
    "savings_to_loan",
]
TARGET = "default"


def monthly_installment(principal, term_months, annual_rate=ANNUAL_INTEREST_RATE):
    r = annual_rate / 12
    return principal * r / (1 - (1 + r) ** (-term_months))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    monthly_income = (df["annual_income"] / 12).clip(lower=1)
    emi = monthly_installment(df["loan_amount"], df["loan_term_months"])
    # DTI includes the new loan's installment, i.e. the burden *after* approval
    df["debt_to_income"] = ((df["monthly_debt"] + emi) / monthly_income).round(4)
    df["loan_to_income"] = (df["loan_amount"] / df["annual_income"].clip(lower=1)).round(4)
    df["savings_to_loan"] = (df["savings_balance"] / df["loan_amount"].clip(lower=1)).round(4)
    return df
