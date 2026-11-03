"""
Step 1: Generate a synthetic credit applicants dataset.

Run:   python data_generator.py
Output: Data/credit_applicants.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from features import add_features

N_APPLICANTS = 6000
SEED = 42
OUTPUT_PATH = Path("Data/credit_applicants.csv")

EMPLOYMENT_RISK = {"SALARIED": 0.0, "SELF_EMPLOYED": 0.3, "CONTRACT": 0.4, "UNEMPLOYED": 1.2}
HOME_RISK = {"OWN": -0.3, "MORTGAGE": 0.0, "RENT": 0.3}
PURPOSE_RISK = {
    "HOME_IMPROVEMENT": -0.2,
    "AUTO": -0.1,
    "EDUCATION": 0.1,
    "PERSONAL": 0.2,
    "DEBT_CONSOLIDATION": 0.3,
    "BUSINESS": 0.5,
}


def generate(n=N_APPLICANTS, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 70, n)
    employment_type = rng.choice(list(EMPLOYMENT_RISK), n, p=[0.6, 0.2, 0.15, 0.05])
    employment_years = np.minimum(rng.gamma(2, 3, n), age - 18).round(1)
    employment_years[employment_type == "UNEMPLOYED"] = 0

    annual_income = rng.lognormal(np.log(55_000), 0.5, n).round(-2)
    annual_income[employment_type == "UNEMPLOYED"] *= 0.3

    home_ownership = rng.choice(list(HOME_RISK), n, p=[0.15, 0.40, 0.45])
    savings_balance = (annual_income * rng.beta(1.2, 6, n)).round(-1)

    credit_history_years = np.clip(rng.normal((age - 18) * 0.6, 3), 0, age - 18).round(1)
    num_open_accounts = rng.poisson(4, n)
    credit_utilization = rng.beta(2, 3, n).round(3)
    num_late_payments = rng.poisson(0.2 + 1.5 * credit_utilization**2)
    num_credit_inquiries = rng.poisson(1.2, n)
    previous_default = (rng.random(n) < 0.03 + 0.08 * (num_late_payments >= 2)).astype(int)
    monthly_debt = (annual_income / 12 * rng.beta(2, 7, n)).round(0)

    loan_purpose = rng.choice(list(PURPOSE_RISK), n)
    loan_amount = (annual_income * rng.uniform(0.05, 0.8, n)).round(-2)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], n)

    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income,
        "employment_type": employment_type,
        "employment_years": employment_years,
        "home_ownership": home_ownership,
        "savings_balance": savings_balance,
        "credit_history_years": credit_history_years,
        "num_open_accounts": num_open_accounts,
        "credit_utilization": credit_utilization,
        "num_late_payments": num_late_payments,
        "num_credit_inquiries": num_credit_inquiries,
        "previous_default": previous_default,
        "monthly_debt": monthly_debt,
        "loan_purpose": loan_purpose,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
    })

    # Hidden "true" risk formula - the model must learn this from the data
    f = add_features(df)
    logit = (
        -2.4
        + 3.0 * (f["debt_to_income"] - 0.35)
        + 2.0 * (f["credit_utilization"] - 0.4)
        + 0.45 * f["num_late_payments"]
        + 1.5 * f["previous_default"]
        + 0.15 * f["num_credit_inquiries"]
        - 0.06 * f["credit_history_years"]
        - 0.05 * f["employment_years"]
        - 0.6 * f["savings_to_loan"].clip(upper=2)
        - 0.5 * np.log(f["annual_income"] / 55_000)
        + 0.01 * (f["loan_term_months"] - 36)
        + f["employment_type"].map(EMPLOYMENT_RISK)
        + f["home_ownership"].map(HOME_RISK)
        + f["loan_purpose"].map(PURPOSE_RISK)
        + rng.normal(0, 0.5, n)
    )
    probability_of_default = 1 / (1 + np.exp(-logit))
    df["default"] = (rng.random(n) < probability_of_default).astype(int)
    return df


def main():
    df = generate()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} applicants to {OUTPUT_PATH}")
    print(f"Default rate: {df['default'].mean():.2%}")


if __name__ == "__main__":
    main()
