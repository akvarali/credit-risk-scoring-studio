"""
Step 3: Load the trained model and score applicants (used by api.py).

Quick test:
    python predict.py                  # scores built-in sample applicants
    python predict.py applicant.json   # scores an applicant from a JSON file
"""

import json
import sys

import joblib
import pandas as pd

from features import add_features
from scorecard import CRITERIA_WEIGHTS, criteria_scores, decide, pd_to_score, policy_violations, risk_grade

MODEL_PATH = "ML/credit_risk_model.pkl"

SAMPLE_APPLICANTS = {
    "Low-risk applicant": {
        "age": 42, "annual_income": 95000, "employment_type": "SALARIED", "employment_years": 12,
        "home_ownership": "OWN", "savings_balance": 40000, "credit_history_years": 18,
        "num_open_accounts": 4, "credit_utilization": 0.15, "num_late_payments": 0,
        "num_credit_inquiries": 0, "previous_default": 0, "monthly_debt": 600,
        "loan_purpose": "HOME_IMPROVEMENT", "loan_amount": 20000, "loan_term_months": 36,
    },
    "High-risk applicant": {
        "age": 26, "annual_income": 28000, "employment_type": "CONTRACT", "employment_years": 0.5,
        "home_ownership": "RENT", "savings_balance": 300, "credit_history_years": 2,
        "num_open_accounts": 7, "credit_utilization": 0.88, "num_late_payments": 4,
        "num_credit_inquiries": 5, "previous_default": 1, "monthly_debt": 700,
        "loan_purpose": "BUSINESS", "loan_amount": 18000, "loan_term_months": 60,
    },
}


def load_model():
    return joblib.load(MODEL_PATH)


def score_applicant(applicant: dict, model) -> dict:
    df = add_features(pd.DataFrame([applicant]))
    row = df.iloc[0]

    probability_of_default = float(model.predict_proba(df)[0, 1])
    credit_score = int(pd_to_score(probability_of_default))
    grade, risk_level = risk_grade(credit_score)

    criteria = {k: float(v) for k, v in criteria_scores(df).iloc[0].items()}
    violations = policy_violations(row)
    weak_areas = [f"{name} ({criteria[name]:.0f}/100)" for name in CRITERIA_WEIGHTS if criteria[name] < 40]

    return {
        "decision": decide(grade, criteria["criteria_score"], violations),
        "credit_score": credit_score,
        "risk_grade": grade,
        "risk_level": risk_level,
        "probability_of_default": round(probability_of_default, 4),
        "criteria_scores": criteria,
        "key_ratios": {
            "debt_to_income": float(row["debt_to_income"]),
            "loan_to_income": float(row["loan_to_income"]),
            "savings_to_loan": float(row["savings_to_loan"]),
        },
        "policy_violations": violations,
        "weak_areas": weak_areas,
    }


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as f:
            applicants = {sys.argv[1]: json.load(f)}
    else:
        applicants = SAMPLE_APPLICANTS

    model = load_model()
    for name, applicant in applicants.items():
        print(f"\n=== {name} ===")
        print(json.dumps(score_applicant(applicant, model), indent=2))


if __name__ == "__main__":
    main()
