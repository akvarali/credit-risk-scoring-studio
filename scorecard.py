"""
Scoring logic: turns model output + applicant data into a credit decision.

  1. ML score       - probability of default (PD) -> credit score 300-900 -> risk grade A-E
  2. Criteria score - rule-based "5 Cs of Credit" sub-scores (0-100), weighted
  3. Policy rules   - hard knock-out rules that decline regardless of score
"""

import numpy as np
import pandas as pd

# Score scaling: 650 points at 20:1 good:bad odds, +50 points every time the odds double (PDO)
BASE_SCORE, BASE_ODDS, PDO = 650, 20, 50
MIN_SCORE, MAX_SCORE = 300, 900
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)

GRADES = [
    (750, "A", "Very Low Risk"),
    (680, "B", "Low Risk"),
    (620, "C", "Moderate Risk"),
    (560, "D", "High Risk"),
    (MIN_SCORE, "E", "Very High Risk"),
]

CRITERIA_WEIGHTS = {
    "character": 0.30,   # repayment behaviour: late payments, past defaults, history length, inquiries
    "capacity": 0.30,    # ability to repay: DTI, income, employment stability
    "capital": 0.15,     # financial buffer: savings vs loan, unused credit
    "collateral": 0.10,  # security: home ownership, secured loan purpose
    "conditions": 0.15,  # loan terms: size vs income, tenure
}

EMPLOYMENT_SCORE = {"SALARIED": 100, "SELF_EMPLOYED": 75, "CONTRACT": 65, "UNEMPLOYED": 0}
HOME_SCORE = {"OWN": 100, "MORTGAGE": 70, "RENT": 40}
PURPOSE_SCORE = {
    "HOME_IMPROVEMENT": 80,
    "AUTO": 80,
    "EDUCATION": 60,
    "DEBT_CONSOLIDATION": 40,
    "PERSONAL": 35,
    "BUSINESS": 30,
}


def pd_to_score(probability_of_default):
    p = np.clip(probability_of_default, 1e-6, 1 - 1e-6)
    score = OFFSET + FACTOR * np.log((1 - p) / p)
    return np.clip(np.round(score), MIN_SCORE, MAX_SCORE).astype(int)


def risk_grade(score):
    for cutoff, grade, label in GRADES:
        if score >= cutoff:
            return grade, label


def _scale(value, best, worst):
    """Map a value linearly to 0-100 where `best` -> 100 and `worst` -> 0."""
    return np.clip((np.asarray(value, dtype=float) - worst) / (best - worst), 0, 1) * 100


def criteria_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised 5 Cs scorecard; expects the output of features.add_features()."""
    character = (
        _scale(df["num_late_payments"], 0, 6)
        + _scale(df["credit_history_years"], 10, 0)
        + _scale(df["num_credit_inquiries"], 0, 6)
    ) / 3 - 40 * df["previous_default"].to_numpy()

    capacity = (
        _scale(df["debt_to_income"], 0.2, 0.6)
        + _scale(df["employment_years"], 5, 0)
        + _scale(df["annual_income"], 100_000, 15_000)
        + df["employment_type"].map(EMPLOYMENT_SCORE).to_numpy()
    ) / 4

    capital = (
        _scale(df["savings_to_loan"], 1.0, 0)
        + _scale(df["credit_utilization"], 0.1, 0.9)
    ) / 2

    collateral = (
        df["home_ownership"].map(HOME_SCORE).to_numpy()
        + df["loan_purpose"].map(PURPOSE_SCORE).to_numpy()
    ) / 2

    conditions = (
        _scale(df["loan_to_income"], 0.1, 0.8)
        + _scale(df["loan_term_months"], 12, 60)
    ) / 2

    scores = pd.DataFrame(
        {"character": character, "capacity": capacity, "capital": capital,
         "collateral": collateral, "conditions": conditions},
        index=df.index,
    ).clip(0, 100)
    scores["criteria_score"] = sum(scores[name] * w for name, w in CRITERIA_WEIGHTS.items())
    return scores.round(1)


def policy_violations(row) -> list[str]:
    reasons = []
    if row["debt_to_income"] > 0.60:
        reasons.append("Debt-to-income ratio above 60%")
    if row["previous_default"] and row["num_late_payments"] >= 3:
        reasons.append("Previous default with 3+ recent late payments")
    if row["employment_type"] == "UNEMPLOYED" and row["savings_to_loan"] < 0.5:
        reasons.append("Unemployed without savings covering 50% of the loan")
    if row["credit_utilization"] > 0.95:
        reasons.append("Existing credit lines maxed out (>95% utilisation)")
    return reasons


def decide(grade: str, criteria_score: float, violations: list[str]) -> str:
    if violations or grade == "E":
        return "DECLINE"
    if grade in ("A", "B"):
        return "APPROVE"
    if grade == "C" and criteria_score >= 60:
        return "APPROVE"
    return "REVIEW"
