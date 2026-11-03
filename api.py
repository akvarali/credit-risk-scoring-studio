"""
Step 4: Serve the credit risk model as a web API using FastAPI.

Run:   uvicorn api:app --reload
Open:  http://127.0.0.1:8000/docs  to try it in the browser
"""

from typing import Literal  # literal used for fixed set of string values in the model

from fastapi import FastAPI
from pydantic import BaseModel, Field   # field used for defining model attributes with validation and metadata

from predict import load_model, score_applicant

app = FastAPI(title="Credit Risk Scoring API", description="API for scoring credit risk based on applicant information.")
model = load_model()


class Applicant(BaseModel):
    age: int = Field(..., ge=18, le=100)
    annual_income: float = Field(..., gt=0, le=10_000_000)
    employment_type: Literal["SALARIED", "SELF_EMPLOYED", "CONTRACT", "UNEMPLOYED"]
    employment_years: float = Field(..., ge=0, le=60)
    home_ownership: Literal["OWN", "MORTGAGE", "RENT"]
    savings_balance: float = Field(..., ge=0, le=100_000_000)
    credit_history_years: float = Field(..., ge=0, le=80)
    num_open_accounts: int = Field(..., ge=0, le=100)
    credit_utilization: float = Field(..., ge=0, le=1, description="Used credit / total credit limit (0-1)")
    num_late_payments: int = Field(..., ge=0, le=100, description="Late payments in the last 2 years")
    num_credit_inquiries: int = Field(..., ge=0, le=100, description="Hard inquiries in the last 6 months")
    previous_default: Literal[0, 1]
    monthly_debt: float = Field(..., ge=0, le=1_000_000, description="Existing monthly debt payments")
    loan_purpose: Literal["HOME_IMPROVEMENT", "AUTO", "EDUCATION", "PERSONAL", "DEBT_CONSOLIDATION", "BUSINESS"]
    loan_amount: float = Field(..., gt=0, le=10_000_000)
    loan_term_months: Literal[12, 24, 36, 48, 60]


@app.get("/")
def home():
    return {"message": "Credit Risk Scoring API is running. Go to /docs to try it."}


@app.post("/score")
def score(applicant: Applicant):
    return score_applicant(applicant.model_dump(), model)   # score the applicant using the loaded model, and return the result
