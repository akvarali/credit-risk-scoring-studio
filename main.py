import json
from datetime import datetime

import pandas as pd
import streamlit as st

from predict import SAMPLE_APPLICANTS, load_model, score_applicant
from scorecard import CRITERIA_WEIGHTS, MAX_SCORE, MIN_SCORE


st.set_page_config(page_title="Credit Risk Studio", page_icon=":material/account_balance:", layout="wide")

EMPLOYMENT_TYPES = ["SALARIED", "SELF_EMPLOYED", "CONTRACT", "UNEMPLOYED"]
HOME_TYPES = ["OWN", "MORTGAGE", "RENT"]
LOAN_PURPOSES = ["HOME_IMPROVEMENT", "AUTO", "EDUCATION", "PERSONAL", "DEBT_CONSOLIDATION", "BUSINESS"]


@st.cache_resource
def get_model():
    return load_model()


def load_profile():
    profile = st.session_state["profile"]
    applicant = SAMPLE_APPLICANTS.get(profile, SAMPLE_APPLICANTS["Low-risk applicant"])
    for field, value in applicant.items():
        st.session_state[field] = bool(value) if field == "previous_default" else value
    st.session_state["utilization_percent"] = int(round(applicant["credit_utilization"] * 100))
    st.session_state.pop("assessment", None)


def readable(value):
    return value.replace("_", " ").title()


def render_results(assessment):
    result = assessment["result"]
    decision = result["decision"]
    st.subheader("Assessment")
    st.caption(f"Submitted {assessment['submitted_at']} | Results reflect the submitted application.")
    if decision == "APPROVE":
        st.success("APPROVE - Meets the model and policy criteria.", icon=":material/check_circle:")
    elif decision == "REVIEW":
        st.warning("REVIEW - Further assessment is required.", icon=":material/person_search:")
    else:
        st.error("DECLINE - Does not meet the model or policy criteria.", icon=":material/cancel:")

    score_column, grade_column, probability_column = st.columns(3)
    score_column.metric("Credit score", str(result["credit_score"]))
    grade_column.metric("Risk grade", result["risk_grade"])
    probability_column.metric("Default probability", f"{result['probability_of_default']:.2%}")
    st.progress((result["credit_score"] - MIN_SCORE) / (MAX_SCORE - MIN_SCORE))
    st.caption(f"Score range {MIN_SCORE}-{MAX_SCORE} | {result['risk_level']}")

    scorecard_tab, policy_tab, report_tab = st.tabs(["Scorecard", "Policy checks", "Report"])
    with scorecard_tab:
        st.metric("Weighted criteria score", f"{result['criteria_scores']['criteria_score']:.1f} / 100")
        criteria = pd.DataFrame({
            "Criterion": [name.title() for name in CRITERIA_WEIGHTS],
            "Score": [result["criteria_scores"][name] for name in CRITERIA_WEIGHTS],
        })
        st.bar_chart(criteria, x="Criterion", y="Score", horizontal=True, color="#148579")
        st.dataframe(pd.DataFrame({
            "Criterion": criteria["Criterion"],
            "Score / 100": criteria["Score"],
            "Weight": [f"{weight:.0%}" for weight in CRITERIA_WEIGHTS.values()],
        }), hide_index=True, width="stretch")
        ratios = result["key_ratios"]
        st.markdown("#### Financial ratios")
        st.metric("Debt-to-income", f"{ratios['debt_to_income']:.1%}", help="Includes the estimated new loan payment.")
        ratio_left, ratio_right = st.columns(2)
        ratio_left.metric("Loan / annual income", f"{ratios['loan_to_income']:.1%}")
        ratio_right.metric("Savings / loan", f"{ratios['savings_to_loan']:.1%}")
    with policy_tab:
        st.markdown("#### Policy violations")
        if result["policy_violations"]:
            for violation in result["policy_violations"]:
                st.error(violation, icon=":material/block:")
        else:
            st.success("No policy violations.", icon=":material/check_circle:")
        st.markdown("#### Weak areas")
        if result["weak_areas"]:
            for area in result["weak_areas"]:
                st.warning(area)
        else:
            st.write("All criteria scores are at least 40 / 100.")
        if decision == "DECLINE" and not result["policy_violations"]:
            st.error("Risk grade E is below the approval threshold.")
    with report_tab:
        with st.expander("Submitted application"):
            st.json(assessment["applicant"])
        st.json(result, expanded=False)
        st.download_button(
            "Download assessment", data=json.dumps(assessment, indent=2),
            file_name="credit_risk_assessment.json", mime="application/json",
            icon=":material/download:", width="stretch",
        )


def main():
    st.html("""
        <style>
        h1, h2, h3 { font-family: Georgia, serif !important; letter-spacing: 0 !important; }
        .stMainBlockContainer { padding-top: 2rem; }
        </style>
    """)
    if "age" not in st.session_state:
        st.session_state["profile"] = "Custom application"
        load_profile()

    with st.sidebar:
        st.markdown("### Credit Risk Studio")
        st.caption("APPLICANT WORKSPACE")
        st.selectbox("Applicant profile", ["Custom application", *SAMPLE_APPLICANTS], key="profile", on_change=load_profile)
        st.divider()
        st.markdown("#### Risk grades")
        st.table(pd.DataFrame({"Grade": ["A", "B", "C", "D", "E"], "Score": ["750-900", "680-749", "620-679", "560-619", "300-559"]}).set_index("Grade"))
        st.caption("Demo model trained on synthetic applicants. Not a validated lending decision or financial advice.")

    st.title("Credit Risk Studio")
    st.caption("LOAN APPLICATION / RISK ASSESSMENT")
    input_column, output_column = st.columns([1.15, 1], gap="large")
    with input_column:
        st.subheader("Applicant details")
        with st.form("application"):
            personal_tab, credit_tab, loan_tab = st.tabs(["Personal & income", "Credit history", "Loan request"])
            with personal_tab:
                left, right = st.columns(2)
                age = left.number_input("Age", min_value=18, max_value=100, step=1, key="age")
                employment_type = right.selectbox("Employment", EMPLOYMENT_TYPES, format_func=readable, key="employment_type")
                annual_income = left.number_input("Annual income", min_value=1, max_value=10_000_000, step=1000, key="annual_income")
                employment_years = right.number_input("Years employed", min_value=0.0, max_value=60.0, step=0.5, key="employment_years")
                savings_balance = left.number_input("Savings balance", min_value=0, max_value=100_000_000, step=500, key="savings_balance")
                home_ownership = right.selectbox("Home ownership", HOME_TYPES, format_func=readable, key="home_ownership")
            with credit_tab:
                left, right = st.columns(2)
                credit_history_years = left.number_input("Credit history (years)", min_value=0, max_value=80, step=1, key="credit_history_years")
                num_open_accounts = right.number_input("Open credit accounts", min_value=0, max_value=100, step=1, key="num_open_accounts")
                num_late_payments = left.number_input("Late payments (last 2 years)", min_value=0, max_value=100, step=1, key="num_late_payments")
                num_credit_inquiries = right.number_input("Hard inquiries (last 6 months)", min_value=0, max_value=100, step=1, key="num_credit_inquiries")
                utilization_percent = st.slider("Credit utilization (%)", min_value=0, max_value=100, step=1, key="utilization_percent")
                previous_default = st.checkbox("Previous credit default", key="previous_default")
            with loan_tab:
                left, right = st.columns(2)
                loan_amount = left.number_input("Loan amount", min_value=1, max_value=10_000_000, step=1000, key="loan_amount")
                loan_term_months = right.selectbox("Loan term (months)", [12, 24, 36, 48, 60], key="loan_term_months")
                loan_purpose = st.selectbox("Loan purpose", LOAN_PURPOSES, format_func=readable, key="loan_purpose")
                monthly_debt = st.number_input("Existing monthly debt payments", min_value=0, max_value=1_000_000, step=100, key="monthly_debt")
                st.caption("All monetary amounts must use the same currency.")
            submitted = st.form_submit_button("Assess application", type="primary", icon=":material/analytics:", width="stretch")

        if submitted:
            applicant = {
                "age": age, "annual_income": annual_income, "employment_type": employment_type,
                "employment_years": employment_years, "home_ownership": home_ownership,
                "savings_balance": savings_balance, "credit_history_years": credit_history_years,
                "num_open_accounts": num_open_accounts, "credit_utilization": utilization_percent / 100,
                "num_late_payments": num_late_payments, "num_credit_inquiries": num_credit_inquiries,
                "previous_default": int(previous_default), "monthly_debt": monthly_debt,
                "loan_purpose": loan_purpose, "loan_amount": loan_amount, "loan_term_months": loan_term_months,
            }
            st.session_state.pop("assessment", None)
            if employment_years > age - 18:
                st.error("Years employed cannot exceed the number of years since age 18.")
            elif credit_history_years > age - 18:
                st.error("Credit history cannot exceed the number of years since age 18.")
            else:
                try:
                    with st.spinner("Assessing application..."):
                        result = score_applicant(applicant, get_model())
                    st.session_state["assessment"] = {
                        "applicant": applicant, "result": result,
                        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                except FileNotFoundError:
                    st.error("Trained model not found. Run `python train.py` from the project folder, then submit again.")
                except Exception:
                    st.error("The assessment could not be completed. Check the trained model and application logs.")
                    import logging

                    logging.exception("Credit assessment failed")

    with output_column:
        if "assessment" in st.session_state:
            render_results(st.session_state["assessment"])
        else:
            st.subheader("Assessment")
            st.info("No application assessed yet.", icon=":material/assignment:")
            st.markdown("#### Assessment overview")
            st.dataframe(pd.DataFrame({
                "Measure": ["Credit score", "Risk grade", "Criteria score", "Policy status"],
                "Result": ["Not assessed"] * 4,
            }), hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
