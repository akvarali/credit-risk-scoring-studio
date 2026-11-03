# Credit Risk Scoring

Scores loan applicants with a machine learning model plus a multi-criteria scorecard, then makes a decision: **APPROVE**, **REVIEW** or **DECLINE**.

## Project structure

```
Data/credit_applicants.csv   synthetic applicants (created by data_generator.py)
ML/credit_risk_model.pkl     trained pipeline (created by train.py)
features.py                  shared feature engineering (DTI, loan-to-income, savings-to-loan)
scorecard.py                 credit score scaling, risk grades, 5 Cs criteria, policy rules, decision
data_generator.py            Step 1 - generate dataset
train.py                     Step 2 - compare models, train, evaluate
predict.py                   Step 3 - score applicants from the CLI
api.py                       Step 4 - FastAPI service
main.py                      Streamlit applicant form and assessment dashboard
```

## How scoring works

Every applicant is scored three ways:

### 1. ML model: probability of default (PD), credit score and grade
`train.py` compares Logistic Regression, Random Forest and Gradient Boosting using 5-fold cross-validated ROC-AUC, then keeps the best one.
The model's PD is mapped to a **300-900 credit score**: 650 points at 20:1 good:bad odds, and every 50 points (PDO) doubles the odds.

| Grade | Score   | Risk level     |
|-------|---------|----------------|
| A     | 750+    | Very Low       |
| B     | 680-749 | Low            |
| C     | 620-679 | Moderate       |
| D     | 560-619 | High           |
| E     | < 560   | Very High      |

### 2. Criteria scorecard: the 5 Cs of Credit
Each criterion is scored from 0 to 100 and then weighted:

| Criterion  | Weight | Based on |
|------------|--------|----------|
| Character  | 30% | late payments, previous default, credit history length, credit inquiries |
| Capacity   | 30% | debt-to-income (incl. new loan EMI), income, employment years and type |
| Capital    | 15% | savings vs loan amount, credit utilisation |
| Collateral | 10% | home ownership, secured loan purpose |
| Conditions | 15% | loan-to-income, loan term |

Any criterion below 40 is reported as a **weak area**.

### 3. Policy knock-out rules
These rules decline an applicant no matter what the score is:
- Debt-to-income above 60%
- A previous default plus 3 or more recent late payments
- Unemployed, with savings covering less than 50% of the loan
- Credit utilisation above 95%

### Decision
| Condition | Decision |
|-----------|----------|
| Any policy violation, or grade E | DECLINE |
| Grade A or B | APPROVE |
| Grade C and criteria score of 60 or more | APPROVE |
| Everything else | REVIEW |

## Evaluation (printed by train.py)
- ROC-AUC, Gini, KS statistic, Brier score
- Classification report and confusion matrix at the cut-off with the highest KS
- Stand-alone ROC-AUC of each criterion compared with the ML model
- Default rate per risk grade. It should rise from A to E.
- Default rate per decision
- Permutation feature importance

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

python data_generator.py
python train.py
python predict.py                  # sample applicants
uvicorn api:app --reload           # http://127.0.0.1:8000/docs -> POST /score
```

## Streamlit Frontend

With the environment activated and the trained model available, run:

```powershell
python -m streamlit run main.py
```

Open http://localhost:8501. The frontend calls the scoring pipeline directly;
the FastAPI server does not need to be running.

Enter applicant details across the personal, credit history, and loan tabs,
then select **Assess application**. Results include the decision, credit score,
risk grade, default probability, weighted criteria, financial ratios, and policy
warnings. Sample profiles are available in the sidebar. Download an assessment
as JSON from the Report tab; results retain the submitted input snapshot.

Use the same currency for all monetary inputs. This model uses synthetic data
and is a demonstration, not a validated lending decision system.

## Streamlit Community Cloud

Deploy branch `main` with `main.py` as the entry point and Python 3.13.
Include `ML/credit_risk_model.pkl` in the repository.

Cloud deployment installs dependencies from `requirements.txt` using public
PyPI. Do not commit a locally generated `uv.lock` containing private package
mirror URLs: those hosts are not accessible from Streamlit Cloud. The local
lockfile is ignored for this reason. After pushing dependency changes, reboot
the app in Streamlit Cloud if it does not rebuild automatically.
