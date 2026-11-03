"""
Step 2: Train and evaluate the credit risk model.

Run:   python train.py
Output: ML/credit_risk_model.pkl
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # For preprocessing numeric and categorical features separately, 
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, add_features
from predict import MODEL_PATH
from scorecard import CRITERIA_WEIGHTS, criteria_scores, decide, pd_to_score, policy_violations, risk_grade

DATA_PATH = Path("Data/credit_applicants.csv")
SEED = 42


def build_pipeline(estimator):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("prep", preprocessor), ("model", estimator)])


def ks_statistic(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def main():
    # 1. Load data
    df = add_features(pd.read_csv(DATA_PATH))
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    print(f"Applicants: {len(df)}   Default rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # 2. Compare candidate models with 5-fold cross-validated ROC-AUC
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=SEED),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=SEED),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    print("\nModel comparison (5-fold CV ROC-AUC):")
    cv_auc = {}
    for name, estimator in candidates.items():
        scores = cross_val_score(build_pipeline(estimator), X_train, y_train, cv=cv, scoring="roc_auc")
        cv_auc[name] = scores.mean()
        print(f"  {name:<20} {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(cv_auc, key=cv_auc.get)
    print(f"\nSelected model: {best_name}")
    model = build_pipeline(candidates[best_name]).fit(X_train, y_train)

    # 3. Evaluate on the hold-out test set (standard credit-risk metrics)
    pd_test = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pd_test)
    print("\nTest set performance:")
    print(f"  ROC-AUC     : {auc:.4f}")
    print(f"  Gini        : {2 * auc - 1:.4f}")
    print(f"  KS statistic: {ks_statistic(y_test, pd_test):.4f}")
    print(f"  Brier score : {brier_score_loss(y_test, pd_test):.4f}")

    # Cut-off that best separates defaulters from non-defaulters (max KS / Youden's J)
    fpr, tpr, thresholds = roc_curve(y_test, pd_test)
    cutoff = thresholds[np.argmax(tpr - fpr)]
    y_pred = (pd_test >= cutoff).astype(int)
    print(f"\nClassification report at PD cut-off {cutoff:.3f}:")
    print(classification_report(y_test, y_pred, target_names=["Non-default", "Default"]))
    print("Confusion matrix [rows=actual, cols=predicted]:")
    print(confusion_matrix(y_test, y_pred))

    # 4. How predictive is each rule-based criterion on its own? (higher score = safer, so negate)
    crit = criteria_scores(X_test)
    print("\nCriteria scorecard (5 Cs) - stand-alone ROC-AUC:")
    for name in [*CRITERIA_WEIGHTS, "criteria_score"]:
        print(f"  {name:<15} {roc_auc_score(y_test, -crit[name]):.4f}")
    print(f"  {'ML model':<15} {auc:.4f}")

    # 5. Risk grade table - default rate should rise steadily from A to E
    credit_scores = pd_to_score(pd_test)
    grades = [risk_grade(s)[0] for s in credit_scores]
    grade_table = (
        pd.DataFrame({"grade": grades, "score": credit_scores, "default": y_test.to_numpy()})
        .groupby("grade")
        .agg(applicants=("default", "size"), min_score=("score", "min"),
             max_score=("score", "max"), default_rate=("default", "mean"))
    )
    print("\nRisk grade distribution (test set):")
    print(grade_table.to_string(formatters={"default_rate": "{:.2%}".format}))

    # 6. Final decision engine = ML grade + criteria score + policy rules
    decisions = [
        decide(g, c, policy_violations(row))
        for g, c, (_, row) in zip(grades, crit["criteria_score"], X_test.iterrows())
    ]
    decision_table = (
        pd.DataFrame({"decision": decisions, "default": y_test.to_numpy()})
        .groupby("decision")
        .agg(applicants=("default", "size"), default_rate=("default", "mean"))
    )
    print("\nDecision outcomes (test set):")
    print(decision_table.to_string(formatters={"default_rate": "{:.2%}".format}))

    # 7. Which inputs drive the model the most?
    importance = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=SEED)
    ranked = sorted(zip(X_test.columns, importance.importances_mean), key=lambda x: -x[1])
    print("\nTop 10 features (permutation importance, drop in ROC-AUC):")
    for feature, value in ranked[:10]:
        print(f"  {feature:<22} {value:.4f}")

    # 8. Save the full pipeline (preprocessing + model)
    Path(MODEL_PATH).parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
