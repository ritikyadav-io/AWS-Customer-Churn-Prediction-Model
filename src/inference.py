"""
inference.py
-------------
Two jobs:

1. SageMaker entry point functions (model_fn, input_fn, predict_fn, output_fn)
   -> used when this model is deployed as a real SageMaker endpoint.

2. A plain `predict_one(record, model_path)` and `predict_one_detailed(record)` helper
   -> used by the local Flask demo and the test suite, no AWS required.

Author / Maintainer: Ritik Yadav
"""

import json
import os

import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(HERE, "model", "model.joblib")


# ---------------------------------------------------------------------
# SageMaker entry points
# ---------------------------------------------------------------------
def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        if isinstance(data, dict):
            data = [data]
        return pd.DataFrame(data)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    return model.predict(input_data)


def output_fn(prediction, response_content_type):
    results = ["Yes" if p == 1 else "No" for p in prediction]
    result = results[0] if len(results) == 1 else results
    return json.dumps({"Churn Prediction": result})


# ---------------------------------------------------------------------
# Local helper (used by webapp + tests, no AWS needed)
# ---------------------------------------------------------------------
_cached_model = None


def _load_model(model_path=DEFAULT_MODEL_PATH):
    global _cached_model
    if _cached_model is None:
        _cached_model = joblib.load(model_path)
    return _cached_model


def predict_one(record: dict, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """
    record: {"tenure": 72, "MonthlyCharges": 42.1, "Contract": "Two year"}
    returns: {"Churn Prediction": "No"}
    """
    model = _load_model(model_path)
    df = pd.DataFrame([record])
    pred = model.predict(df)[0]
    return {"Churn Prediction": "Yes" if pred == 1 else "No"}


def predict_one_detailed(record: dict, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """
    Detailed prediction returning probability, risk factors, financial LTV risk, and retention recommendations.
    Supports extended feature options (InternetService, TechSupport, OnlineSecurity, PaymentMethod, etc.)
    Author: Ritik Yadav
    """
    model = _load_model(model_path)

    # Extract core features for model pipeline
    core_record = {
        "tenure": float(record.get("tenure", 0)),
        "MonthlyCharges": float(record.get("MonthlyCharges", record.get("monthlycharges", 0))),
        "Contract": str(record.get("Contract", record.get("contract", "Month-to-month"))),
    }
    df = pd.DataFrame([core_record])
    pred = model.predict(df)[0]

    prob_churn = 0.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[0]
        classes = list(model.classes_)
        if 1 in classes:
            idx = classes.index(1)
            prob_churn = float(probs[idx])
        else:
            prob_churn = float(pred)
    else:
        prob_churn = 1.0 if pred == 1 else 0.0

    # Extended Feature Inputs & Risk Impact Analysis
    internet_service = str(record.get("InternetService", "DSL"))
    tech_support = str(record.get("TechSupport", "No"))
    online_security = str(record.get("OnlineSecurity", "No"))
    payment_method = str(record.get("PaymentMethod", "Electronic check"))
    paperless_billing = str(record.get("PaperlessBilling", "Yes"))
    senior_citizen = str(record.get("SeniorCitizen", "No"))
    partner = str(record.get("Partner", "No"))
    dependents = str(record.get("Dependents", "No"))

    extended_risk_delta = 0.0
    risk_factors = []
    recommendations = []

    # Contract Risk
    if core_record["Contract"] == "Month-to-month":
        extended_risk_delta += 0.12
        risk_factors.append("No long-term commitment (Month-to-month contract)")
        recommendations.append("Offer 15% discount for 1-Year or 2-Year contract upgrade")
    elif core_record["Contract"] == "One year":
        recommendations.append("Engage customer 30 days prior to annual renewal with loyalty perk")

    # Tenure Risk
    if core_record["tenure"] < 12:
        extended_risk_delta += 0.10
        risk_factors.append("New account phase (tenure under 12 months)")
        recommendations.append("Assign dedicated account onboarding manager")
    elif core_record["tenure"] >= 48:
        extended_risk_delta -= 0.08
        recommendations.append("VIP status account: eligible for early renewal incentives")

    # Monthly Billing Risk
    if core_record["MonthlyCharges"] > 85:
        extended_risk_delta += 0.08
        risk_factors.append(f"High monthly billing (${core_record['MonthlyCharges']:.2f}/mo)")
        recommendations.append("Conduct product value audit to highlight ROI")

    # Service Tier Risk
    if internet_service == "Fiber optic":
        extended_risk_delta += 0.06
        risk_factors.append("Fiber Optic service tier (statistically higher price sensitivity)")
    if tech_support == "No":
        extended_risk_delta += 0.04
        risk_factors.append("No Tech Support add-on attached")
        recommendations.append("Offer 3 months free Tech Support trial")
    if online_security == "No":
        extended_risk_delta += 0.03
        risk_factors.append("No Online Security add-on attached")
        recommendations.append("Bundle Cybersecurity suite at 50% discount")

    # Payment & Demographics Risk
    if payment_method == "Electronic check":
        extended_risk_delta += 0.04
        risk_factors.append("Manual Electronic Check payment method")
        recommendations.append("Encourage automatic credit card billing with $5 monthly bill credit")
    if senior_citizen == "Yes":
        risk_factors.append("Senior Citizen account tier")
        recommendations.append("Provide simplified billing & priority telephone support")
    if partner == "No" and dependents == "No":
        risk_factors.append("Single user household without family bundle lines")

    final_prob = min(0.98, max(0.02, prob_churn * 0.7 + extended_risk_delta * 0.3))
    prob_percent = round(final_prob * 100, 1)

    if prob_percent >= 75:
        risk_level = "Critical Risk"
    elif prob_percent >= 50:
        risk_level = "High Risk"
    elif prob_percent >= 25:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    if not risk_factors:
        risk_factors.append("Stable usage pattern & low churn risk indicators")
    if not recommendations:
        recommendations.append("Maintain standard account management cadence")

    # Annual Revenue & LTV Risk calculation
    annual_revenue = round(core_record["MonthlyCharges"] * 12, 2)
    ltv_risk_value = round(annual_revenue * final_prob, 2)

    return {
        "Churn Prediction": "Yes" if prob_percent >= 50 else "No",
        "churn_probability": prob_percent,
        "churn_probability_raw": round(final_prob, 4),
        "risk_level": risk_level,
        "financial_ltv": {
            "annual_revenue": annual_revenue,
            "revenue_at_risk": ltv_risk_value,
            "currency": "$"
        },
        "risk_factors": risk_factors,
        "recommendations": recommendations,
        "input_features": {
            "tenure": core_record["tenure"],
            "MonthlyCharges": core_record["MonthlyCharges"],
            "Contract": core_record["Contract"],
            "InternetService": internet_service,
            "TechSupport": tech_support,
            "OnlineSecurity": online_security,
            "PaymentMethod": payment_method,
            "PaperlessBilling": paperless_billing,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents
        },
        "author": "Ritik Yadav"
    }
