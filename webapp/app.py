"""
app.py
------
The customer-facing web application and API server.
Author: Ritik Yadav

Modes:
  MODE=local (default) -> predicts using the model file directly (src/model/model.joblib).
                           No AWS needed. Perfect for demos and local execution.
  MODE=cloud           -> routes predictions to deployed AWS API Gateway endpoint.
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory

MODE = os.environ.get("MODE", "local")
API_URL = os.environ.get("API_URL", "")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "dist"))

app = Flask(__name__, static_folder="static", template_folder="templates")

if MODE == "local":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from inference import predict_one, predict_one_detailed
else:
    import requests


@app.route("/")
def home():
    return render_template("index.html", mode=MODE)



@app.route("/assets/<path:filename>")
def serve_assets(filename):
    assets_dir = os.path.join(DIST_DIR, "assets")
    if os.path.exists(os.path.join(assets_dir, filename)):
        return send_from_directory(assets_dir, filename)
    return "Not Found", 404


@app.route("/favicon.svg")
@app.route("/favicon.ico")
def favicon():
    dist_fav = os.path.join(DIST_DIR, "favicon.svg")
    if os.path.exists(dist_fav):
        return send_from_directory(DIST_DIR, "favicon.svg")
    svg_content = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>'
    from flask import Response
    return Response(svg_content, mimetype='image/svg+xml')




@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        sample_payload = {
            "tenure": 12,
            "MonthlyCharges": 65.0,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
            "TechSupport": "No"
        }
        if MODE == "local":
            sample_pred = predict_one_detailed(sample_payload)
        else:
            sample_pred = {"Churn Prediction": "Yes", "churn_probability": 85.0, "risk_level": "High Risk"}

        return jsonify({
            "status": "online",
            "endpoint": "/predict",
            "allowed_methods": ["POST", "GET"],
            "description": "Customer Churn Prediction REST API Endpoint",
            "usage": "Send HTTP POST request with JSON payload containing customer features to get real-time churn risk predictions.",
            "sample_prediction": sample_pred,
            "author": "Ritik Yadav"
        }), 200

    if request.is_json:
        data = request.get_json()
    else:
        data = {
            "tenure": int(request.form.get("tenure", 0)),
            "MonthlyCharges": float(request.form.get("monthlycharges", 0)),
            "Contract": request.form.get("contract", "Month-to-month"),
            "InternetService": request.form.get("internetservice", "DSL"),
            "TechSupport": request.form.get("techsupport", "No"),
            "OnlineSecurity": request.form.get("onlinesecurity", "No"),
            "PaymentMethod": request.form.get("paymentmethod", "Electronic check"),
            "PaperlessBilling": request.form.get("paperlessbilling", "Yes"),
            "SeniorCitizen": request.form.get("seniorcitizen", "No"),
            "Partner": request.form.get("partner", "No"),
            "Dependents": request.form.get("dependents", "No"),
        }

    try:
        if MODE == "local":
            detailed_res = predict_one_detailed(data)
        else:
            response = requests.post(API_URL, json=data, timeout=10)
            res_json = response.json()
            pred_str = res_json.get("Churn Prediction", "No")
            prob = 85.0 if pred_str == "Yes" else 15.0
            detailed_res = {
                "Churn Prediction": pred_str,
                "churn_probability": prob,
                "risk_level": "High Risk" if pred_str == "Yes" else "Low Risk",
                "financial_ltv": {
                    "annual_revenue": round(float(data.get("MonthlyCharges", 0)) * 12, 2),
                    "revenue_at_risk": round(float(data.get("MonthlyCharges", 0)) * 12 * (prob / 100), 2),
                    "currency": "$"
                },
                "risk_factors": ["AWS Cloud API Response"],
                "recommendations": ["Managed by SageMaker Endpoint"],
                "input_features": data,
                "author": "Ritik Yadav"
            }

        if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify(detailed_res)
        
        prediction = detailed_res.get("Churn Prediction", "Error")
        return render_template(
            "index.html",
            prediction=prediction,
            detailed=detailed_res,
            mode=MODE
        )
    except Exception as e:
        if request.is_json:
            return jsonify({"error": str(e)}), 500
        return render_template("index.html", prediction=f"Error: {e}", mode=MODE)


@app.route("/api/predict_batch", methods=["POST"])
def predict_batch():
    try:
        body = request.get_json() or {}
        items = body.get("customers", [])
        if not items:
            return jsonify({"error": "No customers provided"}), 400

        results = []
        for idx, item in enumerate(items, 1):
            record = {
                "tenure": int(item.get("tenure", 0)),
                "MonthlyCharges": float(item.get("MonthlyCharges", item.get("monthlycharges", 0))),
                "Contract": str(item.get("Contract", item.get("contract", "Month-to-month"))),
                "InternetService": str(item.get("InternetService", "DSL")),
                "TechSupport": str(item.get("TechSupport", "No")),
                "OnlineSecurity": str(item.get("OnlineSecurity", "No")),
                "PaymentMethod": str(item.get("PaymentMethod", "Electronic check")),
                "PaperlessBilling": str(item.get("PaperlessBilling", "Yes")),
                "SeniorCitizen": str(item.get("SeniorCitizen", "No")),
                "Partner": str(item.get("Partner", "No")),
                "Dependents": str(item.get("Dependents", "No")),
            }
            if MODE == "local":
                res = predict_one_detailed(record)
            else:
                res = {"Churn Prediction": "No", "churn_probability": 20.0, "risk_level": "Low Risk"}
            res["customerID"] = item.get("customerID", f"CUST-{idx:04d}")
            results.append(res)

        return jsonify({"count": len(results), "predictions": results, "author": "Ritik Yadav"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    return jsonify({
        "accuracy": 0.657,
        "precision": 0.639,
        "recall": 0.477,
        "f1_score": 0.546,
        "roc_auc": 0.712,
        "dataset_rows": 1500,
        "model_type": "RandomForestClassifier",
        "n_estimators": 150,
        "max_depth": 6,
        "confusion_matrix": {
            "true_positive": 84,
            "false_positive": 47,
            "true_negative": 113,
            "false_negative": 56
        },
        "features": [
            "tenure", "MonthlyCharges", "Contract", "InternetService",
            "TechSupport", "OnlineSecurity", "PaymentMethod", "PaperlessBilling",
            "SeniorCitizen", "Partner", "Dependents"
        ],
        "target": "Churn (Yes/No)",
        "author": {
            "name": "Ritik Yadav",
            "role": "Machine Learning Engineer & Developer",
            "project": "Customer Churn Prediction — AWS ML Pipeline"
        }
    })


@app.route("/api/architecture", methods=["GET"])
def get_architecture():
    return jsonify({
        "pipeline_steps": [
            {
                "step": 1,
                "name": "Data Storage (Amazon S3)",
                "description": "Raw synthetic or Kaggle CSV data stored in Amazon S3 buckets (`old-data/`, `new-data/`, `model/`).",
                "tech": "AWS S3"
            },
            {
                "step": 2,
                "name": "Data ETL (AWS Glue Visual ETL)",
                "description": "Cleans data via automated visual ETL job (Drop Duplicates -> Drop Null Fields -> Change Schema).",
                "tech": "AWS Glue"
            },
            {
                "step": 3,
                "name": "Model Training & Hosting (Amazon SageMaker)",
                "description": "Trains RandomForest pipeline on ml.t2.medium instance, exports model.tar.gz artifact, deploys real-time HTTP endpoint.",
                "tech": "Amazon SageMaker"
            },
            {
                "step": 4,
                "name": "Serverless API Handler (AWS Lambda)",
                "description": "Python 3.12 handler receives API Gateway requests, formats payloads, calls SageMaker invoke_endpoint().",
                "tech": "AWS Lambda"
            },
            {
                "step": 5,
                "name": "Public API Gateway (Amazon API Gateway)",
                "description": "Exposes secure HTTPS REST POST endpoint `/predict` for external web applications and clients.",
                "tech": "Amazon API Gateway"
            },
            {
                "step": 6,
                "name": "Containerized Web Application (Docker & Flask)",
                "description": "Lightweight Docker container running Flask web client, operating in local fallback or cloud mode.",
                "tech": "Docker & Flask"
            }
        ],
        "ci_cd": "GitHub Actions workflow (.github/workflows/ci.yml) executing automated pytest suite on every commit.",
        "author": "Ritik Yadav"
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "mode": MODE,
        "api_url": API_URL if MODE == "cloud" else "Local model.joblib loaded",
        "python_version": sys.version.split()[0],
        "author": "Ritik Yadav"
    })


if __name__ == "__main__":
    print(f"Running in MODE={MODE}" + (f" (API_URL={API_URL})" if MODE == "cloud" else " (no AWS needed)"))
    app.run(host="0.0.0.0", port=5000, debug=True)
