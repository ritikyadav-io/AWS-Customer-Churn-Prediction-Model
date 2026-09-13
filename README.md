# Customer Churn Prediction — End-to-End AWS ML Pipeline

Predicts whether a telecom customer is likely to churn (leave), based on
their tenure, monthly billing, and contract type. Built as an end-to-end
ML pipeline: from raw data, through cleaning and model training, to a
deployed, callable prediction API with a web front-end.

**Runs as a full working demo locally — no AWS account required.** The
same code is also ready to deploy on real AWS infrastructure (S3 → Glue →
SageMaker → Lambda → API Gateway) when you want the production version —
see [`docs/STEP_BY_STEP_GUIDE.md`](docs/STEP_BY_STEP_GUIDE.md).

![CI](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Problem

Businesses lose revenue when customers leave without warning. Manually
tracking churn risk doesn't scale, but early, automated prediction lets a
business act before a customer actually leaves.

## What this project does

- Trains a classification model (Random Forest) on customer tenure,
  monthly charges, and contract type to predict churn (Yes/No)
- Serves that model through a simple web form (Flask)
- Includes the exact AWS code needed to deploy the same model as a
  production-style pipeline: S3 for storage, Glue for ETL, SageMaker for
  training/hosting, Lambda + API Gateway for the API layer, and Docker
  for containerized deployment
- Has an automated test suite and a CI pipeline (GitHub Actions) that
  runs it on every push

## Architecture

```
                          LOCAL DEMO (this repo, by default)
   ┌─────────────┐      ┌──────────────┐      ┌───────────────────┐
   │  sample CSV │ ───▶ │  train.py    │ ───▶ │  model.joblib      │
   └─────────────┘      │ (RandomForest│      └─────────┬──────────┘
                         │  pipeline)   │                │
                         └──────────────┘                ▼
                                                ┌───────────────────┐
                                                │  Flask webapp      │
                                                │  (MODE=local)       │
                                                └───────────────────┘


                          FULL AWS PIPELINE (optional, see docs/)
   ┌─────┐   ┌──────┐   ┌───────────┐   ┌─────┐   ┌────────────┐   ┌────────┐   ┌─────────────┐
   │ CSV │──▶│  S3  │──▶│ AWS Glue  │──▶│ S3  │──▶│ SageMaker   │──▶│ Lambda │──▶│ API Gateway │
   └─────┘   └──────┘   │(clean ETL)│   └─────┘   │(train+host) │   └────────┘   └──────┬──────┘
                         └───────────┘             └────────────┘                       │
                                                                                          ▼
                                                                            ┌────────────────────────┐
                                                                            │ Flask webapp (MODE=cloud)│
                                                                            │  Dockerized              │
                                                                            └────────────────────────┘
```

## Tech stack

| Layer | Tools |
|---|---|
| Data processing | Python, Pandas |
| Model | scikit-learn (Random Forest, OneHotEncoder pipeline) |
| Local serving | Flask |
| Cloud training/hosting | Amazon SageMaker |
| Cloud ETL | AWS Glue |
| Cloud API layer | AWS Lambda, Amazon API Gateway |
| Storage | Amazon S3 |
| Containerization | Docker |
| Testing / CI | pytest, GitHub Actions |

## Project structure

```
├── data/
│   └── generate_sample_data.py   # creates a synthetic dataset (no external download needed)
├── src/
│   ├── train.py                  # trains the model (local CSV or S3 input)
│   ├── inference.py              # SageMaker entry point + reusable local predict() helper
│   └── model/                    # trained model artifacts (checked in, ready to use)
├── aws/
│   ├── deploy.py                 # deploys the model as a SageMaker endpoint
│   └── lambda_function.py        # Lambda handler: API Gateway -> SageMaker endpoint
├── webapp/
│   ├── app.py                    # Flask app (MODE=local or MODE=cloud)
│   ├── templates/index.html
│   └── Dockerfile
├── tests/
│   └── test_model.py             # automated tests for the pipeline
├── docs/
│   └── STEP_BY_STEP_GUIDE.md     # full AWS deployment walkthrough
└── .github/workflows/ci.yml      # runs tests automatically on every push
```

## Getting started (local demo, no AWS needed)

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt

# (optional) regenerate the dataset and retrain — a trained model is already included
python data/generate_sample_data.py
python src/train.py

# run the web app
cd webapp
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:5000
```

### Run with Docker instead

```bash
# from the repo root (not the webapp/ folder)
docker build -t churn-app -f webapp/Dockerfile .
docker run -p 5000:5000 churn-app
```

### Run the tests

```bash
pytest tests/ -v
```

## Model performance

Trained on the included synthetic dataset (1,500 rows). Numbers will
differ (and likely improve) on the real Kaggle Telco Customer Churn
dataset — synthetic data is included purely so the repo works instantly
without any external download.

| Metric | Score |
|---|---|
| Accuracy | 0.657 |
| Precision | 0.639 |
| Recall | 0.477 |

## Deploying the real AWS pipeline

The local demo above is fully self-contained. When you want the actual
cloud pipeline (S3, Glue, SageMaker, Lambda, API Gateway, Docker) —
useful for demonstrating real cloud/MLOps skills — follow
[`docs/STEP_BY_STEP_GUIDE.md`](docs/STEP_BY_STEP_GUIDE.md) end to end.

## Future enhancements

- **Automated retraining** — trigger retraining when fresh data lands in S3
- **Monitoring** — CloudWatch dashboards for endpoint health and prediction drift
- **Scalable deployment** — move from a single SageMaker endpoint to ECS/Kubernetes for high-volume traffic
- **Richer features** — extend beyond tenure/charges/contract to the full Telco feature set (internet service, tech support, payment method, etc.)

## Author

**Ritik Yadav**
Machine Learning Engineer & Platform Architect
📧 ritikyadav@example.com


## License

MIT — see [LICENSE](LICENSE).
