# Deploying This Project to AWS (Full Production Pipeline)

This repo runs as a complete demo locally (see main README) with **zero AWS
required**. This guide is for when you're ready to deploy the *real* AWS
pipeline: S3 → Glue → SageMaker → Lambda → API Gateway → Docker, matching
a full industrial-training style architecture.

> 💡 SageMaker endpoints bill by the hour once deployed. Only deploy a day
> or two before you need to demo it, and delete it right after (Step 8).

---

## Step 1 — Create your S3 bucket and upload the data

1. AWS Console → **S3** → **Create bucket** → name it e.g. `churn-prediction-<yourname>`
2. Inside it, create folders: `old-data/`, `new-data/`, `model/`
3. Upload `data/sample_customer_churn.csv` (or your real Kaggle dataset) into `old-data/`

## Step 2 — Clean the data with AWS Glue (Visual ETL)

1. AWS Console → **Glue** → **ETL jobs** → **Visual ETL** → **Create job**, name it `Customer-Churn-ETL`
2. Source: S3 → point at `old-data/`
3. Add transforms: **Drop Duplicates** → **Drop Null Fields** → **Change Schema**
4. Target: S3 → point at `new-data/`
5. Save, then Run. Wait for status **Succeeded**.

## Step 3 — Train the model in SageMaker

1. AWS Console → **SageMaker** → **Studio** (or **Notebook instances**) → create one (`ml.t2.medium` is enough)
2. Upload `src/train.py` and `src/inference.py` into the notebook
3. Run:
   ```bash
   python train.py --s3-bucket churn-prediction-<yourname> --s3-key new-data/<your-glue-output-filename> --upload-to-s3
   ```
   This trains the model, prints accuracy/precision/recall, and uploads `model.tar.gz` straight to `s3://<bucket>/model/`.

## Step 4 — Deploy a real-time endpoint

1. Upload `aws/deploy.py` into the same notebook folder as `inference.py`
2. Edit the `BUCKET` variable in `deploy.py`
3. Run it — after a few minutes it prints an **Endpoint Name**. Copy it.

## Step 5 — Create the Lambda function

1. AWS Console → **Lambda** → **Create function** → name it `customer-churn-lambda`, Python 3.12
2. Paste in the contents of `aws/lambda_function.py`
3. Replace `ENDPOINT_NAME` with the name from Step 4
4. **Deploy**
5. Configuration → Permissions → attach `AmazonSageMakerFullAccess` to its execution role
6. Test it with: `{"tenure": 72, "MonthlyCharges": 42.1, "Contract": "Two year"}`

## Step 6 — Expose it with API Gateway

1. AWS Console → **API Gateway** → **Create API** → REST or HTTP API
2. Create resource `/predict`, method **POST**, integrate with `customer-churn-lambda`
3. **Deploy API**, copy the **Invoke URL**

## Step 7 — Test with Postman

POST to `<Invoke URL>/predict` with JSON body:
```json
{ "tenure": 72, "MonthlyCharges": 42.1, "Contract": "Two year" }
```
Expect: `{"Churn Prediction": "No"}`

## Step 8 — Point the webapp at your real API (cloud mode)

```bash
cd webapp
MODE=cloud API_URL="https://<your-invoke-url>/predict" python app.py
```
Same app, same UI — now calling your live AWS pipeline instead of the local model file.

## Step 9 — Clean up (avoid ongoing charges)

1. SageMaker → **Endpoints** → delete yours
2. SageMaker → **Notebook instances** → stop (or delete)
3. Optionally delete the Lambda function, API Gateway API, and S3 bucket

---

## Monitoring & security notes (for your presentation)

- **CloudWatch** automatically logs every Lambda invocation and SageMaker endpoint metric — screenshot the logs console as evidence
- **IAM** roles/policies from Steps 4–5 are your access-control layer
- **GitHub Actions CI** (`.github/workflows/ci.yml`) automatically runs the test suite on every push — this covers the "CI/CD Integration" enhancement, already implemented rather than just proposed
