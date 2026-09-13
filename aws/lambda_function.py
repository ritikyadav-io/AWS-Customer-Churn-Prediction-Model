"""
lambda_function.py
-------------------
Deploy this in AWS Lambda. It receives requests from API Gateway, invokes
the SageMaker endpoint, and returns the churn prediction.

Setup:
1. Replace ENDPOINT_NAME with the name printed by deploy.py
2. Attach the "AmazonSageMakerFullAccess" policy to this function's
   execution role (IAM console) so it's allowed to call the endpoint
"""

import json

import boto3

ENDPOINT_NAME = "your-sagemaker-endpoint-name"  # <-- change this

runtime = boto3.client("sagemaker-runtime")


def lambda_handler(event, context):
    try:
        body = json.loads(event["body"]) if "body" in event else event
        payload = json.dumps(body)

        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=payload,
        )
        result = json.loads(response["Body"].read().decode())

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
