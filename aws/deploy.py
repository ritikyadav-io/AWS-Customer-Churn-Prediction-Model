"""
deploy.py
---------
Run inside a SageMaker Studio / Notebook Instance to deploy the trained
model (src/model/model.tar.gz, already uploaded to S3) as a real-time
SageMaker Endpoint.

NOTE: a real-time endpoint bills by the hour while it's InService.
Delete it after your demo (see docs/STEP_BY_STEP_GUIDE.md, Step 10).
"""

import sagemaker
from sagemaker.sklearn.model import SKLearnModel

BUCKET = "your-bucket-name"  # <-- change this

session = sagemaker.Session()
role = sagemaker.get_execution_role()

sk_model = SKLearnModel(
    model_data=f"s3://{BUCKET}/model/model.tar.gz",
    role=role,
    framework_version="1.2-1",
    py_version="py3",
    entry_point="inference.py",  # from src/, upload alongside this script
)

predictor = sk_model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",
)

print("Endpoint Name:", predictor.endpoint_name)
