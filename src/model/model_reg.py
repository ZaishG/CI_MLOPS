import json
from mlflow.tracking import MlflowClient
import mlflow

import dagshub
dagshub.init(repo_owner='ZaishG', repo_name='CI_MLOPS', mlflow=True)


# Set the experiment name in MLflow
mlflow.set_experiment("CI_MLOPS")

# Set the tracking URI for MLflow to log the experiment in DagsHub
mlflow.set_tracking_uri("https://dagshub.com/ZaishG/CI_MLOPS.mlflow")
# Load the run ID and model name from the saved JSON file
reports_path = "reports/run_info.json"
with open(reports_path, 'r') as file:
    run_info = json.load(file)

run_id = run_info['run_id'] # Fetch run id from the JSON file
model_name = run_info['logged_model_name']  # Fetch model name from the JSON file

# Create an MLflow client
client = MlflowClient()

# Create the model URI
# model_uri = f"runs:/{run_id}/{model_name}"
model_uri = run_info["model_uri"]
print(run_info)
# Register the model
reg = mlflow.register_model(model_uri, model_name)

# Get the model version
model_version = reg.version  # Get the registered model version

# Transition the model version to Staging
stage_alias = "champion"
client.set_registered_model_alias(
    name=model_name,
    alias=stage_alias,
    version=model_version
)
#
# client.transition_model_version_stage(
#     name=model_name,
#     version=model_version,
#     stage=new_stage,
#     archive_existing_versions=True
# )

print(f"Model {model_name} version {model_version} transitioned to {stage_alias} stage.")
