import mlflow
from mlflow.tracking import MlflowClient
import os

# Token based Authentication
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "ZaishG"
repo_name= "CI_MLOPS"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

mlflow.set_experiment("CI_MLOPS")

# Set the tracking URI for MLflow to log the experiment in DagsHub

model_name = "Best Model"
champion = "champion"
previous_champion = "previous_champion"
current_candidate = "candidate"
def promote_model_to_prod():
    """Promote model to prod as Champion"""
    client = MlflowClient()

    try:
        # Get Champion and set them previous champion
        champion_mv = client.get_model_version_by_alias(
            name=model_name,
            alias=champion
        )

        client.set_registered_model_alias(
            name=model_name,
            version=champion_mv.version,
            alias=previous_champion
        )
    except Exception:
            print("No champions currently...")

    try:
        candidate_mv = client.get_model_version_by_alias(
            name=model_name,
            alias=current_candidate
        )
    except Exception:
        print("No condidates to promote...")
        return

    client.set_registered_model_alias(
        name=model_name,
        version=candidate_mv.version,
        alias=champion
    )
    # Remove candidate alias so same version does not keep both aliases
    client.delete_registered_model_alias(
        name=model_name,
        alias=current_candidate,
    )

    client.set_model_version_tag(
        name=model_name,
        version=candidate_mv.version,
        key="validation_status",
        value="Success"
    )

    client.set_model_version_tag(
        name=model_name,
        version=candidate_mv.version,
        key="qa_status",
        value="success"
    )

    print("Model Promoted to prod as champion")


if __name__ == "__main__":
    promote_model_to_prod()