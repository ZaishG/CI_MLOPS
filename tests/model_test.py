import unittest
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import os
import pandas as pd

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

class TestModelLoading(unittest.TestCase):
    "Load model"

    def test_model_exists(self):
        '''Check if model exits'''
        client = MlflowClient()
        mv = client.get_model_version_by_alias(
            name=model_name,
            alias="candidate"
        )
        self.assertGreater(len(mv), 0, "No models exists there")

    def test_model_loading(self):
        """Test if model can be loaded"""
        client = MlflowClient()

        mv = client.get_model_version_by_alias(
            name=model_name,
            alias="candidate"
        )
        if not mv:
            self.fail("No model with alias candidate")

        model_uri = f"models:/{model_name}@candidate"
        try:
            loaded_model = mlflow.pyfunc.load_model(model_uri)

        except Exception as e:
            self.fail(f"Failed to load the model: {e}")

        self.assertIsNotNone(loaded_model, "The loaded model is None")
        print(f"Model loaded successfully from {loaded_model}")

if __name__ == "__main__":
    unittest.main()