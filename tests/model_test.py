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
    def test_model_performance(self):
        """ Test model performance"""
        client = MlflowClient()
        mv = client.get_model_version_by_alias(
            name=model_name,
            alias="candidate"
        )
        if not mv:
            self.fail("No model as candidate, skip model performance")

        model_uri = f"models:/{model_name}@candidate"
        try:
            loaded_model = mlflow.pyfunc.load_model(model_uri)

        except Exception as e:
            self.fail(f"Failed to load the model: {e}")

        self.assertIsNotNone(loaded_model, "The loaded model is None")
        print(f"Model loaded successfully from {loaded_model}")

        test_data_path = "./data/processed/test_processed.csv"
        if not os.path.exists(test_data_path):
            self.fail("Test data not found")

        test_data = pd.read_csv(test_data_path)
        X_test = test_data.drop(columns=["Potability"])
        y_test = test_data["Potability"]

        pred = loaded_model.predict(X_test)

        acc = accuracy_score(y_test, pred)
        pre = precision_score(y_test, pred, average="binary")
        recall = recall_score(y_test, pred, average="binary")
        f1score = f1_score(y_test, pred, average="binary")

        print(f"Accuracy : {acc}")
        print(f"Precision : {pre}")
        print(f"Recall : {recall}")
        print(f"F1_score : {f1score}")

        self.assertGreaterEqual(acc, 0.3, "Accuracy is below threshold")
        self.assertGreaterEqual(pre, 0.3, "Precision is below threshold")
        self.assertGreaterEqual(recall, 0.3, "Recall is below threshold")
        self.assertGreaterEqual(f1score, 0.3, "F1_score is below threshold")


if __name__ == "__main__":
    unittest.main()