import mlflow
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel

app = FastAPI(
    title="Water Potability Test",
    description="An App to tell user if water is fit to use."
)

# dagshub_token = os.getenv("DAGSHUB_TOKEN")
# if not dagshub_token:
#     raise EnvironmentError("DAGSHUB_TOKEN environment is not set")
#
# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "ZaishG"
repo_name= "CI_MLOPS"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")


# Set the tracking URI for MLflow to log the experiment in DagsHub
model_name = "Best Model"

def load_model():
    client = mlflow.tracking.MlflowClient()

    mv = client.get_model_version_by_alias(
        name=model_name,
        alias="champion"
    )
    model_uri = f"models:/{model_name}@champion"
    return mlflow.pyfunc.load_model(model_uri)

model = load_model()

class Water(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float


@app.get("/")
def home():
    return {"message": "Welcome to app"}

@app.post("/predict")
def predict(water:Water):
    sample= pd.DataFrame({
        "ph": [water.ph],
        "Hardness": [water.Hardness],
        "Solids": [water.Solids],
        "Chloramines": [water.Chloramines],
        "Sulfate": [water.Sulfate],
        "Conductivity": [water.Conductivity],
        "Organic_carbon": [water.Organic_carbon],
        "Trihalomethanes": [water.Trihalomethanes],
        "Turbidity": [water.Turbidity]
    })

    predicted_val = model.predict(sample)

    if predicted_val[0] == 1:
        return {"result": "Fit to Drink"}
    else:
        return {"result": "Not fit to Drink"}