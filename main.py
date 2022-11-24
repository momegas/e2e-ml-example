from enum import Enum
from fastapi import FastAPI
from xgboost import XGBClassifier
import pandas as pd
from pydantic import BaseModel
from preprocessing import process_data
import numpy as np


class ApplicantInfo(BaseModel):
    Gender: float
    Married: float
    Education: float
    Self_Employed: float
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area_semiurban: float
    Property_Area_urban: float
    Dependents_1: float
    Dependents_2: float
    Dependents_3: float

    class Config:
        use_enum_values = True


app = FastAPI()

# Load XGBoost model
model = XGBClassifier()
model.load_model("./model.json")


@app.post("/predict")
def get_model_prediction(application_info: ApplicantInfo):
    # Convert input data to pandas dataframe

    input = np.asarray([list(k[1] for k in application_info)]).tolist()
    print(input)
    return model.predict(input).tolist()
