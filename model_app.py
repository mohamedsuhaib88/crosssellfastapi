#import necessary libraries
# create api endpoints using fastapi

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
# from nbimporter import import_ipynb
# import JanataHack_CrosssellPrediction  # Import your notebook as a module

# from JanataHack_CrosssellPrediction import CategoricalMapper

import warnings
warnings.filterwarnings('ignore')

app = FastAPI()


class Input(BaseModel):
    Gender: object
    Age: int
    Driving_License: int
    Region_Code: float
    Previously_Insured: int
    Vehicle_Age: object
    Vehicle_Damage: object
    Annual_Premium: float
    Policy_Sales_Channel: float
    Vintage: int

class Output(BaseModel):
    Response: int

@app.post("/predict")

def predict(data: Input) -> Output:
    # input
    # dataframe thru list
    X_input = pd.DataFrame([[data.Gender, data.Age, data.Driving_License, data.Region_Code, data.Previously_Insured, data.Vehicle_Age, data.Vehicle_Damage, data.Annual_Premium, data.Policy_Sales_Channel, data.Vintage]])
    X_input.columns = ['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']

    # dataframe thru dictionary (valid)
    #X_input = pd.DataFrame([{'CONSOLE':  data.CONSOLE,'YEAR':  data.YEAR,'CATEGORY':  data.CATEGORY,'PUBLISHER':  data.PUBLISHER,'RATING':  data.RATING,'CRITICS_POINTS':  data.CRITICS_POINTS,'USER_POINTS':  data.USER_POINTS}])
   
    print(X_input)
    # load the model
    model = joblib.load('janatahackcrosssell_pipeline_model.pkl')

    #predict using the model
    prediction = model.predict(X_input)

    # output
    return Output(Response = prediction)
