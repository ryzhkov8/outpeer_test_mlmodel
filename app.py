from fastapi import FastAPI
import joblib
import pandas as pd

model = joblib.load("iris_model.pkl")  

app = FastAPI()

@app.post("/predict/")
async def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                              columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    
    prediction = model.predict(input_data)[0]
    return {"species": prediction}
