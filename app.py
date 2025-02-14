from fastapi import FastAPI
import joblib
import pandas as pd

# Загрузка обученной модели
model = joblib.load("iris_model.pkl")  # Сохраните модель заранее

app = FastAPI()

@app.post("/predict/")
async def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    # Формируем входные данные для модели
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                              columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    
    # Делаем предсказание
    prediction = model.predict(input_data)[0]
    return {"species": prediction}
