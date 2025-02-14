import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib

data = pd.read_csv("iris.csv")
X = data.drop(columns=["species"])
y = data["species"]

# 2. Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Настройка и обучение модели
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Оценка модели
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 5. Логирование метрик и модели с помощью MLflow
mlflow.set_experiment("Iris Classification")

with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "iris_model")
    print("Модель и метрики сохранены в MLflow")
    


# Сохранение обученной модели
joblib.dump(model, "iris_model.pkl")
print("Модель сохранена как iris_model.pkl")
