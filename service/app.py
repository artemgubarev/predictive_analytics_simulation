import os
import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Energy Consumption Predictor")

# Глобальные переменные для модели и препроцессора
model = None
preprocessor = None

# Модель для валидации входных данных (Pydantic)
class PredictionRequest(BaseModel):
    data: List[dict] # Ожидаем список словарей (как JSON records)

@app.on_event("startup")
def load_models():
    """Загружаем модель и препроцессор при старте сервиса."""
    global model, preprocessor
    
    # В идеале здесь нужно брать run_id из переменных окружения или MLflow Registry
    # Для примера берем последний успешный запуск
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:../mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    
    # ЗАМЕНИТЕ НА ВАШ RUN_ID из MLflow
    RUN_ID = os.getenv("MLFLOW_RUN_ID", "3b9d9553d7e74003a1fe6421257caed5") 
    
    try:
        # Загружаем модель CatBoost
        model_uri = f"runs:/{RUN_ID}/model"
        model = mlflow.catboost.load_model(model_uri)
        
        # Загружаем препроцессор
        prep_uri = f"runs:/{RUN_ID}/preprocessor/preprocessor.joblib"
        local_prep_path = mlflow.artifacts.download_artifacts(prep_uri)
        preprocessor = joblib.load(local_prep_path)
        
        print("✅ Model and preprocessor loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")

@app.post("/predict")
def predict(request: PredictionRequest):
    """Принимает данные, прогоняет через препроцессор и модель."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Превращаем JSON в DataFrame
        df = pd.DataFrame(request.data)
        
        # Прогоняем через препроцессор (он добавит фичи и заполнит пропуски)
        X, _ = preprocessor.transform(df)
        
        # Делаем предсказание
        predictions = model.predict(X)
        
        # Возвращаем результат
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    """Эндпоинт для проверки работоспособности сервиса (нужен для Docker)."""
    return {"status": "ok", "model_loaded": model is not None}