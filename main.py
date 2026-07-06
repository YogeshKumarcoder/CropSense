from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pipeline import get_ndvi, get_weather, calculate_stress_index
from datetime import datetime
import numpy as np
from tensorflow.keras.models import load_model
import json

# Model aur scaler load karo
lstm_model = load_model('cropsense_lstm.keras')
with open('scaler_params.json', 'r') as f:
    scaler = json.load(f)

ndvi_min = scaler['ndvi_min']
ndvi_max = scaler['ndvi_max']   

app = FastAPI(
    title="CropSense API",
    description="Real-time Agricultural Intelligence Platform",
    version="1.0.0"
)

# CORS — React frontend se connect hoga
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# Request Model
# ================================
class AnalyzeRequest(BaseModel):
    bbox: list           # [min_lon, min_lat, max_lon, max_lat]
    time_start: str      # "2024-11-01"
    time_end: str        # "2024-11-30"

# ================================
# Routes
# ================================
@app.get("/")
def root():
    return {
        "name": "CropSense API",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/analyze")
def analyze_region(request: AnalyzeRequest):
    # Center point nikalo bbox se
    lat = (request.bbox[1] + request.bbox[3]) / 2
    lon = (request.bbox[0] + request.bbox[2]) / 2

    # NDVI data
    ndvi_data = get_ndvi(request.bbox, request.time_start, request.time_end)

    # Weather data
    weather_data = get_weather(lat, lon)

    # Stress index
    stress_data = calculate_stress_index(ndvi_data, weather_data)

    return {
        "timestamp": datetime.now().isoformat(),
        "location": {
            "bbox": request.bbox,
            "center": {"lat": lat, "lon": lon}
        },
        "satellite_data": ndvi_data,
        "weather_data": weather_data,
        "ml_output": stress_data
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

class PredictRequest(BaseModel):
    last_3_months: list

@app.post("/predict")
def predict_next_month(request: PredictRequest):
    if len(request.last_3_months) != 3:
        return {"error": "Exactly 3 months ka data chahiye"}
    
    # Normalize
    normalized = [(x - ndvi_min)/(ndvi_max - ndvi_min) 
                  for x in request.last_3_months]
    
    # Predict
    input_array = np.array(normalized).reshape(1, 3, 1)
    pred_norm = lstm_model.predict(input_array)[0][0]
    pred_real = float(pred_norm * (ndvi_max - ndvi_min) + ndvi_min)
    
    return {
        "input": request.last_3_months,
        "predicted_next_month_ndvi": round(pred_real, 2),
        "unit": "healthy_crop_percentage"
    }