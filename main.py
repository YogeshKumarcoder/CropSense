from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pipeline import get_ndvi, get_weather, calculate_stress_index, irrigation_recommendation
from datetime import datetime
import numpy as np
from tensorflow.keras.models import load_model
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

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
    bbox: list

@app.post("/predict")
def predict_next_month(request: PredictRequest):
    
    # Current month aur pichle 2 months calculate karo
    today = datetime.now()
    months = []
    
    for i in range(2, -1, -1):  # 2 months ago, 1 month ago, current
        month_date = today - relativedelta(months=i)
        start = month_date.replace(day=1).strftime("%Y-%m-%d")
        # Month ka last day
        if month_date.month == 12:
            end = month_date.replace(day=31).strftime("%Y-%m-%d")
        else:
            end = (month_date.replace(day=1) + relativedelta(months=1) - timedelta(days=1)).strftime("%Y-%m-%d")
        months.append((start, end))
    
    # Har month ka NDVI fetch karo
    ndvi_values = []
    for start, end in months:
        ndvi_result = get_ndvi(request.bbox, start, end)
        if ndvi_result['status'] == 'success':
            ndvi_values.append(ndvi_result['healthy_percentage'])
        else:
            ndvi_values.append(0)
    
    # LSTM predict karo
    ndvi_min_val = 0
    ndvi_max_val = 72.4
    
    normalized = [(x - ndvi_min_val)/(ndvi_max_val - ndvi_min_val) 
                  for x in ndvi_values]
    
    input_array = np.array(normalized).reshape(1, 3, 1)
    pred_norm = float(lstm_model.predict(input_array, verbose=0)[0][0])
    pred_real = pred_norm * (ndvi_max_val - ndvi_min_val) + ndvi_min_val
    
    return {
        "last_3_months_ndvi": ndvi_values,
        "predicted_next_month": round(pred_real, 2),
        "months_used": [m[0] for m in months],
        "unit": "healthy_crop_percentage"
    }

class IrrigationRequest(BaseModel):
    bbox: list
    crop_type: str
    time_start: str
    time_end: str

@app.post("/irrigation")
def get_irrigation_recommendation(request: IrrigationRequest):
    lat = (request.bbox[1] + request.bbox[3]) / 2
    lon = (request.bbox[0] + request.bbox[2]) / 2
    
    ndvi_data = get_ndvi(request.bbox, request.time_start, request.time_end)
    weather_data = get_weather(lat, lon)
    
    recommendation = irrigation_recommendation(
        ndvi_data, weather_data, request.crop_type
    )
    
    return {
        "timestamp": datetime.now().isoformat(),
        "location": {"bbox": request.bbox},
        "satellite_data": ndvi_data,
        "weather_data": weather_data,
        "irrigation": recommendation
    }