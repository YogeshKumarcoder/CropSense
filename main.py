from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pipeline import get_ndvi, get_weather, calculate_stress_index
from datetime import datetime

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