# 🌾 CropSense — Real-time Agricultural Intelligence Platform

> Production-grade multimodal ML system for real-time crop health 
> monitoring, yield prediction, and irrigation recommendations

![Python](https://img.shields.io/badge/Python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-1.0-green)
![React](https://img.shields.io/badge/React-18-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Sentinel-2](https://img.shields.io/badge/Sentinel--2-ESA-orange)

## 🎯 Problem Statement
Traditional crop monitoring requires physical field visits — 
expensive, time-consuming, and not scalable. CropSense provides 
real-time agricultural intelligence using satellite imagery 
and live weather data fusion.

## 🏗️ System Architecture

![CropSense Architecture](architecture.png)

## ✨ Features

### 1. 🛰️ Real-time Crop Health Monitoring
- Sentinel-2 satellite imagery (ESA Copernicus)
- NDVI-based pixel-wise crop health classification
- Green/Yellow/Red health zones

### 2. 🧠 Multimodal ML Fusion
- Satellite NDVI + Live weather data fusion
- Weighted Crop Stress Index
- HEALTHY / MODERATE / HIGH alert levels

### 3. 🔮 LSTM Yield Prediction
- Trained on 4 years real satellite data (2022-2024)
- Early stopping to prevent overfitting
- Validated against real crop calendar — Mehrawal, UP

### 4. 💧 Irrigation Recommendation Engine
- Crop-specific water requirements (14 crops)
- Domain knowledge from real farming experience
- Recommendations in mm with reasoning

### 5. 📊 Multi-field Comparison
- Compare multiple regions simultaneously
- Sorted by stress index — most critical first
- Side-by-side health metrics table

### 6. 🗺️ Interactive Dashboard
- World map with search by name
- Draw custom region on map
- Real-time results

## 🔬 ML Pipeline

| Component | Technology |
|---|---|
| Satellite Data | Sentinel-2 L2A (ESA Copernicus) |
| Spectral Analysis | NDVI (NIR-Red)/(NIR+Red) |
| Weather Data | OpenWeatherMap API |
| Fusion Model | Weighted multimodal combination |
| Yield Prediction | LSTM (50 units) + Early Stopping |
| Backend | FastAPI + Python |
| Frontend | React + Leaflet.js |

## 🚀 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/analyze` | POST | Real-time crop health analysis |
| `/predict` | POST | LSTM next month yield prediction |
| `/irrigation` | POST | Crop-specific irrigation recommendation |
| `/compare-fields` | POST | Multi-field comparison |

## 📊 Sample Output

```json
{
  "satellite_data": {
    "healthy_percentage": 34.72,
    "stressed_percentage": 14.28
  },
  "weather_data": {
    "temperature": 38.22,
    "humidity": 23
  },
  "ml_output": {
    "crop_stress_index": 0.235,
    "alert_level": "HEALTHY"
  }
}
```

## 🌍 Real-World Validation
- Validated on real agricultural land — Mehrawal, Aligarh, UP
- Crop calendar validated by domain expert (farming background)
- July 2026 prediction: 7.3% — confirmed by ground truth

## ⚠️ Known Limitations

### Disease Detection
Satellite-based disease detection is currently limited by:
- Sentinel-2 resolution (10m/pixel) — insufficient for 
  field-level disease identification
- Detection lag — by the time NDVI drops, crop damage 
  is already severe

**Proposed solution:** Integration with ground-level IoT 
sensors or drone imagery for early disease detection

## 🚀 Quick Start

### Backend
```bash
cd cropsense
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
```bash
cd cropsense-frontend
npm install
npm start
```

## 🎥 Demo
[![CropSense Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

## 👨‍💻 Author
**Yogesh Kumar** — ML Engineer  
`Python • FastAPI • React • TensorFlow • Geospatial ML • LSTM`