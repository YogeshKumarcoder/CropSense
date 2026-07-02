import requests
import numpy as np
from PIL import Image
import io
from dotenv import load_dotenv
import os

load_dotenv()

CLIENT_ID = os.getenv("COPERNICUS_CLIENT_ID")
CLIENT_SECRET = os.getenv("COPERNICUS_CLIENT_SECRET")
OWM_API_KEY = os.getenv("OWM_API_KEY")

SENTINEL_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# ================================
# 1 — Token
# ================================
def get_token():
    response = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        }
    )
    return response.json().get("access_token")

# ================================
# 2 — NDVI
# ================================
def get_ndvi(bbox: list, time_start: str, time_end: str):
    token = get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}
            },
            "data": [{
                "dataFilter": {
                    "timeRange": {
                        "from": f"{time_start}T00:00:00Z",
                        "to": f"{time_end}T23:59:59Z"
                    }
                },
                "type": "sentinel-2-l2a"
            }]
        },
        "output": {
            "width": 256,
            "height": 256,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}]
        },
        "evalscript": """
        //VERSION=3
        function setup() {
            return { input: ["B04","B08"], output: { bands: 3 } };
        }
        function evaluatePixel(sample) {
            let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
            if (ndvi > 0.5) return [0, 0.8, 0];
            if (ndvi > 0.2) return [0.9, 0.9, 0];
            return [0.8, 0, 0];
        }
        """
    }

    response = requests.post(SENTINEL_URL, json=payload, headers=headers)

    if response.status_code == 200:
        img = np.array(Image.open(io.BytesIO(response.content)))
        green = np.sum((img[:,:,1] > 150) & (img[:,:,0] < 100))
        red = np.sum((img[:,:,0] > 150) & (img[:,:,1] < 100))
        total = img.shape[0] * img.shape[1]
        return {
            "healthy_percentage": round((green/total)*100, 2),
            "stressed_percentage": round((red/total)*100, 2),
            "status": "success"
        }
    else:
        return {"status": "error", "message": response.text}

# ================================
# 3 — Weather
# ================================
def get_weather(lat: float, lon: float):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_API_KEY}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return {
            "temperature": data['main']['temp'],
            "humidity": data['main']['humidity'],
            "wind_speed": data['wind']['speed'],
            "condition": data['weather'][0]['description'],
            "status": "success"
        }
    else:
        return {"status": "error", "message": response.text}

# ================================
# 4 — Crop Stress Index
# ================================
def calculate_stress_index(ndvi_data: dict, weather_data: dict):
    stressed_pct = ndvi_data['stressed_percentage']
    temp = weather_data['temperature']
    humidity = weather_data['humidity']

    temp_stress = max(0, (temp - 30) / 15)
    humidity_stress = max(0, (50 - humidity) / 50)
    weather_stress = (temp_stress + humidity_stress) / 2

    crop_stress_index = (stressed_pct/100 * 0.6) + (weather_stress * 0.4)

    if crop_stress_index > 0.5:
        alert = "HIGH"
        recommendation = "Immediate irrigation needed!"
    elif crop_stress_index > 0.2:
        alert = "MODERATE"
        recommendation = "Monitor closely"
    else:
        alert = "HEALTHY"
        recommendation = "Crops in good condition"

    return {
        "crop_stress_index": round(crop_stress_index, 3),
        "alert_level": alert,
        "recommendation": recommendation
    }