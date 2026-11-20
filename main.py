
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

# Load our previously trained XGBoost model.
MODEL_FILE = Path(__file__).parent / "xgboost_advanced.joblib"

bundle = joblib.load(MODEL_FILE)

model = bundle["model"]
imputer = bundle["imputer"]
features = list(bundle["features"])

app = FastAPI(
    title="Air Pollution Prediction API",
    description="One-hour-ahead PM2.5 prediction using XGBoost",
    version="1.0"
)


class PredictionRequest(BaseModel):
    features: dict[str, float]


@app.get("/")
def home():
    return {
        "message": "Air Pollution ML API is working!",
        "model": "XGBoost",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "number_of_features": len(features)
    }


@app.get("/features")
def get_features():
    return {"required_features": features}


@app.get("/example")
def example():
    # Synthetic test data, NOT live pollution readings.
    hour = 17
    weekday = 4
    month = 11
    wind_direction = 90

    sample = {
        "Latitude": 17.385,
        "Longitude": 78.4867,
        "Temp_2m_C": 30,
        "Humidity_Percent": 55,
        "Wind_Speed_10m_kmh": 8,
        "Wind_Dir_10m": wind_direction,
        "Precipitation_mm": 0,
        "Pressure_MSL_hPa": 1012,
        "PM2_5_ugm3": 65,
        "PM10_ugm3": 100,
        "CO_ugm3": 650,
        "NO2_ugm3": 25,
        "SO2_ugm3": 8,
        "O3_ugm3": 40,
        "Dust_ugm3": 30,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * weekday / 7),
        "dow_cos": np.cos(2 * np.pi * weekday / 7),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        "wind_sin": np.sin(np.deg2rad(wind_direction)),
        "wind_cos": np.cos(np.deg2rad(wind_direction)),
        "pm25_lag_1": 60,
        "pm25_lag_3": 52,
        "pm25_lag_6": 48,
        "pm25_lag_12": 43,
        "pm25_lag_24": 57,
        "pm25_lag_48": 55,
        "pm25_lag_168": 45,
        "pm25_mean_6": 58,
        "pm25_mean_24": 54,
        "pm25_mean_168": 52
    }

    missing = [name for name in features if name not in sample]

    if missing:
        raise HTTPException(
            status_code=500,
            detail={"unsupported_example_features": missing}
        )

    return {
        "features": {
            name: float(sample[name]) for name in features
        }
    }


@app.post("/predict")
def predict(
    request: PredictionRequest = Body(
        openapi_examples={
            "xgboost_sample": {
                "summary": "Our 32 XGBoost Input Features",
                "description": "Example pollution and weather readings for testing.",
                "value": example()
            }
        }
    )
):
    missing = [
        name for name in features
        if name not in request.features
    ]

    extra = [
        name for name in request.features
        if name not in features
    ]

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": missing,
                "unexpected_features": extra
            }
        )

    input_data = pd.DataFrame(
        [[request.features[name] for name in features]],
        columns=features
    )

    if not np.isfinite(input_data.to_numpy()).all():
        raise HTTPException(
            status_code=422,
            detail="All values must be finite numbers."
        )

    # Apply the imputer saved during training.
    processed = imputer.transform(input_data).astype(np.float32)

    # Model was trained on log1p(PM2.5).
    log_prediction = float(model.predict(processed)[0])

    if not np.isfinite(log_prediction):
        raise HTTPException(
            status_code=500,
            detail="The model returned an invalid result."
        )

    prediction = max(0.0, float(np.expm1(log_prediction)))

    return {
        "model": "XGBoost",
        "prediction_type": "One-hour-ahead PM2.5",
        "predicted_pm25_ugm3": round(prediction, 2),
        "unit": "ug/m3"
    }
