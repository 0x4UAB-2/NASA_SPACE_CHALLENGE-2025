from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np
import uvicorn
import pandas as pd
from io import StringIO
import csv

from model_api import load_models, pipeline_predict, pipeline_fit_and_predict

app = FastAPI()

load_models()

class Star(BaseModel):
    radius: float
    mass: float
    temperature: float

class Body(BaseModel):
    radius: float

class TransitParams(BaseModel):
    star: Star
    body: Body
    time_scaling_factor: float
    drag_duration: float
    impact_param: float


@app.post("/api/predict")
async def predict_action(request: Request, file: UploadFile | None = File(None)):
    # Read form data first
    form = await request.form()

    # Ensure we have a file either as the parameter or inside the form
    if file is None:
        # Try to get file from form (some clients put it in the form payload)
        possible = form.get('file')
        if possible is None:
            from fastapi.responses import JSONResponse
            return JSONResponse({'detail': 'file is required'}, status_code=400)
        file = possible

    # Read raw bytes from the uploaded file
    raw = await file.read()

    # Try decoding as utf-8 first, then fall back to latin-1
    try:
        text = raw.decode('utf-8')
    except Exception:
        try:
            text = raw.decode('utf-8', errors='replace')
        except Exception:
            text = raw.decode('latin-1', errors='replace')

    # Use pandas to read from StringIO with the detected delimiter
    try:
        df = pd.read_csv(StringIO(text), comment="#", sep=',')
    except pd.errors.ParserError:
        # Fall back to python engine and more permissive parsing
        df = pd.read_csv(StringIO(text), sep=',', engine='python', error_bad_lines=False)
    
    # form is a FormData / multidict. Use get/getlist for convenience.
    action = form.get('predict-action')
    if not action:
        raise ValueError("Missing predict-action in form")

    if action == "train-predict":
        model_ids = form.getlist('model_id')
        print("MODEL IDS: ", model_ids)
        df = pipeline_fit_and_predict(df, "./data/koi.csv", model_ids)
    elif action == "predict":
        # For single-model prediction, take the first model_id
        model_id = form.getlist('model_id')
        if not model_id:
            raise ValueError('model_id is required for predict')
        # Convert to int if possible
        try:
            model_id_int = int(model_id[0])
        except Exception:
            raise ValueError('model_id must be an integer')
        df = pipeline_predict(df, model_id_int)
    else:
        raise ValueError("Invalid predict-action")


    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    headers = {
        'Content-Disposition': 'attachment; filename="result.csv"'
    }

    return StreamingResponse(output, media_type='text/csv', headers=headers)

@app.post("/predict_basic")
def predict_body(features: dict):
    if features["koi_prad"] < 20:
        return {"prediction": "Exoplanet"}
    else:
        return {"prediction": "Stellar Companion"}

app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
