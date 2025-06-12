from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import re
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка обученных моделей
models = {
    '20_29': joblib.load('models/model_data_20_29.joblib'),
    '30_39': joblib.load('models/model_data_30_39.joblib'),
    '40_49': joblib.load('models/model_data_40_49.joblib'),
    '50_69': joblib.load('models/model_data_50_69.joblib'),
    '70': joblib.load('models/model_data_70.joblib'),
}
mae_values = {
    '20_29': 6.98,
    '30_39': 8.82,
    '40_49': 10.36,
    '50_69': 13.47,
    '70': 11.55
}

def parse_gestalt(value):
    if isinstance(value, str):
        if '+' in value:
            base = int(value.replace('+', '').strip())
            return (base + 80) / 2
        nums = re.findall(r'\d+', value)
        if len(nums) == 2:
            return (int(nums[0]) + int(nums[1])) / 2
        elif len(nums) == 1:
            return float(nums[0])
    elif isinstance(value, (int, float)):
        return float(value)
    return np.nan

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    new_data = pd.read_excel(BytesIO(contents))
    results = []

    for _, row in new_data.iterrows():
        gestalt_raw = row.get('gestalt', np.nan)
        gestalt_age = parse_gestalt(gestalt_raw)
        if gestalt_age <= 29:
            group = '20_29'
        elif gestalt_age <= 39:
            group = '30_39'
        elif gestalt_age <= 49:
            group = '40_49'
        elif gestalt_age >= 70:
            group = '70'
        else:
            group = '50_69'

        model = models[group]
        mae = mae_values[group]
        X_sample = row.drop(labels=['ID', 'gestalt'], errors='ignore').to_frame().T
        y_pred = model.predict(X_sample)[0]

        error_margin = 1.96 * mae
        ci_lower = max(y_pred - error_margin, 16)
        ci_upper = min(y_pred + error_margin, 100)
        mean_age = (y_pred + gestalt_age) / 2

        result_row = row.to_dict()
        result_row.update({
            'gestalt_numeric': gestalt_age,
            'model_group': group,
            'Predicted_AGE': y_pred,
            'CI_95_lower': ci_lower,
            'CI_95_upper': ci_upper,
            'mean_age': mean_age
        })
        results.append(result_row)

    df_results = pd.DataFrame(results)
    output = BytesIO()
    df_results.to_excel(output, index=False)
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=predictions.xlsx"}
    )