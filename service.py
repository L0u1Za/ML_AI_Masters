from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle
import numpy as np
import re
import uvicorn

app = FastAPI()

# Загрузка сохраненных моделей и объектов
with open('encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)

with open('scaler.pickle', 'rb') as f:
    scaler = pickle.load(f)

with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

# Классы для описания данных
class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

def process_torque(torque_str):
    if not isinstance(torque_str, str):  # Проверяем, является ли значение строкой
        return None, None

    match = re.search(r'(\d+\.?\d*)\s*(Nm|kgm)', torque_str, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit == 'kgm':
            value *= 9.8
    else:
        value = None

    match_rpm = re.search(r'(\d{1,3}(?:,\d{3})*|\d+)(?:-| to )?(\d{1,3}(?:,\d{3})*|\d+)?\s*rpm', torque_str, re.IGNORECASE)
    if match_rpm:
        min_rpm = int(match_rpm.group(1).replace(',', ''))
        max_rpm = int(match_rpm.group(2).replace(',', '')) if match_rpm.group(2) else min_rpm
    else:
        max_rpm = None

    return value, max_rpm

required_columns = [
        "name", "year", "km_driven", "fuel", "seller_type", "transmission",
        "owner", "mileage", "engine", "max_power", "torque", "seats"
    ]

# Функция предобработки данных
def preprocess_df(df: pd.DataFrame) -> np.ndarray:
    df['mileage'] = df['mileage'].str.extract(r'(\d+\.?\d*)').astype(float)
    df['engine'] = df['engine'].str.extract(r'(\d+\.?\d*)').astype(float)
    df['max_power'] = df['max_power'].str.extract(r'(\d+\.?\d*)').astype(float)

    df[['torque', 'max_torque_rpm']] = df["torque"].apply(lambda x: pd.Series(process_torque(x)))

    df['seats'] = df['seats'].astype(int)
    df['engine'] = df['engine'].astype(int)

    df['brand'] = df['name'].apply(lambda x: x.split(' ')[0])
    df.drop(['name'], axis=1, inplace=True)

    categorical_columns = df.select_dtypes(include='object').columns.append(pd.Index(['seats']))
    num_columns = [column for column in df.columns if column not in categorical_columns]
    encoded_features = encoder.transform(df[categorical_columns])
    df = df.drop(categorical_columns, axis=1)
    feature_names = encoder.get_feature_names_out(categorical_columns)
    df[feature_names] = encoded_features

    return df

@app.post("/predict_item")
def predict_item(item: Item):
    # Преобразуем данные в dfFrame
    df = pd.DataFrame([item.model_dump()])
    processed_df = preprocess_df(df)
    prediction = model.predict(processed_df)
    return {"predicted_price": prediction[0]}

@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)):
    # Загружаем CSV-файл в dfFrame
    df = pd.read_csv(file.file)

    # Проверяем, чтобы входные данные соответствовали требуемой структуре
    if not all(col in df.columns for col in required_columns):
        return {"error": "Некорректный формат входного файла"}

    # Предобработка и предсказание
    processed_df = preprocess_df(df)
    predictions = model.predict(processed_df)

    # Добавляем предсказания в dfFrame
    df['predicted_price'] = predictions

    # Сохраняем результат в новый CSV-файл
    output_file = "predicted_prices.csv"
    df.to_csv(output_file, index=False)

    return {"message": "Файл обработан", "output_file": output_file}

if __name__ == "__main__":
    uvicorn.run("service:app", host="0.0.0.0", port=8000, reload=True)