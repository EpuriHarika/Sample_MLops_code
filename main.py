from fastapi import FastAPI
from app.model import load_model

app = FastAPI()
model = load_model()

@app.get("/")
def home():
    return {"message": "ML API is running"}

@app.post("/predict")
def predict(features: list):
    prediction = model.predict([features])
    return {"prediction": int(prediction[0])}