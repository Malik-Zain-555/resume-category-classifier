from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "hello"}

from pydantic import BaseModel
class Resume(BaseModel):
    text: str

import re
import numpy as np
import joblib

pipe = joblib.load("resume_pipeline.pkl")

def clean_resume(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)

    # 3. Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # 4. Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # 5. Collapse extra spaces and strip trailing/leading spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

@app.post("/predict")
def predict(resume: Resume):
    cleaned = clean_resume(resume.text)
    probs = pipe.predict_proba([cleaned])[0]
    best = np.argmax(probs)
    if probs[best] < 0.5:
        return {"category": "UNSURE", "confidence": float(probs[best])}

    return {"category": str(pipe.classes_[best]), "confidence": float(probs[best])}
