from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from keras.models import load_model
from io import BytesIO

app = FastAPI(title="Happy Mood Classifier API")
model = load_model("CNN_happy_model.h5")

@app.get("/")
def root():
    return {"message": "API happy/not happy. Use /predict."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = np.load(BytesIO(contents)) 
    img_batch = np.expand_dims(img, axis=0)
    pred = model.predict(img_batch)[0][0]
    label = "happy" if pred > 0.5 else "not_happy"
    confidence = float(pred) if pred > 0.5 else float(1 - pred)
    return JSONResponse(content={"label": label, "confidence": round(confidence, 3)})