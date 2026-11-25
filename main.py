import pickle
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "trained_pipeline-0.1.0.pkl", "rb") as f:
    model = pickle.load(f)

LANGUAGES = [
    "Arabic",
    "Danish",
    "Dutch",
    "English",
    "French",
    "German",
    "Greek",
    "Hindi",
    "Italian",
    "Kannada",
    "Malayalam",
    "Portugeese",
    "Russian",
    "Spanish",
    "Sweedish",
    "Tamil",
    "Turkish",
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextIn(BaseModel):
    text: str


@app.get("/")
def home():
    return {"status": "OK", "version": 2}


@app.post("/predict")
def predict(payload: TextIn):
    result = model.predict([payload.text.lower()])[0]

    if isinstance(result, str):
        language = result
    else:
        language = LANGUAGES[int(result)]

    return {"language": str(language)}
