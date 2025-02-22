# main.py
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import numpy as np

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')

class Article(BaseModel):
    title: str
    abstract: str

class VectorResponse(BaseModel):
    vector: list[float]

class TextRequest(BaseModel):
    text: str

@app.post("/vectorize")
async def vectorize(request: TextRequest):
    text = request.text
    vector = model.encode(text)
    return vector.tolist()

@app.get("/test")
async def test_endpoint():
    return {"message": "Vector service is working!"}