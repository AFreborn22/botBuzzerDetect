import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string

# Load model dan scaler dari file pickle
model = pickle.load(open('models/rfc_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Inisialisasi FastAPI
app = FastAPI()

# Skema untuk fitur input
class Features(BaseModel):
    uniqueId: str
    text: str
    time_diff: float

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    text = text.lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('indonesian'))
    words = [word for word in words if word not in stop_words]
    
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    
    return ' '.join(stemmed_words)

# Endpoint prediksi
@app.post("/predict/")
async def predict(features: Features):
    try:
        raw_text = features.text
        processed_text = preprocess_text(raw_text)
        
        

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

# Jalankan server FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)