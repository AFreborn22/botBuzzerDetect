import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
from sklearn.metrics.pairwise import cosine_similarity

pipeline = pickle.load(open('models/full_pipeline.pkl', 'rb'))

model = pipeline['model']
scaler = pipeline['scaler']
vectorizer = pipeline['vectorizer']
tfidf_matrix_train = pipeline['tfidf_matrix']  

# Inisialisasi FastAPI
app = FastAPI()

# Skema untuk input API
class Features(BaseModel):
    uniqueId: str
    text: str
    comment_count: int
    time_diff: float

# Fungsi preprocessing teks
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    words = word_tokenize(text)
    
    # Hapus stopwords
    stop_words = set(stopwords.words('indonesian'))
    words = [word for word in words if word not in stop_words]

    # Stemming
    stemmer = StemmerFactory().create_stemmer()
    words = [stemmer.stem(word) for word in words]

    return ' '.join(words)

# Endpoint prediksi
@app.post("/predict")
def predict(features: Features):
    try:
        # Preprocessing teks
        cleaned_text = preprocess_text(features.text)
        
        # Vectorization
        vectorized_text = vectorizer.transform([cleaned_text])
        
        # Perhitungan cosine similarity antara teks input dan teks di data latih
        cosine_sim_input = cosine_similarity(vectorized_text, tfidf_matrix_train)
        
        # Menghitung fitur max_cosine_sim
        max_cosine_sim = np.max(cosine_sim_input)

        # Menghitung similar_comments_count dengan threshold 0.7
        threshold = 0.7
        similar_comments_count = (cosine_sim_input > threshold).sum() - 1
        
        # Siapkan data input untuk model prediksi
        input_data = np.array([[max_cosine_sim, similar_comments_count, features.comment_count, features.time_diff]])
        scaled_input = scaler.transform(input_data)

        # Prediksi dengan model
        prediction = model.predict(scaled_input)

        # Konversi hasil prediksi dan fitur lainnya menjadi tipe Python standar
        prediction_result = int(prediction[0])
        max_cosine_sim_result = float(max_cosine_sim)
        similar_comments_count_result = int(similar_comments_count)

        return {
            "prediction": prediction_result,
            "max_cosine_sim": max_cosine_sim_result,
            "similar_comments_count": similar_comments_count_result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)