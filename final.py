import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load data
data = pd.read_csv('data/tiktokData.csv')

# Features dan target
X = data[['max_cosine_sim', 'similar_comments_count', 'comment_count', 'time_diff']]
y = data['cluster']

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
vectorData = data['text']

# Preprocessing
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(vectorData)
cosine_sim = cosine_similarity(tfidf_matrix)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_scaled, y)

# Simpan keseluruhan pipeline, termasuk TF-IDF matrix dari data latih
pipeline = {
    'model': model,
    'scaler': scaler,
    'vectorizer': vectorizer,
    'tfidf_matrix': tfidf_matrix  # Simpan TF-IDF matrix
}

# Save pipeline to file
pickle.dump(pipeline, open('models/full_pipeline.pkl', 'wb'))

print("Model and preprocessing pipeline saved successfully!")