import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('data/tiktokData.csv')

X = data[['max_cosine_sim', 'similar_comments_count', 'comment_count', 'time_diff']]
y = data['cluster']

vectorData = data['text']
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(vectorData)
cosine_sim = cosine_similarity(tfidf_matrix)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_scaled, y)

model_data = {
    'vectorizer': vectorizer,
    'cosine_similarity': cosine_sim
}

pickle.dump(model, open('models/rfc_model.pkl', 'wb'))
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
with open('models/cosineSim.pkl', 'wb') as f:
    pickle.dump(model_data, f)