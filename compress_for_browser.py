# compress_for_browser.py
import joblib
import pickle
import gzip
import base64
import numpy as np

# Load your actual models
ridge = joblib.load("models/ridge_business.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

# Option A: Use fewer features (top 2000 most important)
# Get feature importance from ridge coefficients
coef_abs = np.abs(ridge.coef_)
top_indices = np.argsort(coef_abs)[-2000:]  # Top 2000 features

# Create reduced vectorizer vocabulary
vocab = vectorizer.vocabulary_
inv_vocab = {v: k for k, v in vocab.items()}
top_words = [inv_vocab[i] for i in top_indices if i in inv_vocab]

# Create new small vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
small_vectorizer = TfidfVectorizer(vocabulary=top_words)

# Retrain ridge on reduced features (or subset coefficients)
small_ridge_coef = ridge.coef_[top_indices]
small_ridge_intercept = ridge.intercept_

# Package for browser
model_package = {
    'coef': small_ridge_coef.tolist(),
    'intercept': float(small_ridge_intercept),
    'vocabulary': {word: float(vectorizer.idf_[vocab[word]]) 
                   for word in top_words if word in vocab}
}

# Save as compressed JSON
import json
model_json = json.dumps(model_package)
compressed = gzip.compress(model_json.encode())
b64 = base64.b64encode(compressed).decode()

print(f"Original model: ~{len(pickle.dumps(ridge))/1024:.0f} KB")
print(f"Compressed browser model: {len(b64)/1024:.0f} KB")

# Save to file
with open("models/browser_model.json.gz.b64", "w") as f:
    f.write(b64)

print(f"\nModel ready. Size: {len(b64)/1024:.1f} KB")