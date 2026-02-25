from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(title="Yelp Review Rating Demo API")

# TODO: Replace with your Vercel domain(s) later
ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["*"],
)

BASE = Path(__file__).resolve().parent
ART = BASE.parent / "demo_artifacts"

VEC = joblib.load(ART / "review_tfidf_vectorizer.joblib")
CLF = joblib.load(ART / "review_logreg.joblib")
FEATURES = np.array(VEC.get_feature_names_out())

class ReviewIn(BaseModel):
    text: str
    top_k: int = 10

def top_token_contribs(X_row, coef, top_k: int):
    X_row = X_row.tocsr()
    idx = X_row.indices
    data = X_row.data

    contribs = data * coef[idx]
    tokens = FEATURES[idx]

    order = np.argsort(contribs)
    neg_i = order[:top_k]
    pos_i = order[-top_k:][::-1]

    top_neg = [(tokens[i], float(contribs[i])) for i in neg_i]
    top_pos = [(tokens[i], float(contribs[i])) for i in pos_i]
    return top_pos, top_neg

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict/review")
def predict_review(inp: ReviewIn):
    text = (inp.text or "").strip()
    if not text:
        return {"error": "Empty text"}

    top_k = max(5, min(25, int(inp.top_k)))

    X = VEC.transform([text])
    probs = CLF.predict_proba(X)[0]
    classes = CLF.classes_.astype(int)

    pred_idx = int(np.argmax(probs))
    pred_class = int(classes[pred_idx])

    coef = CLF.coef_[pred_idx]
    top_pos, top_neg = top_token_contribs(X, coef, top_k)

    return {
        "pred_class": pred_class,
        "pred_class_prob": float(probs[pred_idx]),
        "class_probs": {int(c): float(p) for c, p in zip(classes, probs)},
        "top_positive": top_pos,
        "top_negative": top_neg,
    }