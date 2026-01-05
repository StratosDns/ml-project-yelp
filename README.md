# 📌 Yelp Business Rating Prediction — Machine Learning Pipeline

This project builds a **scalable, end-to-end ML system** on the **Yelp Open Dataset** to:

- Predict a business’s **true Yelp rating (1–5 stars)** from recent reviews + metadata  
- Predict the **1–5 star rating** of individual reviews  
- Provide **token-level explainability** for text models  
- Evaluate **in-distribution vs. out-of-distribution (OOD)** performance  
- Serve as a **reproducible, teaching-ready baseline** with clean engineering practices

---

## 🚀 Features

### ✔ Business-level prediction  
Predict the average Yelp rating using the **last K=20 reviews** per business.

### ✔ Review-level prediction  
Predict the star rating of **individual reviews** with a large-scale text classifier.

### ✔ Explainability  
Token-level attributions for linear text models (regression, binary, multiclass).

### ✔ Robust pipeline  
- TF–IDF text features (n-grams 1–2)  
- Tabular metadata features  
- Late fusion (text + tabular)  
- OOD testing (Florida as unseen state)

### ✔ Large-scale processing  
- **3.36M reviews**  
- **72k businesses**

### ✔ Language filtering  
English-only via fastText LID-176 (p(en) ≥ 0.8) + ASCII heuristic.

---

## 📂 Project Structure

```
ml-project-yelp/
│
├── src/
│   ├── cli.py                # Main CLI (entry point)
│   ├── data/                 # Data loading & preparation
│   ├── features/             # Feature extraction (TF-IDF, tabular)
│   ├── models/               # Training modules (business + reviews)
│   ├── evals/                # Evaluation scripts
│   └── utils/                # Shared utilities
│
├── configs/
│   ├── params.yml            # Hyperparameters & pipeline config
│   └── paths.yml             # Directory paths
│
├── data_raw/                 # Raw Yelp dataset (ignored)
├── data_proc/                # Processed parquet & ML features (ignored)
├── models/                   # Saved trained models (ignored)
│
├── figs/                     # Generated figures (optional)
├── requirements.txt
└── README.md
```

---

## 🧠 Pipeline Overview

Run any step via:
```
python -m src.cli <COMMAND>
```

### 1) Prepare Dataset
```
python -m src.cli prepare-data
```
- Loads Yelp JSONL  
- Filters to PA, TN (in-distribution) and FL (OOD)  
- Joins reviews ↔ businesses  
- Cleans text, parses dates  
- Applies English filter (ASCII prefilter + fastText LID-176, p(en) ≥ 0.8)  
- Writes `business.parquet`, `reviews_joined.parquet`

### 2) Build Features
```
python -m src.cli build-features
```
Business:
- Aggregate last **K=20** reviews per business (chronological)
- TF–IDF (30k dims), Tabular (10 dims)
- Saves: `X_business_tfidf.npz`, `X_business_tab.npy`, `y_business_reg.npy`, `y_business_bin.npy`

Review:
- Downsample reviews: **3.355M → 800k**
- TF–IDF (20k dims)
- Saves: `X_review_tfidf.npz`, `y_review_stars.npy`

### 3) Train Business Models
```
python -m src.cli train-business
```
- Ridge Regression (text)
- LightGBM Regression (tabular)
- Logistic Regression (success classifier, text)
- LightGBM Classifier (tabular)
- Fusion weight α tuned → **α = 1.0 (text only best)**

### 4) Train Review Model
```
python -m src.cli train-review
```
- Multinomial logistic regression on 800k TF–IDF vectors  
- Accuracy (val): **71.6%**

### 5) Evaluate
```
python -m src.cli evaluate
```
- Metrics: RMSE, MAE, R²; ROC-AUC, PR-AUC, Brier; accuracy (reviews)
- OOD evaluation (FL held out)

---

## 📊 Results Summary (English-filtered run)

### ⭐ Business Rating Regression
| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| **Text (TF–IDF + Ridge)** | **0.3619** | 0.2833 | **0.8631** |
| Tabular (LightGBM) | 0.9394 | 0.7585 | 0.0776 |
| Fusion (α = 1.0) | 0.3619 | 0.2833 | 0.8631 |

➡ **Text dominates; fusion adds no gain.**

### ⭐ Business Success Classification (≥4 stars)
| Model | ROC-AUC | PR-AUC | Brier |
|-------|---------|--------|-------|
| **Text (LogReg)** | **0.9657** | **0.9650** | 0.0762 |
| Tabular (LightGBM) | 0.6256 | 0.6183 | 0.2319 |

➡ Sentiment-rich text is decisive; tabular is weak.

### ⭐ Review-Level Prediction
- **Accuracy: 71.6%** (multinomial logistic regression, 800k TF–IDF)

### 🌍 OOD (Florida held out)
- Regression (text): RMSE 0.3610, MAE 0.2833, R² 0.8652  
- Classification (text): ROC-AUC 0.9652, PR-AUC 0.9666, Brier 0.0764  
➡ Near-parity with in-distribution (PA/TN).

### 🔍 Language Filter Ablation
- Without language ID (all reviews): regression RMSE 0.3616; classification ROC-AUC 0.9652; review accuracy 0.718.
- With English filter (fastText LID-176): dataset shrank to 3.355M; metrics essentially unchanged → non-English reviews were a small fraction.

---

## ⚙️ Key Settings
- States: PA, TN (ID); FL (OOD)
- Business aggregation: last K=20 reviews
- TF–IDF: n-grams (1,2); min_df=5; vocab sizes 30k (biz), 20k (review)
- Language: ASCII heuristic + fastText LID-176, p(en) ≥ 0.8
- Review cap: 800k (downsampled)
- Seeds: 42 for splits/TF–IDF ordering

---

## 📌 Tips
- Ensure `lid.176.bin` is present and `fasttext-wheel==0.9.2` is installed if language filtering is enabled.
- For plotting (confusion matrix, calibration, learning curve), run:
  ```
  python figs/analysis_plots.py
  ```
  (Uses saved models/features; Agg backend.)

---

## ⚠️ Challenges
- Memory/CPU for 3.3M+ reviews; mitigated via streaming JSONL, downsampling, sparse TF–IDF.
- Tabular features are intentionally minimal → weak performance (expected).

---

## 🔮 Possible Extensions
- Richer tabular signals (price, check-ins, temporal trends, user credibility)
- Calibrated probabilities (Platt/Isotonic)
- Temporal OOD splits; additional regions
- Contextual embeddings vs. TF–IDF
- Simple UI for interactive predictions

---

## 🙌 Credits
- Yelp Open Dataset  
- Python 3.13; scikit-learn; LightGBM; SciPy; pandas  
- Developed for university coursework (Machine Learning and Knowledge Extraction)