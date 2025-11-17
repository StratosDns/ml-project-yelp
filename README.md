# 📌 Yelp Business Rating Prediction — Machine Learning Pipeline

This project builds a **scalable, end-to-end machine learning system** using the **Yelp Open Dataset** to:

- Predict the **true Yelp rating (1–5 stars)** of a business using recent customer reviews + metadata  
- Predict the **true star rating (1–5)** for individual reviews  
- Extract **feature and token importance** for explainability  
- Evaluate **in-distribution vs out-of-distribution** performance  
- Provide insights useful for business owners and analysts  

The system is designed with strong **engineering practices**, using modular CLI commands, reproducible preprocessing, efficient TF-IDF feature extraction, and baseline linear models.

---

# 🚀 Features

### ✔ Business-level prediction  
Predict the average Yelp rating of a business based on its **20 most recent reviews**.

### ✔ Review-level prediction  
Predict the star rating of **individual reviews** using a large-scale text classifier.

### ✔ Explainability (upcoming)  
Interpret the importance of review tokens and business features.

### ✔ Robust pipeline  
Trains and evaluates models using:
- TF-IDF text features  
- Tabular metadata features  
- Fusion of both  
- OOD testing (Florida as unseen state)

### ✔ Large-scale processing  
Handles:  
- **3.37M reviews**  
- **72k businesses**

---

# 📂 Project Structure

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
├── data_raw/                 # Raw Yelp dataset (ignored by Git)
├── data_proc/                # Processed parquet & ML features (ignored)
├── models/                   # Saved trained models (ignored)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🧠 Pipeline Overview

The entire project is runnable via:

```
python -m src.cli <COMMAND>
```

---

## 1. Prepare Dataset  
```
python -m src.cli prepare-data
```

- Loads Yelp JSONL files  
- Filters to PA, TN (train) and FL (OOD)  
- Joins reviews ↔ businesses  
- Cleans text, parses dates  
- Writes:
  - `business.parquet`
  - `reviews_joined.parquet`

---

## 2. Build Features  
```
python -m src.cli build-features
```

### Business features:
- Aggregate last **K = 20** reviews per business  
- TF-IDF (30k dimensions)  
- Tabular features (10 dims)  
- Saves:
  - `X_business_tfidf.npz`
  - `X_business_tab.npy`
  - `y_business_reg.npy`
  - `y_business_bin.npy`

### Review features:
- Downsample reviews: **3.37M → 800k**  
- TF-IDF (20k dimensions)  
- Saves:
  - `X_review_tfidf.npz`
  - `y_review_stars.npy`

---

## 3. Train Business Models  
```
python -m src.cli train-business
```

Trains:
- Ridge Regression (text)
- LightGBM Regression (tabular)
- Logistic Regression (success classifier)
- LightGBM Classifier (tabular)

Performs **fusion tuning**:
- Best weight α = **1.0** → text-only best

---

## 4. Train Review Model  
```
python -m src.cli train-review
```

Trains a **multinomial logistic regression** classifier using 800k TF-IDF vectors.

Accuracy: **71.8%**

---

## 5. Evaluate All Models  
```
python -m src.cli evaluate
```

Computes:
- RMSE, MAE, R²  
- ROC-AUC, PR-AUC, Brier  
- Full OOD evaluation (FL)

---

# 📊 Results Summary

## ⭐ Business Rating Regression

| Model | RMSE | MAE | R² |
|-------|--------|--------|---------|
| **Text (TF-IDF + Ridge)** | **0.3616** | 0.2832 | **0.8633** |
| Tabular (LightGBM) | 0.9394 | 0.7585 | 0.0776 |
| Fusion (α = 1.0) | 0.3616 | 0.2832 | 0.8633 |

➡ **Text model overwhelmingly outperforms metadata-based models.**

---

## ⭐ Business Success Classification (≥4 stars)

| Model | ROC-AUC | PR-AUC | Brier |
|--------|-----------|------------|------------|
| **Text** | **0.9652** | **0.9646** | 0.0766 |
| Tabular | 0.6256 | 0.6183 | 0.2319 |

➡ Text captures sentiment → strong classification ability.

---

## ⭐ Review-Level Prediction

- **Accuracy: 71.8%**
- 800k samples  
- Multinomial logistic regression  

➡ Robust baseline for explainability.

---

## 🌍 OOD Generalization (FL)
Performance in **Florida** closely matches PA/TN → good cross-state stability.

---

# ⚠️ Challenges Encountered

- 3.3M+ reviews → memory limits  
- TF-IDF matrices up to 30k dimensions  
- Need for downsampling and batching  
- Tabular features too weak for rating prediction  
- Long preprocessing times

---

# 🔮 Next Steps

- Add explainability (SHAP, LR coefficients)  
- Build local UI for predictions  
- Business rating exploration tool  
- Prepare final paper & presentation  

---

# 🙌 Credits

- Yelp Open Dataset  
- Python 3.13  
- Scikit-learn, LightGBM, SciPy, Pandas  
- Project developed as part of university coursework  