# Yelp Business Rating & Review Sentiment — End-to-End ML Pipeline

This project trains machine learning models on the **Yelp Open Dataset** to:

1. **Predict a business's average rating (1–5 stars)** based on:
   - Aggregated recent reviews (text)
   - Business attributes (categories, review_count, etc.)

2. **Classify whether a business is "successful" (≥ 4★)**.

3. **Predict the rating of a single review (1–5 stars)** from its text  
   (this doubles as a fine-grained sentiment model).

The goal is to support:
- **Batch usage** (large files),
- **Single-review prediction**, and later
- A local UI (Streamlit or similar).

---

## Project Structure

```text
.
├── configs/
│   ├── paths.yml      # folders for raw/proc/models/figs
│   └── params.yml     # model & data parameters
├── data_raw/          # place Yelp JSON here
├── data_proc/         # processed parquet & feature matrices
├── models/            # saved models & vectorizers
├── figs/              # evaluation & explainability plots
├── src/
│   ├── cli.py
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── load_yelp.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── business_models.py
│   │   └── review_model.py
│   └── evals/
│       ├── __init__.py
│       └── metrics.py
├── requirements.txt
└── README.md
