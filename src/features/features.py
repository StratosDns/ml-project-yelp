from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def _clean_text_series(series: pd.Series) -> pd.Series:
    """
    Basic text cleaning for English reviews:
      - lowercasing
      - remove URLs
      - keep only letters and spaces
      - collapse multiple spaces

    This is deliberately simple and fast. It is good enough for TF-IDF
    models and keeps the preprocessing easy to explain in your report.
    """
    s = series.fillna("").astype(str).str.lower()

    # Remove URLs
    s = s.str.replace(r"http\S+|www\.\S+", " ", regex=True)
    # Keep only letters and spaces
    s = s.str.replace(r"[^a-z\s]", " ", regex=True)
    # Collapse multiple spaces into one
    s = s.str.replace(r"\s+", " ", regex=True)
    # Trim trailing/leading spaces
    s = s.str.strip()

    # Avoid empty docs (TF-IDF does not like empty strings)
    s = s.mask(s == "", "placeholdertoken")
    return s


# -------------------------------------------------------------------
# Business-level features (one row per business)
# -------------------------------------------------------------------
def build_business_features(paths: dict, params: dict) -> None:
    """
    Build features for the business-level models.

    Uses the joined table (reviews + business) to:
      1. Aggregate the last K reviews per business into a single document.
      2. Build a TF-IDF matrix of aggregated texts.
      3. Build simple tabular features from business categories and review_count.
      4. Save:
         - X_business_tfidf.npz     : sparse TF-IDF matrix
         - X_business_tab.npy       : tabular features
         - y_business_reg.npy       : regression target (stars_biz)
         - y_business_bin.npy       : binary success label (stars_biz >= threshold)
         - business_index.parquet   : business_id, city, state (and possibly others)
         - business_tfidf_vectorizer.joblib
         - business_tabular_cols.json
    """
    proc_dir = Path(paths["proc_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(proc_dir / "reviews_joined.parquet")

    # Ensure we have business-level star ratings
    if "stars_biz" not in df.columns:
        raise RuntimeError("Column 'stars_biz' not found in joined dataframe.")

    # ----------------------------
    # 1. Aggregate last K reviews per business
    # ----------------------------
    K = int(params.get("K_reviews_per_business", 20))

    df = df.sort_values("date")
    # Take last K reviews per business (in time order)
    tail = df.groupby("business_id").tail(K)

    agg = (
        tail.groupby("business_id")
        .agg(
            text=("text", lambda s: " \n ".join(s.tail(K).tolist())),
            stars_biz=("stars_biz", "first"),
            city=("city", "first"),
            state=("state", "first"),
            categories=("categories", "first"),
            review_count=("review_count", "first"),
        )
        .reset_index()
    )

    # ----------------------------
    # 2. Targets: regression + binary
    # ----------------------------
    y_reg = agg["stars_biz"].astype(float).values
    thr = float(params.get("success_threshold", 4.0))
    y_bin = (agg["stars_biz"] >= thr).astype(int).values

    # ----------------------------
    # 3. TF-IDF features (business-level)
    # ----------------------------
    texts = _clean_text_series(agg["text"])

    tfidf_cfg = params.get("tfidf_business", {}) or {}
    vectorizer = TfidfVectorizer(
        max_features=tfidf_cfg.get("max_features", 30000),
        min_df=tfidf_cfg.get("min_df", 5),
        ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
        strip_accents="ascii",
        token_pattern=r"(?u)\b\w+\b",
    )

    X_tfidf = vectorizer.fit_transform(texts.tolist())

    # ----------------------------
    # 4. Tabular features (categories & review_count)
    # ----------------------------
    def _has_token(cats, token: str) -> int:
        if not isinstance(cats, str):
            return 0
        return int(token.lower() in cats.lower())

    # A small, interpretable category vocabulary.
    tokens = [
        "Restaurants",
        "Fast Food",
        "Coffee & Tea",
        "Bars",
        "Nightlife",
        "Italian",
        "Chinese",
        "Mexican",
        "Burgers",
    ]

    for t in tokens:
        col_name = "cat_" + t.lower().replace(" & ", "_").replace(" ", "_")
        agg[col_name] = agg["categories"].apply(lambda x, tok=t: _has_token(x, tok))

    # Log-transform review_count (avoid log(0) with +1)
    agg["review_count_log1p"] = np.log1p(agg["review_count"].fillna(0).astype(float))

    feat_cols = [c for c in agg.columns if c.startswith("cat_")] + ["review_count_log1p"]
    X_tab = agg[feat_cols].fillna(0).astype(float).values

    # ----------------------------
    # 5. Save everything
    # ----------------------------
    from scipy import sparse as sp

    sp.save_npz(proc_dir / "X_business_tfidf.npz", X_tfidf)
    np.save(proc_dir / "X_business_tab.npy", X_tab)
    np.save(proc_dir / "y_business_reg.npy", y_reg)
    np.save(proc_dir / "y_business_bin.npy", y_bin)

    # Save business index (allows us to know which row corresponds to which business)
    idx_cols = ["business_id", "city", "state", "stars_biz", "review_count"]
    agg[idx_cols].to_parquet(proc_dir / "business_index.parquet", index=False)

    # Save vectorizer and tabular column names
    joblib.dump(vectorizer, proc_dir / "business_tfidf_vectorizer.joblib")
    with (proc_dir / "business_tabular_cols.json").open("w") as f:
        json.dump(feat_cols, f, indent=2)

    print(f"[build_business_features] TF-IDF shape: {X_tfidf.shape}")
    print(f"[build_business_features] Tabular shape: {X_tab.shape}")
    print(f"[build_business_features] Saved business-level targets & index.")


# -------------------------------------------------------------------
# Review-level features (one row per review)
# -------------------------------------------------------------------
def build_review_features(paths: dict, params: dict) -> None:
    """
    Build features for the review-level model.

    We treat individual reviews as samples and their star rating (1–5) as labels.

    To keep memory and runtime under control on large Yelp subsets, we:
      - optionally downsample to at most `review.max_reviews` reviews
        (e.g., 800k) in a reproducible way.
      - use a slightly smaller TF-IDF vocabulary than the business model.

    Output:
      - X_review_tfidf.npz          (sparse TF-IDF matrix)
      - y_review_stars.npy          (integer star labels 1..5)
      - review_index.parquet        (review_id, business_id, stars, date)
      - review_tfidf_vectorizer.joblib
    """
    proc_dir = Path(paths["proc_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(proc_dir / "reviews_joined.parquet")

    # We only need text + review-level stars for this task
    df = df.dropna(subset=["text", "stars"])

    review_cfg = params.get("review", {}) or {}
    max_reviews = review_cfg.get("max_reviews", None)

    # ----------------------------
    # Optional downsampling
    # ----------------------------
    if max_reviews is not None:
        max_reviews = int(max_reviews)
        n_reviews = len(df)
        if n_reviews > max_reviews:
            # Randomly sample a subset of reviews for the review-level model.
            # This keeps training tractable while still using a huge amount of data.
            print(
                f"[build_review_features] Downsampling reviews: "
                f"{n_reviews} -> {max_reviews}"
            )
            df = df.sample(n=max_reviews, random_state=42).reset_index(drop=True)

    texts = _clean_text_series(df["text"])
    y_stars = df["stars"].astype(int).values  # typical Yelp stars are integers 1..5

    tfidf_cfg = params.get("tfidf_review", {}) or {}
    vectorizer = TfidfVectorizer(
        max_features=tfidf_cfg.get("max_features", 20000),
        min_df=tfidf_cfg.get("min_df", 5),
        ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
        strip_accents="ascii",
        token_pattern=r"(?u)\b\w+\b",
    )

    print("[build_review_features] Fitting TF-IDF vectorizer on review texts...")
    X_tfidf = vectorizer.fit_transform(texts.tolist())

    from scipy import sparse as sp

    sp.save_npz(proc_dir / "X_review_tfidf.npz", X_tfidf)
    np.save(proc_dir / "y_review_stars.npy", y_stars)

    # Minimal index for analysis/debugging
    idx_cols = ["review_id", "business_id", "stars", "date"]
    df[idx_cols].to_parquet(proc_dir / "review_index.parquet", index=False)

    joblib.dump(vectorizer, proc_dir / "review_tfidf_vectorizer.joblib")

    print(f"[build_review_features] TF-IDF shape: {X_tfidf.shape}")
    print(f"[build_review_features] Saved review-level labels & index.")
