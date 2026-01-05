from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train_review_model(paths: dict, params: dict) -> None:
    """
    Train a review-level model that predicts the star rating (1–5) of a single review
    from its text.

    Model: Multinomial Logistic Regression on TF-IDF features.

    Inputs:
      - data_proc/X_review_tfidf.npz
      - data_proc/y_review_stars.npy

    Output:
      - models/review_logreg.joblib
    """
    proc_dir = Path(paths["proc_dir"])
    models_dir = Path(paths["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    from scipy import sparse as sp

    X = sp.load_npz(proc_dir / "X_review_tfidf.npz")
    y = np.load(proc_dir / "y_review_stars.npy")

    # We treat stars (typically 1..5) as discrete classes
    classes = np.unique(y)
    print(f"[train_review_model] Training on {X.shape[0]} reviews with classes: {classes}")

    review_cfg = params.get("review", {}) or {}
    max_iter = int(review_cfg.get("max_iter", 2000))
    C = float(review_cfg.get("C", 1.0))

    # Optionally create a small validation split which is useful but not strictly necessary 
    train_idx, val_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.1,
        random_state=42,
        stratify=y,
    )

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    clf = LogisticRegression(
        max_iter=max_iter,
        C=C,
        n_jobs=-1,
        multi_class="multinomial",
    )
    clf.fit(X_train, y_train)

    # Quick sanity check: accuracy on validation split
    val_acc = float((clf.predict(X_val) == y_val).mean())
    print(f"[train_review_model] Validation accuracy: {val_acc:.3f}")

    joblib.dump(clf, models_dir / "review_logreg.joblib")
    print(f"[train_review_model] Saved review-level model to {models_dir/'review_logreg.joblib'}")
