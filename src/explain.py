"""
Explainability utilities for the Yelp rating project.

This module provides:
- Business-level explanations (text-based regression + classifier)
- Review-level explanations (multiclass rating classifier)

Core idea:
For linear models on TF-IDF features, predictions are essentially:

    y_hat = bias + w · x

where:
- x is the sparse TF-IDF vector for the text
- w are the learned coefficients
- bias is the intercept

Each feature j contributes:  contrib_j = w_j * x_j

We:
- Map TF-IDF feature indices back to tokens (words or n-grams)
- Compute per-token contributions
- Sort them to show top positive and negative tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from pathlib import Path

import numpy as np
import joblib
from scipy import sparse


# ============================
# Helpers for linear explainers
# ============================

def _compute_linear_contributions(
    X_row: sparse.spmatrix,
    coef: np.ndarray,
    feature_names: np.ndarray,
    top_k: int = 15,
) -> Dict[str, Any]:
    """
    Compute per-token contributions for a single sparse row and a linear model.

    Parameters
    ----------
    X_row : sparse.spmatrix, shape (1, n_features)
        TF-IDF vector for a single document.
    coef : np.ndarray, shape (n_features,)
        Coefficient vector for the model (for one target or one class).
    feature_names : np.ndarray, shape (n_features,)
        Feature names from the vectorizer.
    top_k : int
        How many top positive/negative tokens to return.

    Returns
    -------
    dict with:
        - contributions: list[(token, contrib_value)]
        - top_positive: list[(token, contrib_value)]
        - top_negative: list[(token, contrib_value)]
    """
    if not sparse.isspmatrix(X_row):
        raise ValueError("X_row must be a sparse row vector")

    X_row = X_row.tocsr()
    indices = X_row.indices
    data = X_row.data

    # contributions for the non-zero features
    local_coefs = coef[indices]
    contribs = data * local_coefs  # element-wise

    # map indices → tokens
    tokens = feature_names[indices]

    # sort by contribution
    order = np.argsort(contribs)  # ascending
    negative_idx = order[:top_k]
    positive_idx = order[-top_k:][::-1]

    top_negative = [(tokens[i], float(contribs[i])) for i in negative_idx]
    top_positive = [(tokens[i], float(contribs[i])) for i in positive_idx]

    # all contributions for reference
    all_contribs = [(tok, float(c)) for tok, c in zip(tokens, contribs)]

    return {
        "contributions": all_contribs,
        "top_positive": top_positive,
        "top_negative": top_negative,
    }


# ==================================
# Business-level text model explainer
# ==================================

@dataclass
class BusinessTextExplainer:
    """
    Explainer for business-level text models.

    Uses:
    - Ridge regression for rating prediction
    - Logistic Regression for success classification (rating >= threshold)
    """
    proc_dir: Path
    models_dir: Path

    # These are loaded lazily
    _vec = None
    _reg = None
    _clf = None

    def _load_vectorizer(self):
        if self._vec is None:
            self._vec = joblib.load(self.proc_dir / "business_tfidf_vectorizer.joblib")
        return self._vec

    def _load_regressor(self):
        if self._reg is None:
            self._reg = joblib.load(self.models_dir / "business_text_reg.joblib")
        return self._reg

    def _load_classifier(self):
        if self._clf is None:
            self._clf = joblib.load(self.models_dir / "business_text_clf.joblib")
        return self._clf

    def explain_rating(
        self,
        text: str,
        top_k: int = 15,
    ) -> Dict[str, Any]:
        """
        Explain the business-level predicted rating from the Ridge regressor.

        Parameters
        ----------
        text : str
            Aggregated business document (e.g., last K reviews concatenated).
        top_k : int
            Number of top positive/negative tokens to return.

        Returns
        -------
        dict with keys:
            - pred_rating: float
            - bias: float
            - total_contribution: float
            - top_positive: list[(token, contrib)]
            - top_negative: list[(token, contrib)]
            - raw: detailed dict from _compute_linear_contributions
        """
        vec = self._load_vectorizer()
        reg = self._load_regressor()

        X = vec.transform([text])
        pred = float(reg.predict(X)[0])

        # For Ridge, coef_.shape = (1, n_features)
        coef = reg.coef_.ravel()
        feature_names = np.array(vec.get_feature_names_out())

        contrib_info = _compute_linear_contributions(
            X_row=X,
            coef=coef,
            feature_names=feature_names,
            top_k=top_k,
        )

        bias = float(reg.intercept_[0] if np.ndim(reg.intercept_) > 0 else reg.intercept_)
        total_contribution = sum(c for _, c in contrib_info["contributions"])

        return {
            "pred_rating": pred,
            "bias": bias,
            "total_contribution": total_contribution,
            "top_positive": contrib_info["top_positive"],
            "top_negative": contrib_info["top_negative"],
            "raw": contrib_info,
        }

    def explain_success_probability(
        self,
        text: str,
        top_k: int = 15,
    ) -> Dict[str, Any]:
        """
        Explain the business-level success probability from the Logistic Regression model.

        Assumes a binary classifier with classes [0, 1] where:
        - 1 = success (rating >= threshold)
        - 0 = not success

        We attribute contributions to the logit for the positive class.
        """
        vec = self._load_vectorizer()
        clf = self._load_classifier()

        X = vec.transform([text])
        probs = clf.predict_proba(X)[0]
        classes = clf.classes_
        # assume positive class is 1
        pos_idx = int(np.where(classes == 1)[0][0])
        prob_success = float(probs[pos_idx])

        # decision_function returns logit(s)
        # For binary, shape is (n_samples,)
        logits = clf.decision_function(X)
        if logits.ndim == 1:
            logit = float(logits[0])
            coef = clf.coef_.ravel()
        else:
            # Fallback, treat as one-vs-rest and pick positive row
            logit = float(logits[0, pos_idx])
            coef = clf.coef_[pos_idx]

        feature_names = np.array(vec.get_feature_names_out())
        contrib_info = _compute_linear_contributions(
            X_row=X,
            coef=coef,
            feature_names=feature_names,
            top_k=top_k,
        )

        return {
            "prob_success": prob_success,
            "logit": logit,
            "top_positive": contrib_info["top_positive"],
            "top_negative": contrib_info["top_negative"],
            "raw": contrib_info,
        }


# ===============================
# Review-level text model explainer
# ===============================

@dataclass
class ReviewTextExplainer:
    """
    Explainer for the review-level rating classifier.

    Uses the multinomial Logistic Regression model trained on review TF-IDF
    to predict a rating in {1, 2, 3, 4, 5} and attribute contributions of
    each token for the predicted class.
    """
    proc_dir: Path
    models_dir: Path

    _vec = None
    _clf = None

    def _load_vectorizer(self):
        if self._vec is None:
            self._vec = joblib.load(self.proc_dir / "review_tfidf_vectorizer.joblib")
        return self._vec

    def _load_classifier(self):
        if self._clf is None:
            self._clf = joblib.load(self.models_dir / "review_logreg.joblib")
        return self._clf

    def explain_review(
        self,
        text: str,
        top_k: int = 15,
    ) -> Dict[str, Any]:
        """
        Explain the predicted star rating for a single review text.

        Parameters
        ----------
        text : str
            Single review text.
        top_k : int
            Number of top positive/negative tokens to return for the predicted class.

        Returns
        -------
        dict with keys:
            - pred_class: int (predicted star rating)
            - class_probs: dict[class -> probability]
            - top_positive: list[(token, contrib)]
            - top_negative: list[(token, contrib)]
            - raw: details from _compute_linear_contributions
        """
        vec = self._load_vectorizer()
        clf = self._load_classifier()

        X = vec.transform([text])
        probs = clf.predict_proba(X)[0]
        classes = clf.classes_.astype(int)

        pred_idx = int(np.argmax(probs))
        pred_class = int(classes[pred_idx])
        pred_prob = float(probs[pred_idx])

        # For multinomial logistic regression:
        # coef_.shape = (n_classes, n_features)
        coef = clf.coef_[pred_idx]

        feature_names = np.array(vec.get_feature_names_out())

        contrib_info = _compute_linear_contributions(
            X_row=X,
            coef=coef,
            feature_names=feature_names,
            top_k=top_k,
        )

        class_probs = {int(c): float(p) for c, p in zip(classes, probs)}

        return {
            "pred_class": pred_class,
            "pred_class_prob": pred_prob,
            "class_probs": class_probs,
            "top_positive": contrib_info["top_positive"],
            "top_negative": contrib_info["top_negative"],
            "raw": contrib_info,
        }


# ============================
# Convenience factory functions
# ============================

def make_business_explainer(paths: dict) -> BusinessTextExplainer:
    """
    Create a BusinessTextExplainer from the paths.yml dict.
    """
    proc_dir = Path(paths["proc_dir"])
    models_dir = Path(paths["models_dir"])
    return BusinessTextExplainer(proc_dir=proc_dir, models_dir=models_dir)


def make_review_explainer(paths: dict) -> ReviewTextExplainer:
    """
    Create a ReviewTextExplainer from the paths.yml dict.
    """
    proc_dir = Path(paths["proc_dir"])
    models_dir = Path(paths["models_dir"])
    return ReviewTextExplainer(proc_dir=proc_dir, models_dir=models_dir)
