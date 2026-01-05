import matplotlib
matplotlib.use("Agg")  # non-GUI backend

import json
from pathlib import Path
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.metrics import root_mean_squared_error
import scipy.sparse as sp
import yaml

DATA_DIR = Path("data_proc")
MODELS_DIR = Path("models")

def plot_review_confusion():
    X_review = sp.load_npz(DATA_DIR / "X_review_tfidf.npz")
    y_review = np.load(DATA_DIR / "y_review_stars.npy")
    model = joblib.load(MODELS_DIR / "review_logreg.joblib")
    y_pred = model.predict(X_review)
    cm = confusion_matrix(y_review, y_pred, labels=[1, 2, 3, 4, 5], normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3, 4, 5])
    disp.plot(cmap="Blues", values_format=".2f")
    plt.title("Review rating confusion matrix (normalized)")
    plt.tight_layout()
    plt.savefig("fig_review_confusion.png", dpi=200)
    plt.close()

def plot_business_calibration():
    with open("configs/params.yml", "r") as f:
        params = yaml.safe_load(f)
    thr = float(params.get("success_threshold", 4.0))

    X_text = sp.load_npz(DATA_DIR / "X_business_tfidf.npz")
    bus_df = pd.read_parquet(DATA_DIR / "business.parquet")
    y_bin = (bus_df["stars"] >= thr).astype(int).to_numpy()

    clf = joblib.load(MODELS_DIR / "business_text_clf.joblib")
    prob_pos = clf.predict_proba(X_text)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_bin, prob_pos, n_bins=15, strategy="quantile")
    plt.plot(mean_pred, frac_pos, "o-", label="Text LogReg")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curve (business success)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_business_calibration.png", dpi=200)
    plt.close()

def plot_business_reg_learning_curve():
    from sklearn.model_selection import ShuffleSplit
    from sklearn.linear_model import Ridge

    X_text = sp.load_npz(DATA_DIR / "X_business_tfidf.npz")
    y_reg = np.load(DATA_DIR / "y_business_reg.npy")

    est = Ridge(alpha=1.0)
    cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        est, X_text, y_reg, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="neg_root_mean_squared_error",
        random_state=42,
    )
    train_rmse = -train_scores
    val_rmse = -val_scores
    plt.plot(train_sizes, train_rmse.mean(axis=1), "o-", label="Train RMSE")
    plt.fill_between(train_sizes,
                     train_rmse.mean(axis=1) - train_rmse.std(axis=1),
                     train_rmse.mean(axis=1) + train_rmse.std(axis=1),
                     alpha=0.2)
    plt.plot(train_sizes, val_rmse.mean(axis=1), "o-", label="Val RMSE")
    plt.fill_between(train_sizes,
                     val_rmse.mean(axis=1) - val_rmse.std(axis=1),
                     val_rmse.mean(axis=1) + val_rmse.std(axis=1),
                     alpha=0.2)
    plt.xlabel("Training examples")
    plt.ylabel("RMSE")
    plt.title("Learning curve (business regression, text Ridge)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_business_learning_curve.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    plot_review_confusion()
    plot_business_calibration()
    plot_business_reg_learning_curve()
    print("Saved: fig_review_confusion.png, fig_business_calibration.png, fig_business_learning_curve.png")