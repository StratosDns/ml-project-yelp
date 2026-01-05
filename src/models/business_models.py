from pathlib import Path

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LogisticRegression
from lightgbm import LGBMRegressor, LGBMClassifier


def _load_business_data(proc_dir: Path):
    """
    Helper to load all business-level feature arrays from disk.
    """
    from scipy import sparse as sp

    X_text = sp.load_npz(proc_dir / "X_business_tfidf.npz")  # sparse matrix (n, d_text)
    X_tab = np.load(proc_dir / "X_business_tab.npy")         # dense (n, d_tab)
    y_reg = np.load(proc_dir / "y_business_reg.npy")         # (n,)
    y_bin = np.load(proc_dir / "y_business_bin.npy")         # (n,)

    return X_text, X_tab, y_reg, y_bin


def train_business_models(paths: dict, params: dict) -> None:
    """
    Train business-level models:

      1) Text-only regression model (Ridge on TF-IDF)
      2) Tabular-only regression model (LGBMRegressor)
      3) Binary "success" classifier:
         - Text-only (LogisticRegression)
         - Tabular-only (LGBMClassifier)
      4) Simple fusion for regression (weighted average of text + tabular predictions).

    Models are saved under models_dir:

      - business_text_reg.joblib
      - business_tab_reg.joblib
      - business_text_clf.joblib
      - business_tab_clf.joblib
      - business_fusion_reg.npz (contains alpha + validation metrics)
    """
    proc_dir = Path(paths["proc_dir"])
    models_dir = Path(paths["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    X_text, X_tab, y_reg, y_bin = _load_business_data(proc_dir)

    # ----------------------------
    # Train/val split (only on in-distribution states)
    # For now, we do a standard random split.
    # We can later adapt this to a time-based or state-based split if desired.
    # It is completely up to future work goals
    # ----------------------------
    val_fraction = float(params.get("business", {}).get("val_fraction", 0.2))

    # For simplicity we use the same split for all tasks
    idx = np.arange(len(y_reg))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_fraction,
        random_state=42,
        shuffle=True,
        stratify=y_bin,  # keep success ratio balanced
    )

    X_text_train = X_text[train_idx]
    X_text_val = X_text[val_idx]
    X_tab_train = X_tab[train_idx]
    X_tab_val = X_tab[val_idx]

    y_reg_train = y_reg[train_idx]
    y_reg_val = y_reg[val_idx]
    y_bin_train = y_bin[train_idx]
    y_bin_val = y_bin[val_idx]

    # ----------------------------
    # 1) Text-only regression (Ridge)
    # ----------------------------
    print("[train_business_models] Training text regression model (Ridge)...")
    text_reg = Ridge(alpha=1.0, random_state=42)
    text_reg.fit(X_text_train, y_reg_train)
    joblib.dump(text_reg, models_dir / "business_text_reg.joblib")

    # ----------------------------
    # 2) Tabular-only regression (LGBMRegressor)
    # ----------------------------
    print("[train_business_models] Training tabular regression model (LGBMRegressor)...")
    tab_reg = LGBMRegressor(
        objective="regression",
        learning_rate=0.05,
        n_estimators=2000,
        num_leaves=63,
        max_depth=-1,
        min_data_in_leaf=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    tab_reg.fit(X_tab_train, y_reg_train, eval_set=[(X_tab_val, y_reg_val)], eval_metric="l2")
    joblib.dump(tab_reg, models_dir / "business_tab_reg.joblib")

    # ----------------------------
    # 3a) Text-only binary classifier (success vs not)
    # ----------------------------
    print("[train_business_models] Training text classifier (LogisticRegression)...")
    text_clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced",
    )
    text_clf.fit(X_text_train, y_bin_train)
    joblib.dump(text_clf, models_dir / "business_text_clf.joblib")

    # ----------------------------
    # 3b) Tabular-only binary classifier (LGBMClassifier)
    # ----------------------------
    print("[train_business_models] Training tabular classifier (LGBMClassifier)...")
    tab_clf = LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        n_estimators=2000,
        num_leaves=63,
        max_depth=-1,
        min_data_in_leaf=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    tab_clf.fit(X_tab_train, y_bin_train, eval_set=[(X_tab_val, y_bin_val)], eval_metric="auc")
    joblib.dump(tab_clf, models_dir / "business_tab_clf.joblib")

    # ----------------------------
    # 4) Fusion for regression
    #    We compute validation predictions from the two regression models,
    #    then choose a weight alpha that minimizes validation MSE.
    # ----------------------------
    print("[train_business_models] Tuning fusion weight for regression...")

    # Validation predictions
    y_pred_text_val = text_reg.predict(X_text_val)
    y_pred_tab_val = tab_reg.predict(X_tab_val)

    alphas = np.linspace(0.0, 1.0, 21)  # 0.0, 0.05, ..., 1.0
    best_alpha = 0.5
    best_mse = float("inf")

    for a in alphas:
        y_pred_fused = a * y_pred_text_val + (1.0 - a) * y_pred_tab_val
        mse = float(np.mean((y_reg_val - y_pred_fused) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_alpha = a

    print(
        f"[train_business_models] Best fusion alpha={best_alpha:.2f}  (val MSE={best_mse:.4f})"
    )

    # Save fusion info as a tiny npz file
    fusion_path = models_dir / "business_fusion_reg.npz"
    np.savez(
        fusion_path,
        alpha=best_alpha,
        val_mse=best_mse,
        y_reg_val=y_reg_val,
        y_pred_text_val=y_pred_text_val,
        y_pred_tab_val=y_pred_tab_val,
    )
    print(f"[train_business_models] Saved fusion info to {fusion_path}")
