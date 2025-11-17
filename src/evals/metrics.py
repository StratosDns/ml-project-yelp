from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


def _load_business_data(proc_dir: Path):
    """
    Helper to load all business-level arrays and index.
    """
    from scipy import sparse as sp

    X_text = sp.load_npz(proc_dir / "X_business_tfidf.npz")
    X_tab = np.load(proc_dir / "X_business_tab.npy")
    y_reg = np.load(proc_dir / "y_business_reg.npy")
    y_bin = np.load(proc_dir / "y_business_bin.npy")
    idx = pd.read_parquet(proc_dir / "business_index.parquet")
    return X_text, X_tab, y_reg, y_bin, idx


def _fmt(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "n/a"
    return f"{x:.4f}"


def evaluate_business_models(paths: dict, params: dict) -> None:
    """
    Evaluate the business-level models on the full dataset.

    We compute:
      - Regression metrics (RMSE, MAE, R^2) for:
          * text-only regression
          * tabular-only regression
          * fused regression
      - Classification metrics (ROC-AUC, PR-AUC, Brier) for:
          * text-only classifier
          * tabular-only classifier

    We also optionally report metrics on a held-out OOD state,
    configured via params['business']['ood_state'].
    """
    proc_dir = Path(paths["proc_dir"])
    models_dir = Path(paths["models_dir"])

    X_text, X_tab, y_reg, y_bin, idx = _load_business_data(proc_dir)

    # Load models
    text_reg = joblib.load(models_dir / "business_text_reg.joblib")
    tab_reg = joblib.load(models_dir / "business_tab_reg.joblib")
    text_clf = joblib.load(models_dir / "business_text_clf.joblib")
    tab_clf = joblib.load(models_dir / "business_tab_clf.joblib")

    fusion_npz = np.load(models_dir / "business_fusion_reg.npz")
    alpha = float(fusion_npz["alpha"])

    # ----------------------------
    # Predictions
    # ----------------------------
    y_pred_text_reg = text_reg.predict(X_text)
    y_pred_tab_reg = tab_reg.predict(X_tab)
    y_pred_fused_reg = alpha * y_pred_text_reg + (1.0 - alpha) * y_pred_tab_reg

    p_text_clf = text_clf.predict_proba(X_text)[:, 1]
    p_tab_clf = tab_clf.predict_proba(X_tab)[:, 1]

    # ----------------------------
    # Regression metrics (overall)
    # ----------------------------
    def reg_metrics(y_true, y_pred):
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, r2

    rmse_t, mae_t, r2_t = reg_metrics(y_reg, y_pred_text_reg)
    rmse_tab, mae_tab, r2_tab = reg_metrics(y_reg, y_pred_tab_reg)
    rmse_f, mae_f, r2_f = reg_metrics(y_reg, y_pred_fused_reg)

    print("=== Regression — Overall ===")
    print(
        f"Text   : RMSE={_fmt(rmse_t)}  MAE={_fmt(mae_t)}  R^2={_fmt(r2_t)}"
    )
    print(
        f"Tabular: RMSE={_fmt(rmse_tab)}  MAE={_fmt(mae_tab)}  R^2={_fmt(r2_tab)}"
    )
    print(
        f"Fusion : RMSE={_fmt(rmse_f)}  MAE={_fmt(mae_f)}  R^2={_fmt(r2_f)}  (alpha={alpha:.2f})"
    )

    # ----------------------------
    # Classification metrics (overall)
    # ----------------------------
    def clf_metrics(y_true, p):
        if len(np.unique(y_true)) < 2:
            return None, None, None
        roc = roc_auc_score(y_true, p)
        pr = average_precision_score(y_true, p)
        brier = brier_score_loss(y_true, p)
        return roc, pr, brier

    roc_t, pr_t, br_t = clf_metrics(y_bin, p_text_clf)
    roc_tab, pr_tab, br_tab = clf_metrics(y_bin, p_tab_clf)

    print("\n=== Classification (success) — Overall ===")
    print(
        f"Text   : ROC-AUC={_fmt(roc_t)}  PR-AUC={_fmt(pr_t)}  Brier={_fmt(br_t)}"
    )
    print(
        f"Tabular: ROC-AUC={_fmt(roc_tab)}  PR-AUC={_fmt(pr_tab)}  Brier={_fmt(br_tab)}"
    )

    # ----------------------------
    # Optional OOD evaluation by state
    # ----------------------------
    business_cfg = params.get("business", {}) or {}
    ood_state = business_cfg.get("ood_state", None)
    if ood_state is not None:
        mask_ood = (idx["state"] == ood_state).values
        mask_in = ~mask_ood

        if mask_ood.sum() > 0:
            print(f"\n=== OOD State: {ood_state} ===")

            rmse_t_ood, mae_t_ood, r2_t_ood = reg_metrics(
                y_reg[mask_ood], y_pred_text_reg[mask_ood]
            )
            rmse_tab_ood, mae_tab_ood, r2_tab_ood = reg_metrics(
                y_reg[mask_ood], y_pred_tab_reg[mask_ood]
            )
            rmse_f_ood, mae_f_ood, r2_f_ood = reg_metrics(
                y_reg[mask_ood], y_pred_fused_reg[mask_ood]
            )

            print("Regression:")
            print(
                f"Text   : RMSE={_fmt(rmse_t_ood)}  MAE={_fmt(mae_t_ood)}  R^2={_fmt(r2_t_ood)}"
            )
            print(
                f"Tabular: RMSE={_fmt(rmse_tab_ood)}  MAE={_fmt(mae_tab_ood)}  R^2={_fmt(r2_tab_ood)}"
            )
            print(
                f"Fusion : RMSE={_fmt(rmse_f_ood)}  MAE={_fmt(mae_f_ood)}  R^2={_fmt(r2_f_ood)}"
            )

            roc_t_ood, pr_t_ood, br_t_ood = clf_metrics(
                y_bin[mask_ood], p_text_clf[mask_ood]
            )
            roc_tab_ood, pr_tab_ood, br_tab_ood = clf_metrics(
                y_bin[mask_ood], p_tab_clf[mask_ood]
            )

            print("Classification:")
            print(
                f"Text   : ROC-AUC={_fmt(roc_t_ood)}  PR-AUC={_fmt(pr_t_ood)}  Brier={_fmt(br_t_ood)}"
            )
            print(
                f"Tabular: ROC-AUC={_fmt(roc_tab_ood)}  PR-AUC={_fmt(pr_tab_ood)}  Brier={_fmt(br_tab_ood)}"
            )
        else:
            print(f"\n[Warning] No rows found for OOD state '{ood_state}'. Skipping OOD evaluation.")
