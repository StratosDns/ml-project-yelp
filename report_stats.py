import json
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from src.explain import make_business_explainer, make_review_explainer
from src.features.features import _clean_text_series

def load_cfg():
    with open("configs/paths.yml", "r") as f:
        paths = yaml.safe_load(f)
    with open("configs/params.yml", "r") as f:
        params = yaml.safe_load(f)
    return paths, params

def main():
    paths, params = load_cfg()
    proc_dir = Path(paths["proc_dir"])

    # --- Load prepared data ---
    bus_df = pd.read_parquet(proc_dir / "business.parquet")
    rev_df = pd.read_parquet(proc_dir / "reviews_joined.parquet")

    # --- Basic counts ---
    n_bus = len(bus_df)
    n_rev = len(rev_df)
    rev_per_bus = rev_df.groupby("business_id").size()
    rev_per_bus_stats = rev_per_bus.describe(percentiles=[0.5, 0.9, 0.99]).to_dict()

    # --- Success prevalence ---
    thr = float(params.get("success_threshold", 4.0))
    if "stars_biz" in rev_df.columns:
        bus_stars = (
            rev_df.groupby("business_id")["stars_biz"]
            .first()
            .rename("stars_biz")
            .to_frame()
        )
    else:
        bus_stars = bus_df[["business_id", "stars"]].rename(columns={"stars": "stars_biz"}).set_index("business_id")
    success_rate = (bus_stars["stars_biz"] >= thr).mean()

    # --- Review star distribution ---
    star_counts = rev_df["stars"].value_counts().sort_index()
    star_dist = (star_counts / star_counts.sum()).to_dict()

    # --- OOD split info ---
    states_cfg = params.get("states", [])
    ood_state = params.get("business", {}).get("ood_state", None)
    state_counts = rev_df["state"].value_counts().to_dict()

    # --- Review-level sample size after feature build ---
    y_review = np.load(proc_dir / "y_review_stars.npy")
    n_review_feat = len(y_review)

    # --- Business-level sample size after feature build ---
    y_bus_reg = np.load(proc_dir / "y_business_reg.npy")
    n_bus_feat = len(y_bus_reg)

    # --- Aggregated business docs (to mirror feature build) ---
    K = int(params.get("K_reviews_per_business", 20))
    rev_df_sorted = rev_df.sort_values("date")
    tail = rev_df_sorted.groupby("business_id").tail(K)
    agg = (
        tail.groupby("business_id")
        .agg(text=("text", lambda s: " \n ".join(s.tail(K).tolist())))
        .reset_index()
    )

    # Picking a sample business document for explainability demo
    sample_biz = agg.iloc[0]
    sample_biz_text = _clean_text_series(pd.Series([sample_biz["text"]])).iloc[0]

    # --- Explainability: business ---
    biz_explainer = make_business_explainer(paths)
    biz_rating_exp = biz_explainer.explain_rating(sample_biz_text, top_k=8)
    biz_success_exp = biz_explainer.explain_success_probability(sample_biz_text, top_k=8)

    # --- Explainability: review ---
    sample_review = rev_df["text"].iloc[0]
    review_explainer = make_review_explainer(paths)
    review_exp = review_explainer.explain_review(sample_review, top_k=8)

    # --- Print summary ---
    print("\n=== DATA STATS ===")
    print(f"Businesses (prepared): {n_bus}")
    print(f"Reviews (prepared):    {n_rev}")
    print(f"Reviews per business (count describe): {json.dumps(rev_per_bus_stats, indent=2)}")
    print(f"Business success prevalence (stars_biz >= {thr}): {success_rate:.4f}")
    print(f"Review star distribution: {json.dumps(star_dist, indent=2)}")
    print(f"States in config: {states_cfg}, OOD state: {ood_state}")
    print(f"Reviews by state: {json.dumps(state_counts, indent=2)}")
    print(f"Feature rows — business: {n_bus_feat}, review: {n_review_feat}")

    print("\n=== EXPLAINABILITY (BUSINESS RATING) ===")
    print(f"Sample business_id: {sample_biz['business_id']}")
    print(f"Pred rating: {biz_rating_exp['pred_rating']:.3f}, bias: {biz_rating_exp['bias']:.3f}")
    print("Top + tokens:", biz_rating_exp["top_positive"])
    print("Top - tokens:", biz_rating_exp["top_negative"])

    print("\n=== EXPLAINABILITY (BUSINESS SUCCESS PROB) ===")
    print(f"Prob success: {biz_success_exp['prob_success']:.3f}")
    print("Top + tokens:", biz_success_exp["top_positive"])
    print("Top - tokens:", biz_success_exp["top_negative"])

    print("\n=== EXPLAINABILITY (REVIEW MULTICLASS) ===")
    print(f"Pred class: {review_exp['pred_class']}, prob: {review_exp['pred_class_prob']:.3f}")
    print("Class probs (truncated):", review_exp["class_probs"])
    print("Top + tokens:", review_exp["top_positive"])
    print("Top - tokens:", review_exp["top_negative"])

if __name__ == "__main__":
    main()