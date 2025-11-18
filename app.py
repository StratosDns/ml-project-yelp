import yaml
from pathlib import Path

import streamlit as st

from src.explain import make_business_explainer, make_review_explainer


# =========================
# Load config and explainers
# =========================

@st.cache_resource
def load_paths_and_explainers():
    with open("configs/paths.yml") as f:
        paths = yaml.safe_load(f)

    biz_expl = make_business_explainer(paths)
    rev_expl = make_review_explainer(paths)
    return paths, biz_expl, rev_expl


paths, biz_expl, rev_expl = load_paths_and_explainers()


# =========================
# Page layout
# =========================

st.set_page_config(
    page_title="Yelp Rating Prediction & Explainability",
    layout="wide",
)

st.title("Yelp Rating Prediction & Explainability")
st.write(
    "This app uses linear models on TF-IDF features to predict Yelp ratings "
    "and explain which words contribute most to the predictions."
)

tab_business, tab_review = st.tabs(["Business-Level", "Review-Level"])


# =========================
# Business-level tab
# =========================

with tab_business:
    st.header("Business-Level Rating Prediction")

    st.write(
        "Paste or construct a *business document* here (e.g., the last ~20 reviews "
        "for a business concatenated). The model will predict the business's rating "
        "and highlight the most positive and negative tokens."
    )

    default_biz_text = (
        "The food was amazing, service was fast and friendly. "
        "However, the place was a bit noisy on Friday night. "
        "Overall, great experience, would come again."
    )

    biz_text = st.text_area(
        "Business reviews text",
        value=default_biz_text,
        height=200,
    )

    top_k = st.slider(
        "Number of top positive/negative tokens to show",
        min_value=5,
        max_value=30,
        value=10,
    )

    if st.button("Explain business rating"):
        if not biz_text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Computing explanation..."):
                exp = biz_expl.explain_rating(biz_text, top_k=top_k)

            st.subheader("Prediction")
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted rating", f"{exp['pred_rating']:.3f}")
            col2.metric("Bias term", f"{exp['bias']:.3f}")
            col3.metric("Sum of contributions", f"{exp['total_contribution']:.3f}")

            st.subheader("Top Positive Tokens (increase rating)")
            pos_tokens, pos_vals = zip(*exp["top_positive"]) if exp["top_positive"] else ([], [])
            st.table(
                {
                    "token": pos_tokens,
                    "contribution": [f"{v:+.4f}" for v in pos_vals],
                }
            )

            st.subheader("Top Negative Tokens (decrease rating)")
            neg_tokens, neg_vals = zip(*exp["top_negative"]) if exp["top_negative"] else ([], [])
            st.table(
                {
                    "token": neg_tokens,
                    "contribution": [f"{v:+.4f}" for v in neg_vals],
                }
            )


# =========================
# Review-level tab
# =========================

with tab_review:
    st.header("Review-Level Rating Prediction")

    st.write(
        "Paste a single review text. The model predicts a star rating (1–5) and "
        "shows which tokens contributed most towards that predicted class."
    )

    default_review_text = "Service was terrible and the food was cold. Definitely not coming back."

    review_text = st.text_area(
        "Review text",
        value=default_review_text,
        height=200,
        key="review_text_area",
    )

    top_k_review = st.slider(
        "Number of top positive/negative tokens to show (for predicted class)",
        min_value=5,
        max_value=30,
        value=10,
        key="review_top_k",
    )

    if st.button("Explain review rating"):
        if not review_text.strip():
            st.warning("Please enter some review text.")
        else:
            with st.spinner("Computing explanation..."):
                exp = rev_expl.explain_review(review_text, top_k=top_k_review)

            st.subheader("Prediction")

            col1, col2 = st.columns(2)
            col1.metric("Predicted stars", str(exp["pred_class"]))
            col2.metric("Probability of predicted class", f"{exp['pred_class_prob']:.3f}")

            st.write("**Class probabilities:**")
            prob_rows = [
                {"stars": stars, "probability": f"{p:.3f}"}
                for stars, p in sorted(exp["class_probs"].items())
            ]
            st.table(prob_rows)

            st.subheader("Top Positive Tokens (for predicted class)")
            pos_tokens, pos_vals = zip(*exp["top_positive"]) if exp["top_positive"] else ([], [])
            st.table(
                {
                    "token": pos_tokens,
                    "contribution": [f"{v:+.4f}" for v in pos_vals],
                }
            )

            st.subheader("Top Negative Tokens (for predicted class)")
            neg_tokens, neg_vals = zip(*exp["top_negative"]) if exp["top_negative"] else ([], [])
            st.table(
                {
                    "token": neg_tokens,
                    "contribution": [f"{v:+.4f}" for v in neg_vals],
                }
            )

st.write("---")
st.caption(
    "Models: Ridge Regression & Multinomial Logistic Regression on TF-IDF features. "
    "This app is a mid-semester prototype for exploring rating prediction and explainability."
)
