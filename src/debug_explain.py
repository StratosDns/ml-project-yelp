import yaml
from pathlib import Path
from src.explain import make_business_explainer, make_review_explainer

with open("configs/paths.yml") as f:
    paths = yaml.safe_load(f)

biz_expl = make_business_explainer(paths)
rev_expl = make_review_explainer(paths)

business_text = """
The food was amazing, service was fast and friendly.
However, the place was a bit noisy on Friday night.
Overall, great experience, would come again.
"""

biz_rating_exp = biz_expl.explain_rating(business_text, top_k=10)
print("Predicted rating:", biz_rating_exp["pred_rating"])

review_text = "Service was good and the food was tasty which I liked a lot. Definitely coming back."
rev_exp = rev_expl.explain_review(review_text, top_k=10)
print("Predicted stars:", rev_exp["pred_class"])
