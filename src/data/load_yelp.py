import json
import re
from pathlib import Path

import pandas as pd

# Optional fastText language ID (use fasttext-wheel==0.9.2 and lid.176.bin)
try:
    import fasttext  # noqa: F401
except ImportError:  # fasttext is optional; only needed if use_fasttext is enabled in params
    fasttext = None

_ft_model = None


def _read_jsonl(path, usecols=None, chunksize=100_000):
    """
    Streaming reader for large JSONL files.

    Parameters
    ----------
    path : Path or str
        Path to the Yelp .json file (one JSON object per line).
    usecols : list or None
        If not None, keep only these keys from each JSON object.
    chunksize : int
        Number of lines to accumulate before converting to a DataFrame.

    Yields
    ------
    pd.DataFrame
        Dataframe with up to `chunksize` rows.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        batch = []
        for line in f:
            obj = json.loads(line)
            if usecols is not None:
                obj = {k: obj.get(k) for k in usecols}
            batch.append(obj)
            if len(batch) >= chunksize:
                yield pd.DataFrame(batch)
                batch = []
        if batch:
            yield pd.DataFrame(batch)


def _load_ft_model(path: str):
    """
    Lazy-load the fastText language ID model (lid.176.bin).
    """
    global _ft_model
    if _ft_model is None:
        if fasttext is None:
            raise ImportError(
                "fasttext is not installed. Install fasttext-wheel==0.9.2 "
                "and provide lid.176.bin, or set use_fasttext: false."
            )
        _ft_model = fasttext.load_model(path)
    return _ft_model


def _is_probably_english(text: str, ascii_ratio_threshold: float, min_alpha_chars: int) -> bool:
    """
    Cheap, dependency-free heuristic for English-like text:
    - Require a minimum number of ASCII alphabetic characters.
    - Require that the ratio of (ASCII letters + spaces) to total chars meets threshold.
    """
    if not isinstance(text, str) or len(text) == 0:
        return False
    total = len(text)
    alpha_spaces = sum(1 for ch in text if (ch.isalpha() and ch.isascii()) or ch == " ")
    alpha_only = sum(1 for ch in text if ch.isalpha() and ch.isascii())
    if alpha_only < min_alpha_chars:
        return False
    ratio = alpha_spaces / total
    return ratio >= ascii_ratio_threshold


def _is_english_fasttext(text: str, model, thr: float) -> bool:
    """
    fastText language ID gate: keep if predicted English with probability >= thr.
    """
    if not isinstance(text, str) or len(text) == 0:
        return False
    labels, probs = model.predict(text.replace("\n", " "), k=1)
    return (labels[0] == "__label__en") and (probs[0] >= thr)


def prepare_joined_reviews(paths: dict, params: dict) -> None:
    """
    Create a joined table of reviews and business metadata.

    Output:
      - data_proc/reviews_joined.parquet
      - data_proc/business.parquet

    The joined table has:
      - Per-review info (text, stars, date, etc.)
      - The corresponding business's attributes (stars_biz, categories, city, state, etc.)

    Steps:
      1. Load business dataset (subset of columns).
      2. Optionally filter by state (to reduce data volume).
      3. Load review dataset but keep only reviews for selected businesses.
      4. Filter reviews by minimal text length and language (if enabled).
      5. Merge and sort by date.
    """
    raw_dir = Path(paths["raw_dir"])
    proc_dir = Path(paths["proc_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 1. Load businesses
    # ----------------------------
    business_cols = [
        "business_id",
        "name",
        "city",
        "state",
        "categories",
        "attributes",
        "hours",
        "stars",
        "review_count",
    ]
    business_path = raw_dir / "yelp_academic_dataset_business.json"

    bus_parts = list(_read_jsonl(business_path, usecols=business_cols))
    if not bus_parts:
        raise RuntimeError(f"No business data loaded from {business_path}")
    bus = pd.concat(bus_parts, ignore_index=True)

    # Optionally filter by state to reduce dataset size
    states = params.get("states", []) or []
    if len(states) > 0:
        bus = bus[bus["state"].isin(states)].copy()

    # ----------------------------
    # 2. Load reviews for those businesses
    # ----------------------------
    review_cols = [
        "review_id",
        "user_id",
        "business_id",
        "stars",
        "text",
        "date",
        "useful",
        "funny",
        "cool",
    ]
    review_path = raw_dir / "yelp_academic_dataset_review.json"

    biz_ids = set(bus["business_id"])
    rev_parts = []
    for df_chunk in _read_jsonl(review_path, usecols=review_cols):
        df_chunk = df_chunk[df_chunk["business_id"].isin(biz_ids)]
        rev_parts.append(df_chunk)

    if not rev_parts:
        raise RuntimeError("No reviews overlapping with selected businesses were found.")
    rev = pd.concat(rev_parts, ignore_index=True)

    # ----------------------------
    # 3. Basic review cleaning
    # ----------------------------
    rev.dropna(subset=["text", "date"], inplace=True)
    rev["text_len"] = rev["text"].str.len()
    min_len = int(params.get("min_review_len", 30))
    rev = rev[rev["text_len"] >= min_len].copy()
    rev["date"] = pd.to_datetime(rev["date"])

    # Optional language filter (ASCII prefilter + fastText)
    lang_cfg = params.get("language", {}) or {}
    if lang_cfg.get("enabled", False):
        thr_ascii = float(lang_cfg.get("ascii_ratio_threshold", 0.6))
        min_alpha = int(lang_cfg.get("min_alpha_chars", 20))
        use_ft = lang_cfg.get("use_fasttext", False)
        ft_thr = float(lang_cfg.get("fasttext_threshold", 0.8))
        ft_path = lang_cfg.get("fasttext_model_path", "lid.176.bin")

        if use_ft:
            ft_model = _load_ft_model(ft_path)
            rev["keep_lang"] = rev["text"].apply(
                lambda s: _is_probably_english(s, thr_ascii, min_alpha)
                and _is_english_fasttext(s, ft_model, ft_thr)
            )
        else:
            rev["keep_lang"] = rev["text"].apply(
                lambda s: _is_probably_english(s, thr_ascii, min_alpha)
            )
        rev = rev[rev["keep_lang"]].drop(columns=["keep_lang"])

    # ----------------------------
    # 4. Merge reviews with business info
    # ----------------------------
    joined = rev.merge(
        bus,
        on="business_id",
        how="left",
        suffixes=("", "_biz"),
    ).sort_values("date")

    # Rename business-level stars for clarity
    joined.rename(columns={"stars_biz": "stars_biz_tmp"}, inplace=True)
    # In some datasets, the merge may produce both stars and stars_biz;
    # we standardize to "stars_biz" as the business's average rating.
    if "stars_biz_tmp" in joined.columns:
        joined["stars_biz"] = joined["stars_biz_tmp"]
        joined.drop(columns=["stars_biz_tmp"], inplace=True)
    else:
        # If no "stars_biz" came from the business file, fallback to "stars"
        joined["stars_biz"] = joined["stars"]

    # Save outputs
    (proc_dir / "business.parquet").parent.mkdir(parents=True, exist_ok=True)
    bus.to_parquet(proc_dir / "business.parquet", index=False)
    joined.to_parquet(proc_dir / "reviews_joined.parquet", index=False)

    print(f"[prepare_joined_reviews] Saved {proc_dir/'business.parquet'}  shape={bus.shape}")
    print(f"[prepare_joined_reviews] Saved {proc_dir/'reviews_joined.parquet'}  shape={joined.shape}")