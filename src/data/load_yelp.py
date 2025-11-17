import json
from pathlib import Path

import pandas as pd


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
      4. Filter reviews by minimal text length.
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
