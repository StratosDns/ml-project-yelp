import argparse
import yaml

from src.data.load_yelp import prepare_joined_reviews
from src.features.features import build_business_features, build_review_features
from src.models.business_models import train_business_models
from src.models.review_model import train_review_model
from src.evals.metrics import evaluate_business_models


def load_cfg():
    """
    Helper to load paths + params from YAML configs.

    Returns
    -------
    paths : dict
        Contains directories like raw_dir, proc_dir, models_dir, figs_dir.
    params : dict
        Contains modeling and preprocessing hyperparameters.
    """
    with open("configs/paths.yml", "r") as f:
        paths = yaml.safe_load(f)
    with open("configs/params.yml", "r") as f:
        params = yaml.safe_load(f)
    return paths, params


def main():
    parser = argparse.ArgumentParser(description="Yelp Rating & Sentiment Pipeline")
    subparsers = parser.add_subparsers(dest="cmd")

    subparsers.add_parser("prepare-data", help="Prepare joined reviews+businesses parquet")
    subparsers.add_parser("build-features", help="Build business-level and review-level features")
    subparsers.add_parser("train-business", help="Train business-level models (regression + binary)")
    subparsers.add_parser("train-review", help="Train review-level 5-class rating model")
    subparsers.add_parser("evaluate", help="Evaluate business models (regression + classification)")

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        return

    paths, params = load_cfg()

    if args.cmd == "prepare-data":
        prepare_joined_reviews(paths, params)
    elif args.cmd == "build-features":
        build_business_features(paths, params)
        build_review_features(paths, params)
    elif args.cmd == "train-business":
        train_business_models(paths, params)
    elif args.cmd == "train-review":
        train_review_model(paths, params)
    elif args.cmd == "evaluate":
        evaluate_business_models(paths, params)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
