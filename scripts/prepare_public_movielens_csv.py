"""
Prepare a public MovieLens CSV dataset for recsys-universe industrial demo.

Why this exists:
- public MovieLens CSV variants such as `ml-latest-small` usually contain
  `movies.csv` and `ratings.csv`, but do not provide a `users.csv` with
  demographics.
- this repo's CSV ingestion path works best when a minimal `users.csv` exists.

This script generates a lightweight compatibility layer:
1. copies / validates `movies.csv`
2. copies / validates `ratings.csv`
3. generates a default `users.csv` from unique user IDs in ratings

Usage:
    python scripts/prepare_public_movielens_csv.py \
      --src /path/to/ml-latest-small \
      --dst /path/to/output_dir
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def prepare_dataset(src: Path, dst: Path) -> None:
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    movies_file = src / "movies.csv"
    ratings_file = src / "ratings.csv"

    if not movies_file.exists():
        raise FileNotFoundError(f"movies.csv not found: {movies_file}")
    if not ratings_file.exists():
        raise FileNotFoundError(f"ratings.csv not found: {ratings_file}")

    movies = pd.read_csv(movies_file)
    ratings = pd.read_csv(ratings_file)

    required_movie_cols = {"movieId", "title", "genres"}
    required_rating_cols = {"userId", "movieId", "rating", "timestamp"}

    if not required_movie_cols.issubset(set(movies.columns)):
        raise ValueError(
            f"movies.csv must contain columns {sorted(required_movie_cols)}, got {list(movies.columns)}"
        )
    if not required_rating_cols.issubset(set(ratings.columns)):
        raise ValueError(
            f"ratings.csv must contain columns {sorted(required_rating_cols)}, got {list(ratings.columns)}"
        )

    # Keep original public files as-is.
    movies.to_csv(dst / "movies.csv", index=False)
    ratings.to_csv(dst / "ratings.csv", index=False)

    # Generate a minimal compatibility users.csv.
    user_ids = sorted(ratings["userId"].dropna().astype(int).unique().tolist())
    users = pd.DataFrame(
        {
            "user_id": user_ids,
            "gender": ["M" for _ in user_ids],
            "age": [25 for _ in user_ids],
            "occupation": ["other" for _ in user_ids],
            "zip_code": ["00000" for _ in user_ids],
        }
    )
    users.to_csv(dst / "users.csv", index=False)

    print("Prepared dataset:")
    print(f"  src={src}")
    print(f"  dst={dst}")
    print(f"  users={len(users)}")
    print(f"  items={len(movies)}")
    print(f"  ratings={len(ratings)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source directory containing movies.csv and ratings.csv")
    parser.add_argument("--dst", required=True, help="Output directory for repo-compatible CSV files")
    args = parser.parse_args()

    prepare_dataset(Path(args.src), Path(args.dst))


if __name__ == "__main__":
    main()
