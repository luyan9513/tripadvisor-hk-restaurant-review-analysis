from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REVIEWS_CSV = RAW_DATA_DIR / "hongkong_restaurant_reviews.csv"
RESTAURANTS_CSV = RAW_DATA_DIR / "hongkong_restaurants.csv"
OUTPUT_DIR = ROOT / "outputs"


def ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(ord(ch) < 128 for ch in text) / len(text)


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\uff06": "&",
        "\ufffd": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"[\U0001F300-\U0001FAFF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_title_and_text(title: str, text: str) -> str:
    title = title.strip()
    text = text.strip()
    if not title:
        return text
    if not text:
        return title
    if title.lower() == text.lower():
        return text
    if text.lower().startswith(title.lower()):
        return text
    return f"{title} {text}".strip()


def build_review_features(reviews: pd.DataFrame) -> pd.DataFrame:
    df = reviews.copy()

    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["title_clean"] = df["title"].map(normalize_text)
    df["text_clean"] = df["text"].map(normalize_text)
    df["combined_text"] = [
        combine_title_and_text(title, text)
        for title, text in zip(df["title_clean"], df["text_clean"])
    ]

    df["title_len"] = df["title_clean"].str.len()
    df["text_len"] = df["text_clean"].str.len()
    df["combined_len"] = df["combined_text"].str.len()

    df["title_has_chinese"] = df["title_clean"].map(contains_chinese)
    df["text_has_chinese"] = df["text_clean"].map(contains_chinese)
    df["has_chinese"] = df["title_has_chinese"] | df["text_has_chinese"]

    df["title_ascii_ratio"] = df["title_clean"].map(ascii_ratio)
    df["text_ascii_ratio"] = df["text_clean"].map(ascii_ratio)
    df["combined_ascii_ratio"] = df["combined_text"].map(ascii_ratio)

    df["is_original_english"] = df["original_language"].eq("en")
    df["is_language_en"] = df["language"].eq("en")
    df["has_replacement_char"] = (
        df["title"].str.contains("\ufffd", regex=False)
        | df["text"].str.contains("\ufffd", regex=False)
    )
    df["is_text_long_enough"] = df["text_len"] >= 30
    df["is_mostly_ascii"] = df["combined_ascii_ratio"] >= 0.95

    df["rating_3class"] = pd.Categorical(
        pd.cut(
            df["rating"],
            bins=[0, 2, 3, 5],
            labels=["negative", "neutral", "positive"],
            right=True,
            include_lowest=True,
        ),
        categories=["negative", "neutral", "positive"],
        ordered=True,
    )
    df["rating_3class_code"] = df["rating_3class"].cat.codes
    df["target_five_star"] = (df["rating"] == 5).astype(int)
    df["target_five_star_label"] = df["target_five_star"].map(
        {0: "non_five_star", 1: "five_star"}
    )

    df["published_at_date"] = pd.to_datetime(df["published_at_date"], errors="coerce")
    df["created_at_date"] = pd.to_datetime(df["created_at_date"], errors="coerce")
    df["stay_date"] = pd.to_datetime(df["stay_date"], errors="coerce")
    df["review_year"] = df["published_at_date"].dt.year
    df["review_month"] = df["published_at_date"].dt.month
    df["stay_year"] = df["stay_date"].dt.year
    df["stay_month"] = df["stay_date"].dt.month
    df["reviewer_contribution_count_log1p"] = (
        pd.to_numeric(df["reviewer_contribution_count"], errors="coerce").fillna(0).map(np.log1p)
    )
    df["like_count_log1p"] = (
        pd.to_numeric(df["like_count"], errors="coerce").fillna(0).map(np.log1p)
    )

    return df


def build_restaurant_features(restaurants: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "tripadvisor_entity_id",
        "name",
        "price_range",
        "address",
        "latitude",
        "longitude",
        "has_reservation",
        "featured_image",
        "phone",
    ]
    restaurant_df = restaurants[keep_cols].copy()
    restaurant_df = restaurant_df.rename(
        columns={
            "tripadvisor_entity_id": "restaurant_entity_id",
            "name": "restaurant_name_from_restaurants",
        }
    )
    return restaurant_df


def build_descriptive_dataset(
    review_df: pd.DataFrame, restaurant_df: pd.DataFrame
) -> pd.DataFrame:
    descriptive = review_df.merge(restaurant_df, on="restaurant_entity_id", how="left")
    descriptive["keep_for_descriptive"] = True
    return descriptive


def build_modeling_dataset(
    review_df: pd.DataFrame, restaurant_df: pd.DataFrame
) -> pd.DataFrame:
    modeling = review_df.merge(restaurant_df, on="restaurant_entity_id", how="left")
    modeling = modeling[
        modeling["is_original_english"]
        & ~modeling["has_chinese"]
        & modeling["is_text_long_enough"]
        & modeling["is_mostly_ascii"]
    ].copy()

    safe_columns = [
        "restaurant_entity_id",
        "restaurant_name",
        "review_id",
        "review_link",
        "title_clean",
        "text_clean",
        "combined_text",
        "rating",
        "rating_3class",
        "rating_3class_code",
        "target_five_star",
        "target_five_star_label",
        "trip_type",
        "like_count",
        "reviewer_contribution_count",
        "published_at_date",
        "created_at_date",
        "stay_date",
        "review_year",
        "review_month",
        "stay_year",
        "stay_month",
        "text_len",
        "combined_len",
        "combined_ascii_ratio",
        "like_count_log1p",
        "reviewer_contribution_count_log1p",
        "price_range",
        "latitude",
        "longitude",
        "has_reservation",
    ]
    return modeling[safe_columns].sort_values(["published_at_date", "review_id"]).reset_index(
        drop=True
    )


def markdown_table_from_series(series: pd.Series, header_left: str, header_right: str) -> str:
    lines = [f"| {header_left} | {header_right} |", "| --- | ---: |"]
    for idx, value in series.items():
        lines.append(f"| {idx} | {value} |")
    return "\n".join(lines)


def build_audit_summary(
    reviews_raw: pd.DataFrame,
    restaurants_raw: pd.DataFrame,
    reviews_featured: pd.DataFrame,
    modeling_df: pd.DataFrame,
) -> str:
    raw_rating_counts = reviews_raw["rating"].value_counts().sort_index()
    raw_3class_counts = reviews_featured["rating_3class"].value_counts().reindex(
        ["negative", "neutral", "positive"]
    )
    raw_five_star_counts = reviews_featured["target_five_star_label"].value_counts().reindex(
        ["non_five_star", "five_star"]
    )
    modeling_rating_counts = modeling_df["rating"].value_counts().sort_index()
    modeling_3class_counts = modeling_df["rating_3class"].value_counts().reindex(
        ["negative", "neutral", "positive"]
    )
    modeling_five_star_counts = modeling_df["target_five_star_label"].value_counts().reindex(
        ["non_five_star", "five_star"]
    )
    review_counts_per_restaurant = reviews_raw.groupby("restaurant_entity_id").size()

    lines = [
        "# Data Audit Summary",
        "",
        "## Overview",
        f"- Restaurants: {len(restaurants_raw):,}",
        f"- Reviews: {len(reviews_raw):,}",
        f"- Unique reviewers: {reviews_raw['reviewer_id'].nunique():,}",
        f"- Restaurants with exactly 120 reviews: {(review_counts_per_restaurant == 120).sum()} / {len(review_counts_per_restaurant)}",
        "",
        "## Review Data Quality",
        f"- Duplicate `review_id`: {reviews_raw['review_id'].duplicated().sum()}",
        f"- Duplicate `review_link`: {reviews_raw['review_link'].duplicated().sum()}",
        f"- Duplicate (`title`, `text`) pairs: {reviews_raw[['title', 'text']].duplicated().sum()}",
        f"- Missing `trip_type`: {reviews_raw['trip_type'].isna().sum():,}",
        f"- Missing `title`: {reviews_raw['title'].isna().sum():,}",
        f"- Missing `stay_date`: {reviews_raw['stay_date'].isna().sum():,}",
        "",
        "## Text Quality",
        f"- Reviews with non-ASCII characters: {(reviews_featured['combined_ascii_ratio'] < 1).sum():,}",
        f"- Reviews containing Chinese characters: {reviews_featured['has_chinese'].sum():,}",
        f"- Reviews marked `original_language = en`: {reviews_featured['is_original_english'].sum():,}",
        f"- Reviews kept for modeling: {len(modeling_df):,}",
        "",
        "## Recommended Modeling Rules",
        "- Keep only reviews where `original_language == 'en'`.",
        "- Exclude reviews whose title or text contains Chinese characters.",
        "- Exclude reviews with cleaned text length below 30 characters.",
        "- Exclude reviews whose combined cleaned text has ASCII ratio below 0.95.",
        "- Exclude restaurant aggregate fields such as overall restaurant rating and review count from predictive models.",
        "- Treat current-state fields such as `is_open_now` and `status_text` as descriptive only, not predictive features.",
        "",
        "## Labeling Strategy",
        "- Primary task: binary classification of `five_star` vs `non_five_star (1-4)`.",
        "- Secondary robustness task: 3-class sentiment-aligned labeling `negative (1-2)`, `neutral (3)`, `positive (4-5)`.",
        "",
        "## Raw Rating Distribution",
        markdown_table_from_series(raw_rating_counts, "Rating", "Count"),
        "",
        "## Raw 3-Class Distribution",
        markdown_table_from_series(raw_3class_counts, "Class", "Count"),
        "",
        "## Raw Binary Distribution",
        markdown_table_from_series(raw_five_star_counts, "Class", "Count"),
        "",
        "## Modeling Rating Distribution",
        markdown_table_from_series(modeling_rating_counts, "Rating", "Count"),
        "",
        "## Modeling 3-Class Distribution",
        markdown_table_from_series(modeling_3class_counts, "Class", "Count"),
        "",
        "## Modeling Binary Distribution",
        markdown_table_from_series(modeling_five_star_counts, "Class", "Count"),
        "",
        "## Limitations",
        "- A large share of restaurants hit an apparent review collection cap at 120 reviews, so the review set is likely truncated rather than complete.",
        "- The `language` field is unreliable; filtering should rely on `original_language` plus content inspection.",
        "- Some reviews appear bilingual or machine-translated, which can distort lexicon-based sentiment analysis if not filtered.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reviews_raw = pd.read_csv(REVIEWS_CSV, encoding="utf-8-sig")
    restaurants_raw = pd.read_csv(RESTAURANTS_CSV, encoding="utf-8-sig")

    reviews_featured = build_review_features(reviews_raw)
    restaurants_featured = build_restaurant_features(restaurants_raw)

    descriptive_df = build_descriptive_dataset(reviews_featured, restaurants_featured)
    modeling_df = build_modeling_dataset(reviews_featured, restaurants_featured)

    descriptive_df.to_csv(OUTPUT_DIR / "reviews_descriptive.csv", index=False, encoding="utf-8-sig")
    modeling_df.to_csv(OUTPUT_DIR / "reviews_modeling.csv", index=False, encoding="utf-8-sig")

    audit_summary = build_audit_summary(
        reviews_raw=reviews_raw,
        restaurants_raw=restaurants_raw,
        reviews_featured=reviews_featured,
        modeling_df=modeling_df,
    )
    (OUTPUT_DIR / "data_audit_summary.md").write_text(audit_summary, encoding="utf-8")

    print(f"Wrote {OUTPUT_DIR / 'reviews_descriptive.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'reviews_modeling.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'data_audit_summary.md'}")


if __name__ == "__main__":
    main()
