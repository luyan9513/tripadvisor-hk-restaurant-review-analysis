from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "eda"
MPLCONFIGDIR = ROOT / "outputs" / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def write_markdown(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def markdown_table_from_df(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    descriptive = pd.read_csv(ROOT / "outputs" / "reviews_descriptive.csv", encoding="utf-8-sig")
    modeling = pd.read_csv(ROOT / "outputs" / "reviews_modeling.csv", encoding="utf-8-sig")
    restaurants = pd.read_csv(ROOT / "data" / "raw" / "hongkong_restaurants.csv", encoding="utf-8-sig")

    descriptive["published_at_date"] = pd.to_datetime(descriptive["published_at_date"], errors="coerce")
    descriptive["stay_date"] = pd.to_datetime(descriptive["stay_date"], errors="coerce")
    modeling["published_at_date"] = pd.to_datetime(modeling["published_at_date"], errors="coerce")

    descriptive["published_year"] = descriptive["published_at_date"].dt.year
    descriptive["stay_year"] = descriptive["stay_date"].dt.year
    descriptive["stay_month"] = descriptive["stay_date"].dt.month
    descriptive["trip_type_clean"] = descriptive["trip_type"].fillna("Unknown")

    modeling["trip_type_clean"] = modeling["trip_type"].fillna("Unknown")
    modeling["price_range_clean"] = modeling["price_range"].fillna("Unknown")

    review_count_per_restaurant = (
        descriptive.groupby("restaurant_entity_id")
        .size()
        .rename("review_count")
        .reset_index()
        .merge(
            restaurants[["tripadvisor_entity_id", "name", "price_range"]],
            left_on="restaurant_entity_id",
            right_on="tripadvisor_entity_id",
            how="left",
        )
    )

    rating_counts = (
        descriptive["rating"].value_counts().sort_index().rename_axis("rating").reset_index(name="count")
    )
    plt.figure(figsize=(7, 4))
    sns.barplot(data=rating_counts, x="rating", y="count", color="#2a6f97")
    plt.title("Raw Review Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    save_plot(OUTPUT_DIR / "raw_rating_distribution.png")

    binary_counts = (
        modeling["target_five_star_label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
    )
    plt.figure(figsize=(6, 4))
    sns.barplot(data=binary_counts, x="label", y="count", color="#355070")
    plt.title("Primary Binary Target Distribution")
    plt.xlabel("Target")
    plt.ylabel("Count")
    save_plot(OUTPUT_DIR / "binary_target_distribution.png")

    plt.figure(figsize=(8, 4.5))
    sns.histplot(review_count_per_restaurant["review_count"], bins=20, color="#6c757d")
    plt.axvline(120, color="#bc4749", linestyle="--", linewidth=2, label="Apparent crawl cap")
    plt.title("Review Counts per Restaurant")
    plt.xlabel("Review count in crawled dataset")
    plt.ylabel("Number of restaurants")
    plt.legend()
    save_plot(OUTPUT_DIR / "review_count_per_restaurant.png")

    review_year_counts = (
        descriptive["published_year"]
        .value_counts()
        .sort_index()
        .rename_axis("year")
        .reset_index(name="count")
    )
    plt.figure(figsize=(10, 4.5))
    sns.lineplot(data=review_year_counts, x="year", y="count", marker="o", color="#1d3557")
    plt.title("Review Volume by Published Year")
    plt.xlabel("Published year")
    plt.ylabel("Number of reviews")
    save_plot(OUTPUT_DIR / "review_volume_by_year.png")

    avg_rating_by_year = (
        descriptive.groupby("published_year")["rating"]
        .mean()
        .reset_index(name="avg_rating")
    )
    plt.figure(figsize=(10, 4.5))
    sns.lineplot(data=avg_rating_by_year, x="published_year", y="avg_rating", marker="o", color="#2a9d8f")
    plt.title("Average Rating by Published Year")
    plt.xlabel("Published year")
    plt.ylabel("Average rating")
    plt.ylim(3.2, 5.05)
    save_plot(OUTPUT_DIR / "average_rating_by_year.png")

    trip_counts = (
        modeling["trip_type_clean"]
        .value_counts()
        .rename_axis("trip_type")
        .reset_index(name="count")
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=trip_counts, y="trip_type", x="count", color="#588157")
    plt.title("Trip Type Distribution in Modeling Dataset")
    plt.xlabel("Count")
    plt.ylabel("Trip type")
    save_plot(OUTPUT_DIR / "trip_type_distribution.png")

    trip_rating = (
        modeling.groupby("trip_type_clean")
        .agg(avg_rating=("rating", "mean"), review_count=("review_id", "count"))
        .sort_values("avg_rating", ascending=False)
        .reset_index()
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=trip_rating, y="trip_type_clean", x="avg_rating", color="#8d99ae")
    plt.title("Average Rating by Trip Type")
    plt.xlabel("Average rating")
    plt.ylabel("Trip type")
    plt.xlim(3.0, 5.0)
    save_plot(OUTPUT_DIR / "average_rating_by_trip_type.png")

    price_rating = (
        modeling.groupby("price_range_clean")
        .agg(
            avg_rating=("rating", "mean"),
            review_count=("review_id", "count"),
            five_star_rate=("target_five_star", "mean"),
        )
        .reset_index()
    )
    price_order = ["$", "$$ - $$$", "$$$$", "Unknown"]
    price_rating["price_range_clean"] = pd.Categorical(
        price_rating["price_range_clean"], categories=price_order, ordered=True
    )
    price_rating = price_rating.sort_values("price_range_clean")
    price_rating["price_range_plot"] = price_rating["price_range_clean"].astype(str).replace(
        {"$": r"\$", "$$ - $$$": r"\$\$ - \$\$\$", "$$$$": r"\$\$\$\$"}
    )

    plt.figure(figsize=(7, 4.5))
    sns.barplot(data=price_rating, x="price_range_plot", y="avg_rating", color="#ffb703")
    plt.title("Average Rating by Price Range")
    plt.xlabel("Price range")
    plt.ylabel("Average rating")
    plt.ylim(3.5, 5.0)
    save_plot(OUTPUT_DIR / "average_rating_by_price_range.png")

    plt.figure(figsize=(7, 4.5))
    sns.barplot(data=price_rating, x="price_range_plot", y="five_star_rate", color="#fb8500")
    plt.title("Five-Star Rate by Price Range")
    plt.xlabel("Price range")
    plt.ylabel("Five-star rate")
    plt.ylim(0, 1)
    save_plot(OUTPUT_DIR / "five_star_rate_by_price_range.png")

    plt.figure(figsize=(8, 4.5))
    sns.boxplot(
        data=modeling,
        x="rating",
        y="text_len",
        color="#90be6d",
        showfliers=False,
    )
    plt.title("Review Text Length by Rating")
    plt.xlabel("Rating")
    plt.ylabel("Cleaned text length")
    save_plot(OUTPUT_DIR / "text_length_by_rating.png")

    recent = modeling[modeling["review_year"].isin([2023, 2024, 2025, 2026])].copy()
    recent_summary = (
        recent.groupby("review_year")
        .agg(
            review_count=("review_id", "count"),
            avg_rating=("rating", "mean"),
            five_star_rate=("target_five_star", "mean"),
            trip_type_missing_rate=("trip_type", lambda s: s.isna().mean()),
        )
        .reset_index()
    )

    summary = []
    summary.append("# EDA Summary")
    summary.append("")
    summary.append("## Key Findings")
    summary.append("- Review ratings are highly skewed toward 5 stars, so the primary prediction task should remain binary `five_star` vs `non_five_star`.")
    summary.append("- Review collection is visibly capped at 120 reviews for most restaurants, so restaurant-level conclusions should be interpreted as based on truncated samples.")
    summary.append("- `stay_date` behaves like a month-end placeholder rather than a true visit date and should only be used at month/year granularity.")
    summary.append("- `trip_type` coverage improves sharply in recent years, implying time-related missingness rather than purely random missing values.")
    summary.append("- Higher price ranges appear associated with higher five-star rates, which makes `price_range` a useful non-text feature.")
    summary.append("")
    summary.append("## Core Tables")
    summary.append("")
    summary.append("### Review Counts per Rating")
    summary.append(markdown_table_from_df(rating_counts))
    summary.append("")
    summary.append("### Review Count by Year")
    summary.append(markdown_table_from_df(review_year_counts))
    summary.append("")
    summary.append("### Trip Type Summary")
    summary.append(markdown_table_from_df(trip_rating.round(3)))
    summary.append("")
    summary.append("### Price Range Summary")
    summary.append(
        markdown_table_from_df(
            price_rating[["price_range_clean", "avg_rating", "review_count", "five_star_rate"]].round(3)
        )
    )
    summary.append("")
    summary.append("### Recent-Year Quality Checks")
    summary.append(markdown_table_from_df(recent_summary.round(4)))
    summary.append("")
    summary.append("## Generated Figures")
    for file_name in [
        "raw_rating_distribution.png",
        "binary_target_distribution.png",
        "review_count_per_restaurant.png",
        "review_volume_by_year.png",
        "average_rating_by_year.png",
        "trip_type_distribution.png",
        "average_rating_by_trip_type.png",
        "average_rating_by_price_range.png",
        "five_star_rate_by_price_range.png",
        "text_length_by_rating.png",
    ]:
        summary.append(f"- `{file_name}`")

    write_markdown(OUTPUT_DIR / "eda_summary.md", "\n".join(summary) + "\n")

    print(f"Wrote figures and summary to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
