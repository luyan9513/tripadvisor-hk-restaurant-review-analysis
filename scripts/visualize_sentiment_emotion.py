from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "sentiment_emotion"
MPLCONFIGDIR = ROOT / "outputs" / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SENTIMENT_PALETTE = {
    "lexicon_sentiment_score": "#1d3557",
    "vader_compound": "#e76f51",
}

EMOTION_PALETTE = {
    "happy_ratio": "#2a9d8f",
    "anger_ratio": "#d62828",
    "sad_ratio": "#577590",
    "fear_ratio": "#6a4c93",
    "surprise_ratio": "#f4a261",
}

LABEL_PALETTE = {
    "positive": "#2a9d8f",
    "neutral": "#adb5bd",
    "negative": "#d62828",
}


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


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
    sns.set_theme(style="whitegrid", context="talk")

    df = pd.read_csv(ROOT / "outputs" / "reviews_modeling_features.csv", encoding="utf-8-sig")
    df["rating"] = df["rating"].astype(int)
    df["target_five_star_label"] = df["target_five_star_label"].fillna("Unknown")
    df["lexicon_sentiment_label"] = df["lexicon_sentiment_label"].fillna("neutral")
    df["dominant_emotion"] = df["dominant_emotion"].fillna("none")

    sentiment_by_rating = (
        df.groupby("rating")
        .agg(
            lexicon_sentiment_score=("lexicon_sentiment_score", "mean"),
            vader_compound=("vader_compound", "mean"),
        )
        .reset_index()
    )
    sentiment_long = sentiment_by_rating.melt(
        id_vars="rating",
        value_vars=["lexicon_sentiment_score", "vader_compound"],
        var_name="metric",
        value_name="score",
    )
    sentiment_long["metric_label"] = sentiment_long["metric"].replace(
        {
            "lexicon_sentiment_score": "Lexicon Score",
            "vader_compound": "VADER Compound",
        }
    )

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=sentiment_long,
        x="rating",
        y="score",
        hue="metric_label",
        style="metric_label",
        markers=True,
        dashes=False,
        palette=[SENTIMENT_PALETTE["lexicon_sentiment_score"], SENTIMENT_PALETTE["vader_compound"]],
        linewidth=2.5,
    )
    plt.title("Average Sentiment Score by Review Rating")
    plt.xlabel("Review rating")
    plt.ylabel("Average sentiment score")
    plt.ylim(-0.35, 1.0)
    plt.legend(title="")
    save_plot(OUTPUT_DIR / "sentiment_score_by_rating.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="rating",
        y="lexicon_sentiment_score",
        color="#7aa6c2",
        showfliers=False,
    )
    plt.axhline(0, color="#6c757d", linestyle="--", linewidth=1.4)
    plt.title("Lexicon Sentiment Distribution by Rating")
    plt.xlabel("Review rating")
    plt.ylabel("Lexicon sentiment score")
    save_plot(OUTPUT_DIR / "lexicon_sentiment_distribution_by_rating.png")

    label_mix = (
        pd.crosstab(df["rating"], df["lexicon_sentiment_label"], normalize="index")
        .reindex(columns=["negative", "neutral", "positive"], fill_value=0)
        .reset_index()
    )
    plt.figure(figsize=(8, 5))
    bottom = None
    for label in ["negative", "neutral", "positive"]:
        values = label_mix[label].values
        plt.bar(
            label_mix["rating"],
            values,
            bottom=bottom,
            color=LABEL_PALETTE[label],
            label=label.capitalize(),
            width=0.7,
        )
        bottom = values if bottom is None else bottom + values
    plt.title("Sentiment Label Mix by Rating")
    plt.xlabel("Review rating")
    plt.ylabel("Share of reviews")
    plt.ylim(0, 1)
    plt.legend(title="")
    save_plot(OUTPUT_DIR / "sentiment_label_mix_by_rating.png")

    emotion_by_rating = (
        df.groupby("rating")[["happy_ratio", "anger_ratio", "sad_ratio", "fear_ratio", "surprise_ratio"]]
        .mean()
        .reset_index()
    )
    emotion_long = emotion_by_rating.melt(
        id_vars="rating",
        value_vars=["happy_ratio", "anger_ratio", "sad_ratio", "fear_ratio", "surprise_ratio"],
        var_name="emotion",
        value_name="ratio",
    )
    emotion_long["emotion_label"] = emotion_long["emotion"].replace(
        {
            "happy_ratio": "Happy",
            "anger_ratio": "Anger",
            "sad_ratio": "Sad",
            "fear_ratio": "Fear",
            "surprise_ratio": "Surprise",
        }
    )

    plt.figure(figsize=(8.5, 5.2))
    sns.lineplot(
        data=emotion_long,
        x="rating",
        y="ratio",
        hue="emotion_label",
        style="emotion_label",
        markers=True,
        dashes=False,
        palette=[
            EMOTION_PALETTE["happy_ratio"],
            EMOTION_PALETTE["anger_ratio"],
            EMOTION_PALETTE["sad_ratio"],
            EMOTION_PALETTE["fear_ratio"],
            EMOTION_PALETTE["surprise_ratio"],
        ],
        linewidth=2.3,
    )
    plt.title("Average Emotion Ratios by Review Rating")
    plt.xlabel("Review rating")
    plt.ylabel("Average ratio")
    plt.legend(title="", ncol=3)
    save_plot(OUTPUT_DIR / "emotion_ratios_by_rating.png")

    dominant_emotion = (
        df["dominant_emotion"]
        .value_counts()
        .rename_axis("dominant_emotion")
        .reset_index(name="count")
    )
    dominant_emotion["emotion_label"] = dominant_emotion["dominant_emotion"].str.capitalize()
    plt.figure(figsize=(7.5, 4.8))
    sns.barplot(
        data=dominant_emotion,
        y="emotion_label",
        x="count",
        hue="emotion_label",
        palette=["#2a9d8f", "#d62828", "#6c757d", "#577590", "#6a4c93", "#f4a261"],
        dodge=False,
        legend=False,
    )
    plt.title("Dominant Emotion Distribution")
    plt.xlabel("Count")
    plt.ylabel("")
    save_plot(OUTPUT_DIR / "dominant_emotion_distribution.png")

    binary_emotion = (
        df.groupby("target_five_star_label")[["happy_ratio", "anger_ratio", "sad_ratio", "fear_ratio", "surprise_ratio"]]
        .mean()
        .reset_index()
    )
    binary_emotion_long = binary_emotion.melt(
        id_vars="target_five_star_label",
        value_vars=["happy_ratio", "anger_ratio", "sad_ratio", "fear_ratio", "surprise_ratio"],
        var_name="emotion",
        value_name="ratio",
    )
    binary_emotion_long["emotion_label"] = binary_emotion_long["emotion"].replace(
        {
            "happy_ratio": "Happy",
            "anger_ratio": "Anger",
            "sad_ratio": "Sad",
            "fear_ratio": "Fear",
            "surprise_ratio": "Surprise",
        }
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=binary_emotion_long,
        x="emotion_label",
        y="ratio",
        hue="target_five_star_label",
        palette=["#577590", "#2a9d8f"],
    )
    plt.title("Emotion Profile: Five-Star vs Non-Five-Star Reviews")
    plt.xlabel("")
    plt.ylabel("Average ratio")
    plt.legend(title="")
    save_plot(OUTPUT_DIR / "emotion_profile_by_binary_target.png")

    corr = df[
        [
            "rating",
            "target_five_star",
            "lexicon_sentiment_score",
            "vader_compound",
            "happy_ratio",
            "anger_ratio",
            "sad_ratio",
            "fear_ratio",
            "surprise_ratio",
        ]
    ].corr(numeric_only=True)
    corr_subset = corr.loc[
        ["rating", "target_five_star"],
        [
            "lexicon_sentiment_score",
            "vader_compound",
            "happy_ratio",
            "anger_ratio",
            "sad_ratio",
            "fear_ratio",
            "surprise_ratio",
        ],
    ]
    corr_subset.index = ["Rating", "Five-Star Target"]
    corr_subset.columns = [
        "Lexicon Score",
        "VADER",
        "Happy",
        "Anger",
        "Sad",
        "Fear",
        "Surprise",
    ]
    plt.figure(figsize=(9.2, 3.8))
    sns.heatmap(
        corr_subset,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation of Sentiment/Emotion Features with Targets")
    plt.xlabel("")
    plt.ylabel("")
    save_plot(OUTPUT_DIR / "sentiment_emotion_correlation_heatmap.png")

    summary_lines = [
        "# Sentiment and Emotion Visualization Summary",
        "",
        "## Recommended PPT Figures",
        "- `sentiment_score_by_rating.png`: strongest single-slide chart for showing that sentiment scores increase with rating.",
        "- `sentiment_label_mix_by_rating.png`: intuitive chart for showing that low-rated reviews contain far more negative sentiment.",
        "- `emotion_ratios_by_rating.png`: best chart for explaining how emotion signals shift from anger/sadness to happiness.",
        "- `sentiment_emotion_correlation_heatmap.png`: compact summary figure for the methodology/results section.",
        "",
        "## Key Talking Points",
        "- Both sentiment measures increase monotonically as review rating increases.",
        "- The positive sentiment share dominates 4-star and 5-star reviews, while negative sentiment is concentrated in 1-star and 2-star reviews.",
        "- `happy_ratio` rises with rating, while `anger_ratio`, `sad_ratio`, and `fear_ratio` decline.",
        "- Emotion features therefore add interpretable signals beyond raw text frequency features.",
        "",
        "## Core Tables",
        "### Average Sentiment Score by Rating",
        markdown_table_from_df(sentiment_by_rating.round(4)),
        "",
        "### Average Emotion Ratios by Rating",
        markdown_table_from_df(emotion_by_rating.round(4)),
        "",
        "### Dominant Emotion Distribution",
        markdown_table_from_df(dominant_emotion[["dominant_emotion", "count"]]),
        "",
        "## Generated Figures",
    ]

    for file_name in [
        "sentiment_score_by_rating.png",
        "lexicon_sentiment_distribution_by_rating.png",
        "sentiment_label_mix_by_rating.png",
        "emotion_ratios_by_rating.png",
        "dominant_emotion_distribution.png",
        "emotion_profile_by_binary_target.png",
        "sentiment_emotion_correlation_heatmap.png",
    ]:
        summary_lines.append(f"- `{file_name}`")

    (OUTPUT_DIR / "visualization_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_DIR / 'visualization_summary.md'}")


if __name__ == "__main__":
    main()
