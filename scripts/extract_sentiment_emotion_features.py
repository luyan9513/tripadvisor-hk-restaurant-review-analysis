from pathlib import Path
import csv
import re

import numpy as np
import pandas as pd
from nltk.corpus import opinion_lexicon
from nltk.sentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
RESOURCE_DIR = ROOT / "resources" / "lexicons"
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
NEGATION_WINDOW = 3
SIA = SentimentIntensityAnalyzer()
FALLBACK_NEGATIONS = {
    "not", "don't", "dont", "doesn't", "doesnt", "didn't", "didnt", "no",
    "never", "without", "cannot", "can't", "cant", "hardly", "rarely",
    "barely", "nothing", "none", "nor", "neither",
}

SENTIMENT_LEXICON_PATH = RESOURCE_DIR / "sentiment" / "senti_lexicon.tff"
NEGATION_WORDS_PATH = RESOURCE_DIR / "sentiment" / "Negation.csv"
EMOTION_LEXICON_PATH = RESOURCE_DIR / "emotion" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

NRC_TO_PROJECT_EMOTION = {
    "joy": "happy",
    "anger": "anger",
    "sadness": "sad",
    "fear": "fear",
    "surprise": "surprise",
}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def load_negation_words() -> set[str]:
    words = set()
    try:
        with NEGATION_WORDS_PATH.open("r", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if row and row[0].strip():
                    words.add(row[0].strip().lower())
    except OSError:
        words = set(FALLBACK_NEGATIONS)
    return words


def load_sentiment_lexicon() -> tuple[set[str], set[str]]:
    pos_words: set[str] = set()
    neg_words: set[str] = set()
    try:
        with SENTIMENT_LEXICON_PATH.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                attrs = {}
                for item in line.split():
                    if "=" in item:
                        key, value = item.split("=", 1)
                        attrs[key] = value
                word = attrs.get("word1", "").lower()
                polarity = attrs.get("priorpolarity", "").lower()
                if not word:
                    continue
                if polarity in {"positive", "both"}:
                    pos_words.add(word)
                if polarity in {"negative", "both"}:
                    neg_words.add(word)
    except OSError:
        pos_words = set(opinion_lexicon.positive())
        neg_words = set(opinion_lexicon.negative())
    return pos_words, neg_words


def load_nrc_lexicon() -> dict[str, set[str]]:
    lexicon: dict[str, set[str]] = {}
    try:
        with EMOTION_LEXICON_PATH.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                word, emotion, flag = parts
                if flag != "1":
                    continue
                lexicon.setdefault(word.lower(), set()).add(emotion.lower())
    except OSError:
        raw = NRCLex().__lexicon__
        for word, emotions in raw.items():
            lexicon[word.lower()] = set(emotions)
    return lexicon


NEGATION_WORDS = load_negation_words()
POSITIVE_WORDS, NEGATIVE_WORDS = load_sentiment_lexicon()
NRC_LEXICON = load_nrc_lexicon()


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def compute_lexicon_sentiment(tokens: list[str]) -> dict[str, float | int | str]:
    pos_count = 0
    neg_count = 0
    negated_count = 0

    for idx, token in enumerate(tokens):
        is_pos = token in POSITIVE_WORDS
        is_neg = token in NEGATIVE_WORDS
        if not is_pos and not is_neg:
            continue

        start = max(0, idx - NEGATION_WINDOW)
        reverse = any(prev_token in NEGATION_WORDS for prev_token in tokens[start:idx])
        if reverse:
            negated_count += 1
            is_pos, is_neg = is_neg, is_pos

        if is_pos:
            pos_count += 1
        if is_neg:
            neg_count += 1

    senti_total = pos_count + neg_count
    senti_score = safe_div(pos_count - neg_count, senti_total)

    senti_label = "neutral"
    if senti_score > 0:
        senti_label = "positive"
    elif senti_score < 0:
        senti_label = "negative"

    return {
        "lexicon_positive_count": pos_count,
        "lexicon_negative_count": neg_count,
        "lexicon_negated_hits": negated_count,
        "lexicon_sentiment_total": senti_total,
        "lexicon_sentiment_score": round(senti_score, 6),
        "lexicon_sentiment_label": senti_label,
    }


def compute_emotion_features(tokens: list[str]) -> dict[str, float | int | str]:
    nrc_counts = {
        "anger": 0,
        "anticipation": 0,
        "disgust": 0,
        "fear": 0,
        "joy": 0,
        "sadness": 0,
        "surprise": 0,
        "trust": 0,
        "positive": 0,
        "negative": 0,
    }

    for token in tokens:
        for emotion in NRC_LEXICON.get(token, set()):
            if emotion in nrc_counts:
                nrc_counts[emotion] += 1

    token_count = max(len(tokens), 1)
    project_counts = {
        "happy": nrc_counts["joy"],
        "anger": nrc_counts["anger"],
        "sad": nrc_counts["sadness"],
        "fear": nrc_counts["fear"],
        "surprise": nrc_counts["surprise"],
    }

    dominant_emotion = "none"
    if any(project_counts.values()):
        dominant_emotion = max(project_counts, key=project_counts.get)

    result: dict[str, float | int | str] = {
        "dominant_emotion": dominant_emotion,
        "nrc_positive_count": nrc_counts["positive"],
        "nrc_negative_count": nrc_counts["negative"],
        "nrc_trust_count": nrc_counts["trust"],
        "nrc_anticipation_count": nrc_counts["anticipation"],
        "nrc_disgust_count": nrc_counts["disgust"],
    }

    for emotion, count in project_counts.items():
        result[f"{emotion}_count"] = count
        result[f"{emotion}_ratio"] = round(safe_div(count, token_count), 6)

    return result


def compute_features(text: str) -> dict[str, float | int | str]:
    tokens = tokenize(text)
    vader = SIA.polarity_scores(text)

    result: dict[str, float | int | str] = {
        "token_count": len(tokens),
        "unique_token_count": len(set(tokens)),
        "vader_compound": round(vader["compound"], 6),
        "vader_neg": round(vader["neg"], 6),
        "vader_neu": round(vader["neu"], 6),
        "vader_pos": round(vader["pos"], 6),
    }
    result.update(compute_lexicon_sentiment(tokens))
    result.update(compute_emotion_features(tokens))
    return result


def table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def build_summary(enriched: pd.DataFrame) -> str:
    sentiment_by_rating = (
        enriched.groupby("rating")
        .agg(
            review_count=("review_id", "count"),
            avg_lexicon_score=("lexicon_sentiment_score", "mean"),
            avg_vader_compound=("vader_compound", "mean"),
            avg_positive_count=("lexicon_positive_count", "mean"),
            avg_negative_count=("lexicon_negative_count", "mean"),
        )
        .round(4)
        .reset_index()
    )

    emotion_by_rating = (
        enriched.groupby("rating")[["happy_ratio", "anger_ratio", "sad_ratio", "fear_ratio", "surprise_ratio"]]
        .mean()
        .round(4)
        .reset_index()
    )

    lexicon_label_mix = (
        pd.crosstab(enriched["rating"], enriched["lexicon_sentiment_label"], normalize="index")
        .round(4)
        .reset_index()
    )

    dominant_emotion_counts = (
        enriched["dominant_emotion"].value_counts()
        .rename_axis("dominant_emotion")
        .reset_index(name="count")
    )

    corr = enriched[
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
    ].round(4)

    summary = [
        "# Sentiment and Emotion Summary",
        "",
        "## Feature Notes",
        "- Primary sentiment feature is a rule-based lexicon score: `(positive - negative) / (positive + negative)`.",
        f"- Negation handling uses a fixed backward window of {NEGATION_WINDOW} tokens.",
        "- Emotion features use the NRC word-emotion lexicon and report a 5-emotion view: `happy`, `anger`, `sad`, `fear`, `surprise`.",
        "- `vader_compound` is included as an additional benchmark feature.",
        "",
        "## Dataset Coverage",
        f"- Modeling rows processed: {len(enriched):,}",
        f"- Mean lexicon sentiment score: {enriched['lexicon_sentiment_score'].mean():.4f}",
        f"- Share of lexicon-positive reviews: {(enriched['lexicon_sentiment_label'].eq('positive').mean()):.4f}",
        f"- Share of lexicon-negative reviews: {(enriched['lexicon_sentiment_label'].eq('negative').mean()):.4f}",
        "",
        "## Sentiment by Rating",
        table(sentiment_by_rating),
        "",
        "## Emotion Ratios by Rating",
        table(emotion_by_rating),
        "",
        "## Lexicon Sentiment Label Mix by Rating",
        table(lexicon_label_mix),
        "",
        "## Dominant Emotion Distribution",
        table(dominant_emotion_counts),
        "",
        "## Correlation with Targets",
        table(corr_subset.reset_index().rename(columns={"index": "target"})),
    ]
    return "\n".join(summary) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    modeling = pd.read_csv(OUTPUT_DIR / "reviews_modeling.csv", encoding="utf-8-sig")

    feature_rows = [compute_features(text) for text in modeling["combined_text"].astype(str)]
    features = pd.DataFrame(feature_rows)
    enriched = pd.concat([modeling, features], axis=1)

    for col in ["lexicon_sentiment_score", "vader_compound"]:
        enriched[f"{col}_z"] = (
            (enriched[col] - enriched[col].mean()) / enriched[col].std(ddof=0)
        ).replace([np.inf, -np.inf], 0).fillna(0)

    enriched.to_csv(OUTPUT_DIR / "reviews_modeling_features.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "sentiment_emotion_summary.md").write_text(
        build_summary(enriched), encoding="utf-8"
    )

    print(f"Wrote {OUTPUT_DIR / 'reviews_modeling_features.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'sentiment_emotion_summary.md'}")


if __name__ == "__main__":
    main()
