# Hong Kong TripAdvisor Restaurant Reviews

Python utilities for crawling, processing, exploring, and modeling TripAdvisor reviews of Hong Kong restaurants. This repo bundles the original CSV exports, data pipelines, and supporting visualizations created as part of the IS6941 group project.

## Repository structure
- `data/raw/`: original `hongkong_restaurant_reviews.csv` and `hongkong_restaurants.csv` collections (dataset already included).
- `scripts/`: Python scripts for crawling TripAdvisor, cleaning & featurizing reviews, running EDA, extracting sentiment/emotion features, training baseline models, and visualizing emotion summaries.
- `outputs/`: generated CSVs, plots, and summaries from the pipeline (ignored by git; regenerate with the scripts).
- `resources/`: supporting sentiment/emotion lexicons used when enriching the review corpus.
- `tripadvisor_hk_restaurants_reviews_crawl.py`: standalone crawler client that can refresh the raw CSVs if you have a TripAdvisor scraper API key.
- `report.pdf` / `report.docx` / `IS6941-GroupProject-2026.docx` / `Quartet-V2.pptx`: deliverable documents & slides.

## Getting started
1. **Python environment**: Python 3.11+ is recommended. Install dependencies with `pip install -r requirements.txt`.
2. **NLTK data** (needed by the sentiment/emotion script):
   ```bash
   python -m nltk.downloader opinion_lexicon vader_lexicon
   ```
3. **Crawler (optional)**: set `API_KEY` (and optionally `BASE_URL`) environment variables, then run:
   ```bash
   python tripadvisor_hk_restaurants_reviews_crawl.py
   ```
   This rewrites `data/raw/hongkong_restaurants.csv` and `data/raw/hongkong_restaurant_reviews.csv`.
4. **Prepare modeling data**:
   ```bash
   python scripts/prepare_modeling_data.py
   ```
   This generates `outputs/reviews_descriptive.csv`, `outputs/reviews_modeling.csv`, and `outputs/data_audit_summary.md`.
5. **Run EDA**:
   ```bash
   python scripts/run_eda.py
   ```
   Output charts and `outputs/eda/eda_summary.md` describe the dataset.
6. **Sentiment & emotion features**:
   ```bash
   python scripts/extract_sentiment_emotion_features.py
   ```
   This creates `outputs/reviews_modeling_features.csv` and `outputs/sentiment_emotion_summary.md`.
7. **Train baseline models**:
   ```bash
   python scripts/train_baseline_models.py
   ```
   Results + plots land in `outputs/modeling/`.
8. **Visualize emotion profiles**:
   ```bash
   python scripts/visualize_sentiment_emotion.py
   ```
   See `outputs/sentiment_emotion/visualization_summary.md` for guidance on each figure.

## Scripts at a glance
| Script | Purpose |
| --- | --- |
| `tripadvisor_hk_restaurants_reviews_crawl.py` | Wraps the TripAdvisor Scraper API with retry logic and writes the raw restaurant/review CSVs. |
| `scripts/prepare_modeling_data.py` | Normalizes text, engineers features, merges restaurant metadata, and writes descriptive/modeling sheets plus a data audit. |
| `scripts/run_eda.py` | Builds summary stats & plots (rating distribution, trip types, price range trends, etc.). |
| `scripts/extract_sentiment_emotion_features.py` | Adds lexicon sentiment, VADER, and emotion-ratio features along with a markdown summary. |
| `scripts/train_baseline_models.py` | Benchmarks metadata-only, sentiment-only, TF-IDF-only, and combined classifiers (KNN, RF, HGB, XGBoost). |
| `scripts/visualize_sentiment_emotion.py` | Produces publication-ready charts comparing sentiment and emotion ratios by rating/five-star label. |

## Notes
- The pipeline assumes the raw CSVs live in `data/raw/`. Keep these files if you want reproducibility.
- `outputs/` contains derived data (ignored by git). Remove or re-run the scripts to refresh visuals/reports.
- Sentiment/emotion scripts depend on the NRC lexicon and local CSV lexicons under `resources/lexicons/`.
- The `report.*` files summarize findings and can be adapted for presentations.
