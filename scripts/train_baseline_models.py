from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

MPLCONFIGDIR = Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "modeling"
RANDOM_STATE = 42
CV_FOLDS = 5

PRIMARY_TARGET = "target_five_star"
TEXT_COL = "combined_text"

METADATA_NUMERIC = [
    "reviewer_contribution_count_log1p",
    "like_count_log1p",
    "text_len",
    "combined_len",
    "latitude",
    "longitude",
    "review_year",
    "review_month",
]

METADATA_CATEGORICAL = [
    "trip_type",
    "price_range",
    "has_reservation",
]

SENTIMENT_EMOTION_NUMERIC = [
    "lexicon_sentiment_score",
    "lexicon_positive_count",
    "lexicon_negative_count",
    "lexicon_negated_hits",
    "lexicon_sentiment_total",
    "vader_compound",
    "vader_neg",
    "vader_neu",
    "vader_pos",
    "happy_ratio",
    "anger_ratio",
    "sad_ratio",
    "fear_ratio",
    "surprise_ratio",
    "nrc_positive_count",
    "nrc_negative_count",
    "nrc_trust_count",
    "nrc_anticipation_count",
    "nrc_disgust_count",
]

SENTIMENT_EMOTION_CATEGORICAL = [
    "lexicon_sentiment_label",
    "dominant_emotion",
]


def metric_dict(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | list[list[int]]]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def probability_metrics(y_true: pd.Series, y_score: np.ndarray | None) -> dict[str, float | None]:
    if y_score is None:
        return {"roc_auc": None, "average_precision": None}
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "average_precision": round(float(average_precision_score(y_true, y_score)), 4),
    }


def make_tabular_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def make_tfidf_pipeline_knn() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=5)),
            ("knn", KNeighborsClassifier(n_neighbors=15, metric="cosine", algorithm="brute")),
        ]
    )


def make_tabular_knn(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", make_tabular_preprocessor(numeric_cols, categorical_cols)),
            ("knn", KNeighborsClassifier(n_neighbors=15, metric="cosine", algorithm="brute")),
        ]
    )


def make_tabular_rf(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", make_tabular_preprocessor(numeric_cols, categorical_cols)),
            ("rf", RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
            )),
        ]
    )


def make_combined_preprocessor() -> ColumnTransformer:
    text_pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=5)),
            ("svd", TruncatedSVD(n_components=100, random_state=RANDOM_STATE)),
            ("scale", StandardScaler()),
        ]
    )
    tabular_numeric = METADATA_NUMERIC + SENTIMENT_EMOTION_NUMERIC
    tabular_categorical = METADATA_CATEGORICAL + SENTIMENT_EMOTION_CATEGORICAL
    tabular_pipeline = make_tabular_preprocessor(tabular_numeric, tabular_categorical)
    return ColumnTransformer(
        transformers=[
            ("text", text_pipeline, TEXT_COL),
            ("tab", tabular_pipeline, tabular_numeric + tabular_categorical),
        ]
    )


def make_combined_knn() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", make_combined_preprocessor()),
            ("knn", KNeighborsClassifier(n_neighbors=15, metric="cosine", algorithm="brute")),
        ]
    )


def make_combined_rf() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", make_combined_preprocessor()),
            ("rf", RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
            )),
        ]
    )


def make_tabular_hgb(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", make_tabular_preprocessor(numeric_cols, categorical_cols)),
            ("hgb", HistGradientBoostingClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                early_stopping=True,
            )),
        ]
    )


def make_combined_hgb() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", make_combined_preprocessor()),
            ("hgb", HistGradientBoostingClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                early_stopping=True,
            )),
        ]
    )


def make_combined_xgb(scale_pos_weight: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", make_combined_preprocessor()),
            ("xgb", XGBClassifier(
                random_state=RANDOM_STATE,
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                tree_method="hist",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                n_jobs=1,
            )),
        ]
    )


def run_search(name: str, pipeline: Pipeline, param_grid: dict, X_train: pd.DataFrame, y_train: pd.Series):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    if name in {"combined_rf", "combined_xgb"}:
        n_iter = 10 if name == "combined_rf" else 4
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="f1_macro",
            n_jobs=1,
            cv=cv,
            refit=True,
            random_state=RANDOM_STATE,
            verbose=0,
        )
    else:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1_macro",
            n_jobs=1,
            cv=cv,
            refit=True,
            verbose=0,
        )
    search.fit(X_train, y_train)
    return search


def run_experiment(name: str, search, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_score = None
    if hasattr(best_model, "predict_proba"):
        y_score = best_model.predict_proba(X_test)[:, 1]
    result = {
        "experiment": name,
        "cv_best_score": round(float(search.best_score_), 4),
        "best_params": search.best_params_,
        **metric_dict(y_test, y_pred),
        **probability_metrics(y_test, y_score),
        "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
    }
    return result, best_model


def to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_results(results_df: pd.DataFrame, best_model, X_test, y_test: pd.Series) -> None:
    sns.set_theme(style="whitegrid")

    plot_df = results_df.copy()
    order = plot_df["experiment"].tolist()[::-1]

    plt.figure(figsize=(9, 5))
    sns.barplot(data=plot_df, y="experiment", x="f1_macro", order=order, color="#355070")
    plt.title("Model Comparison by Macro-F1")
    plt.xlabel("Macro-F1")
    plt.ylabel("")
    save_plot(OUTPUT_DIR / "macro_f1_comparison.png")

    plt.figure(figsize=(9, 5))
    sns.barplot(data=plot_df, y="experiment", x="accuracy", order=order, color="#588157")
    plt.title("Model Comparison by Accuracy")
    plt.xlabel("Accuracy")
    plt.ylabel("")
    save_plot(OUTPUT_DIR / "accuracy_comparison.png")

    plt.figure(figsize=(9, 5))
    sns.barplot(data=plot_df, y="experiment", x="cv_best_score", order=order, color="#bc6c25")
    plt.title("Cross-Validated Macro-F1 on Training Data")
    plt.xlabel("CV Macro-F1")
    plt.ylabel("")
    save_plot(OUTPUT_DIR / "cv_macro_f1_comparison.png")

    cm = confusion_matrix(y_test, best_model.predict(X_test))
    plt.figure(figsize=(5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["non_five_star", "five_star"],
        yticklabels=["non_five_star", "five_star"],
    )
    plt.title("Best Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_plot(OUTPUT_DIR / "best_model_confusion_matrix.png")

    if hasattr(best_model, "predict_proba"):
        from sklearn.metrics import precision_recall_curve, roc_curve

        y_score = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        precision, recall, _ = precision_recall_curve(y_test, y_score)

        plt.figure(figsize=(6.5, 5))
        plt.plot(fpr, tpr, color="#1d3557", linewidth=2.5)
        plt.plot([0, 1], [0, 1], linestyle="--", color="#adb5bd")
        plt.title("Best Model ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        save_plot(OUTPUT_DIR / "best_model_roc_curve.png")

        plt.figure(figsize=(6.5, 5))
        plt.plot(recall, precision, color="#e76f51", linewidth=2.5)
        plt.title("Best Model Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        save_plot(OUTPUT_DIR / "best_model_pr_curve.png")

    if isinstance(X_test, pd.DataFrame):
        importance = permutation_importance(
            best_model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, scoring="f1_macro", n_jobs=1
        )
        feature_names = np.array(X_test.columns.tolist())
        importance_df = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance_mean": importance.importances_mean,
                    "importance_std": importance.importances_std,
                }
            )
            .sort_values("importance_mean", ascending=False)
            .head(15)
            .sort_values("importance_mean", ascending=True)
        )
        importance_df.to_csv(OUTPUT_DIR / "best_model_permutation_importance.csv", index=False, encoding="utf-8-sig")

        plt.figure(figsize=(9, 6))
        plt.barh(importance_df["feature"], importance_df["importance_mean"], xerr=importance_df["importance_std"], color="#2a9d8f")
        plt.title("Best Model Permutation Importance")
        plt.xlabel("Mean decrease in Macro-F1")
        plt.ylabel("")
        save_plot(OUTPUT_DIR / "best_model_permutation_importance.png")


def write_summary(results_df: pd.DataFrame, best_result: pd.Series) -> None:
    lines = [
        "# Baseline Modeling Summary",
        "",
        "## Task",
        "- Primary target: `five_star` vs `non_five_star (1-4)`",
        f"- Train/test split: 80/20 with `random_state={RANDOM_STATE}` and stratification",
        f"- Hyperparameter tuning: training split only, using `{CV_FOLDS}`-fold stratified cross-validation and `macro F1` as the refit metric",
        "",
        "## Experiments",
        results_df.to_markdown(index=False),
        "",
        "## Best Macro-F1",
        f"- Best experiment: `{best_result['experiment']}`",
        f"- Best CV Macro-F1: `{best_result['cv_best_score']}`",
        f"- Accuracy: `{best_result['accuracy']}`",
        f"- Macro F1: `{best_result['f1_macro']}`",
        f"- Precision Macro: `{best_result['precision_macro']}`",
        f"- Recall Macro: `{best_result['recall_macro']}`",
        f"- ROC-AUC: `{best_result['roc_auc']}`",
        f"- Average Precision: `{best_result['average_precision']}`",
        f"- Best parameters: `{best_result['best_params']}`",
        "",
        "## Notes",
        "- `metadata_only` uses reviewer/log-count, time, location, trip type, price range, and reservation metadata.",
        "- `sentiment_emotion_only` uses the lexicon-based sentiment score, VADER benchmark score, and NRC-derived emotion ratios.",
        "- `tfidf_only_knn` uses review text only.",
        "- `combined_*` uses text plus metadata plus sentiment/emotion features.",
        "- `*_hgb` refers to HistGradientBoostingClassifier as the boosting baseline.",
        "- `combined_xgb` refers to XGBoost on the combined feature set.",
        "- Result plots are written to the same `outputs/modeling` directory.",
    ]
    (OUTPUT_DIR / "baseline_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ROOT / "outputs" / "reviews_modeling_features.csv", encoding="utf-8-sig", low_memory=False)
    df = df.copy()
    df["has_reservation"] = df["has_reservation"].astype(str)
    df["trip_type"] = df["trip_type"].fillna("Unknown")
    df["price_range"] = df["price_range"].fillna("Unknown")
    df["lexicon_sentiment_label"] = df["lexicon_sentiment_label"].fillna("neutral")
    df["dominant_emotion"] = df["dominant_emotion"].fillna("none")

    X = df
    y = df[PRIMARY_TARGET]
    scale_pos_weight = float((y == 0).sum()) / float((y == 1).sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    experiments = {
        "metadata_only_knn": {
            "pipeline": make_tabular_knn(METADATA_NUMERIC, METADATA_CATEGORICAL),
            "param_grid": {
                "knn__n_neighbors": [11, 15, 21, 31],
                "knn__weights": ["uniform", "distance"],
            },
        },
        "metadata_only_rf": {
            "pipeline": make_tabular_rf(METADATA_NUMERIC, METADATA_CATEGORICAL),
            "param_grid": {
                "rf__n_estimators": [200, 400],
                "rf__max_depth": [None, 20],
                "rf__min_samples_leaf": [1, 2, 4],
            },
        },
        "metadata_only_hgb": {
            "pipeline": make_tabular_hgb(METADATA_NUMERIC, METADATA_CATEGORICAL),
            "param_grid": {
                "hgb__learning_rate": [0.05, 0.1],
                "hgb__max_depth": [None, 8],
                "hgb__max_leaf_nodes": [15, 31],
                "hgb__min_samples_leaf": [20, 50],
            },
        },
        "sentiment_emotion_only_knn": {
            "pipeline": make_tabular_knn(SENTIMENT_EMOTION_NUMERIC, SENTIMENT_EMOTION_CATEGORICAL),
            "param_grid": {
                "knn__n_neighbors": [11, 15, 21, 31],
                "knn__weights": ["uniform", "distance"],
            },
        },
        "sentiment_emotion_only_rf": {
            "pipeline": make_tabular_rf(SENTIMENT_EMOTION_NUMERIC, SENTIMENT_EMOTION_CATEGORICAL),
            "param_grid": {
                "rf__n_estimators": [200, 400],
                "rf__max_depth": [None, 20],
                "rf__min_samples_leaf": [1, 2, 4],
            },
        },
        "sentiment_emotion_only_hgb": {
            "pipeline": make_tabular_hgb(SENTIMENT_EMOTION_NUMERIC, SENTIMENT_EMOTION_CATEGORICAL),
            "param_grid": {
                "hgb__learning_rate": [0.05, 0.1],
                "hgb__max_depth": [None, 8],
                "hgb__max_leaf_nodes": [15, 31],
                "hgb__min_samples_leaf": [20, 50],
            },
        },
        "tfidf_only_knn": {
            "pipeline": make_tfidf_pipeline_knn(),
            "param_grid": {
                "tfidf__max_features": [3000, 4000],
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "knn__n_neighbors": [11, 15, 21],
                "knn__weights": ["uniform", "distance"],
            },
        },
        "combined_rf": {
            "pipeline": make_combined_rf(),
            "param_grid": {
                "prep__text__tfidf__max_features": [3000, 4000],
                "prep__text__svd__n_components": [80, 120],
                "rf__n_estimators": [200, 400],
                "rf__max_depth": [None, 20],
                "rf__min_samples_leaf": [1, 2],
            },
        },
        "combined_hgb": {
            "pipeline": make_combined_hgb(),
            "param_grid": {
                "prep__text__tfidf__max_features": [3000, 4000],
                "prep__text__svd__n_components": [80],
                "hgb__learning_rate": [0.05, 0.1],
                "hgb__max_depth": [None, 8],
                "hgb__max_leaf_nodes": [31],
                "hgb__min_samples_leaf": [20, 50],
            },
        },
        "combined_xgb": {
            "pipeline": make_combined_xgb(scale_pos_weight),
            "param_grid": {
                "prep__text__tfidf__max_features": [4000],
                "prep__text__svd__n_components": [80],
                "xgb__n_estimators": [200, 400],
                "xgb__learning_rate": [0.05, 0.1],
                "xgb__max_depth": [4, 6],
                "xgb__min_child_weight": [1, 3],
                "xgb__subsample": [0.8, 1.0],
                "xgb__colsample_bytree": [0.8, 1.0],
            },
        },
    }

    results = []
    reports = {}
    best_models = {}
    search_summaries = {}
    for name, config in experiments.items():
        pipeline = config["pipeline"]
        param_grid = config["param_grid"]
        if name == "tfidf_only_knn":
            search = run_search(name, pipeline, param_grid, X_train[TEXT_COL], y_train)
            result, best_model = run_experiment(name, search, X_test[TEXT_COL], y_test)
        else:
            search = run_search(name, pipeline, param_grid, X_train, y_train)
            result, best_model = run_experiment(name, search, X_test, y_test)
        search_summaries[name] = {
            "best_score": round(float(search.best_score_), 4),
            "best_params": to_serializable(search.best_params_),
        }
        best_models[name] = best_model
        reports[name] = result.pop("classification_report")
        result["best_params"] = json.dumps(to_serializable(result["best_params"]), ensure_ascii=False)
        results.append(result)
        print(f"Finished {name}: macro_f1={result['f1_macro']}", flush=True)

    results_df = pd.DataFrame(results).sort_values(["f1_macro", "accuracy"], ascending=False).reset_index(drop=True)
    results_df.to_csv(OUTPUT_DIR / "baseline_model_results.csv", index=False, encoding="utf-8-sig")

    with (OUTPUT_DIR / "baseline_model_reports.json").open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2)

    with (OUTPUT_DIR / "baseline_search_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(search_summaries, handle, indent=2)

    best_result = results_df.iloc[0]
    best_model_name = best_result["experiment"]
    best_model = best_models[best_model_name]
    write_summary(results_df, best_result)
    if best_model_name == "tfidf_only_knn":
        plot_results(results_df, best_model, X_test[[TEXT_COL]], y_test)
    else:
        plot_results(results_df, best_model, X_test, y_test)

    print(f"Wrote {OUTPUT_DIR / 'baseline_model_results.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'baseline_model_reports.json'}")
    print(f"Wrote {OUTPUT_DIR / 'baseline_search_summary.json'}")
    print(f"Wrote {OUTPUT_DIR / 'baseline_model_summary.md'}")


if __name__ == "__main__":
    main()
