from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path("/Users/laasyavenugopal/Desktop/Yelp JSON/yelp_dataset")
DATA_FILE = BASE_DIR / "new_orleans_restaurants.csv"
OUTPUT_DIR = BASE_DIR / "output" / "model_results"
SUMMARY_FILE = OUTPUT_DIR / "quick_random_forest_summary.txt"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE).copy()
    df = df[df["business_id"].notna()].copy()
    df = df[df["business_id"] != "#NAME?"].copy()
    df = df.sort_values("review_count", ascending=False).drop_duplicates("business_id")
    df["postal_code"] = df["postal_code"].astype(str).str.replace(".0", "", regex=False)
    df["log_review_count"] = np.log1p(df["review_count"])
    df["success"] = (df["stars"] >= 4.0).astype(int)
    return df


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                ["review_count", "log_review_count", "latitude", "longitude", "is_open"],
            ),
            (
                "zip",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["postal_code"],
            ),
            (
                "categories",
                TfidfVectorizer(
                    lowercase=True,
                    token_pattern=r"(?u)\b[\w&/]+\b",
                    ngram_range=(1, 2),
                    min_df=5,
                    max_features=250,
                ),
                "categories",
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def majority_baseline(y_train: pd.Series, y_test: pd.Series) -> dict[str, float]:
    majority_class = int(y_train.mode()[0])
    predictions = np.full(len(y_test), majority_class)
    return {
        "majority_class": majority_class,
        "accuracy": accuracy_score(y_test, predictions),
    }


def get_top_importances(model: Pipeline, top_n: int = 15) -> list[tuple[str, float]]:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    top_idx = np.argsort(importances)[-top_n:][::-1]
    cleaned = []
    for i in top_idx:
        name = feature_names[i]
        name = name.replace("categories__", "")
        name = name.replace("zip__postal_code_", "ZIP ")
        name = name.replace("num__", "")
        cleaned.append((name, float(importances[i])))
    return cleaned


def format_feature_list(items: list[tuple[str, float]]) -> str:
    return "\n".join(f"- {name}: {value:.4f}" for name, value in items)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    X = df[["review_count", "log_review_count", "latitude", "longitude", "is_open", "postal_code", "categories"]]
    y = df["success"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    baseline = majority_baseline(y_train, y_test)

    pipeline = build_model()
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    pred_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, pred_prob)
    report = classification_report(y_test, pred, digits=3)
    top_features = get_top_importances(pipeline)

    summary = f"""Quick Random Forest Restaurant Success Model
========================================

Target definition:
- Success = Yelp stars >= 4.0

Modeling sample:
- Unique restaurants: {len(df):,}
- Positive class rate: {y.mean():.3f}
- Train rows: {len(X_train):,}
- Test rows: {len(X_test):,}

Baseline:
- Majority class: {baseline['majority_class']}
- Baseline accuracy: {baseline['accuracy']:.3f}

Chosen model:
- Random Forest Classifier
- Features: review_count, log_review_count, latitude, longitude, is_open, postal_code, TF-IDF of category text

Test performance:
- Accuracy: {accuracy:.3f}
- ROC AUC: {auc:.3f}

Classification report:
{report}

Top feature importances:
{format_feature_list(top_features)}
"""

    SUMMARY_FILE.write_text(summary)
    print(summary)
    print(f"Saved summary to {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
