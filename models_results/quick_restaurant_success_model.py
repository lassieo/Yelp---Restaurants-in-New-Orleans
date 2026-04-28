from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path("/Users/laasyavenugopal/Desktop/Yelp JSON/yelp_dataset")
DATA_FILE = BASE_DIR / "new_orleans_restaurants.csv"
OUTPUT_DIR = BASE_DIR / "output"
SUMMARY_FILE = OUTPUT_DIR / "quick_model_summary.txt"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE).copy()

    # Remove malformed placeholder IDs so the model uses real restaurants only.
    df = df[df["business_id"].notna()].copy()
    df = df[df["business_id"] != "#NAME?"].copy()

    # Keep one row per business ID.
    df = df.sort_values("review_count", ascending=False).drop_duplicates("business_id")

    df["postal_code"] = df["postal_code"].astype(str).str.replace(".0", "", regex=False)
    df["log_review_count"] = np.log1p(df["review_count"])
    df["success"] = (df["stars"] >= 4.0).astype(int)
    return df


def build_model() -> Pipeline:
    numeric_features = ["review_count", "log_review_count", "latitude", "longitude", "is_open"]
    categorical_features = ["postal_code"]
    text_feature = "categories"

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "zip",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
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
                text_feature,
            ),
        ]
    )

    model = LogisticRegression(max_iter=4000, class_weight="balanced")

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


def extract_top_features(model: Pipeline, top_n: int = 12) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    top_positive_idx = np.argsort(coefficients)[-top_n:][::-1]
    top_negative_idx = np.argsort(coefficients)[:top_n]

    top_positive = [(feature_names[i], float(coefficients[i])) for i in top_positive_idx]
    top_negative = [(feature_names[i], float(coefficients[i])) for i in top_negative_idx]
    return top_positive, top_negative


def format_feature_list(items: list[tuple[str, float]]) -> str:
    return "\n".join(f"- {name}: {value:.3f}" for name, value in items)


def main() -> None:
    df = load_data()

    feature_columns = [
        "review_count",
        "log_review_count",
        "latitude",
        "longitude",
        "is_open",
        "postal_code",
        "categories",
    ]

    X = df[feature_columns]
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
    top_positive, top_negative = extract_top_features(pipeline)

    summary = f"""Quick Restaurant Success Model
========================================

Target definition:
- Success = Yelp stars >= 4.0

Modeling sample:
- Rows after removing malformed duplicate placeholder IDs: {len(df):,}
- Unique business IDs: {df['business_id'].nunique():,}
- Positive class rate: {y.mean():.3f}
- Train rows: {len(X_train):,}
- Test rows: {len(X_test):,}

Baseline:
- Majority class: {baseline['majority_class']}
- Baseline accuracy: {baseline['accuracy']:.3f}

Chosen model:
- Logistic Regression with balanced class weights
- Features: review_count, log_review_count, latitude, longitude, is_open, postal_code, TF-IDF of category text

Test performance:
- Accuracy: {accuracy:.3f}
- ROC AUC: {auc:.3f}

Classification report:
{report}

Top positive predictors of 4.0+ stars:
{format_feature_list(top_positive)}

Top negative predictors of 4.0+ stars:
{format_feature_list(top_negative)}
"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(summary)
    print(summary)
    print(f"Saved summary to {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
