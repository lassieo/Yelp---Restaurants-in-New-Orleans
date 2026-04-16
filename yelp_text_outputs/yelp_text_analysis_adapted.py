from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS


BASE_DIR = Path(__file__).resolve().parent.parent
BUSINESS_FILE = BASE_DIR / "new_orleans_restaurants.csv"
REVIEW_FILE = BASE_DIR / "yelp_academic_dataset_review.json"
OUTPUT_DIR = BASE_DIR / "yelp_text_outputs"
SAMPLE_SIZE = 50000
RANDOM_STATE = 42


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def label_sentiment(stars: int) -> str:
    if stars >= 4:
        return "Positive"
    if stars <= 2:
        return "Negative"
    return "Neutral"


def load_new_orleans_reviews() -> pd.DataFrame:
    business_df = pd.read_csv(BUSINESS_FILE, usecols=["business_id", "name"])
    business_ids = set(business_df["business_id"])

    chunks = []
    for chunk in pd.read_json(REVIEW_FILE, lines=True, chunksize=100000):
        matched = chunk[chunk["business_id"].isin(business_ids)].copy()
        if not matched.empty:
            chunks.append(matched)

    review_df = pd.concat(chunks, ignore_index=True)
    review_df = review_df.merge(business_df, on="business_id", how="left")
    return review_df


def save_top_terms(series: pd.Series, label: str) -> None:
    vectorizer = CountVectorizer(stop_words="english", max_features=40)
    matrix = vectorizer.fit_transform(series)
    counts = matrix.sum(axis=0).A1
    terms = pd.DataFrame(
        {"term": vectorizer.get_feature_names_out(), "count": counts}
    ).sort_values("count", ascending=False)
    terms.to_csv(OUTPUT_DIR / f"top_terms_{label}.csv", index=False)


def save_topics(series: pd.Series) -> None:
    custom_stopwords = set(ENGLISH_STOP_WORDS).union(
        {
            "food",
            "place",
            "restaurant",
            "order",
            "ordered",
            "got",
            "really",
            "new",
            "orleans",
        }
    )
    vectorizer = CountVectorizer(
        stop_words=list(custom_stopwords),
        max_df=0.9,
        min_df=20,
        max_features=2500,
    )
    matrix = vectorizer.fit_transform(series)
    lda = LatentDirichletAllocation(
        n_components=5,
        random_state=RANDOM_STATE,
        learning_method="batch",
    )
    lda.fit(matrix)

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for idx, weights in enumerate(lda.components_):
        top_idx = weights.argsort()[-10:][::-1]
        topics.append(
            {
                "topic": idx + 1,
                "top_words": ", ".join(feature_names[i] for i in top_idx),
            }
        )

    pd.DataFrame(topics).to_csv(OUTPUT_DIR / "yelp_topics.csv", index=False)


def save_sentiment_plot(df: pd.DataFrame) -> None:
    counts = df["sentiment_label"].value_counts().reindex(
        ["Positive", "Neutral", "Negative"]
    )
    plt.figure(figsize=(7, 4))
    counts.plot(kind="bar", color=["#4c956c", "#f4a261", "#c1121f"])
    plt.title("New Orleans Yelp Review Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sentiment_distribution.png", dpi=200)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_new_orleans_reviews()
    df = df.dropna(subset=["text", "stars"]).copy()
    df["sentiment_label"] = df["stars"].astype(int).map(label_sentiment)
    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.len() > 0].copy()

    full_summary = {
        "full_review_count": len(df),
        "unique_businesses": int(df["business_id"].nunique()),
        "positive_reviews": int((df["sentiment_label"] == "Positive").sum()),
        "neutral_reviews": int((df["sentiment_label"] == "Neutral").sum()),
        "negative_reviews": int((df["sentiment_label"] == "Negative").sum()),
    }

    if len(df) > SAMPLE_SIZE:
        sample_df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
    else:
        sample_df = df.copy()

    sample_summary = {
        "sample_review_count": len(sample_df),
        "sample_unique_businesses": int(sample_df["business_id"].nunique()),
        "sample_positive_reviews": int((sample_df["sentiment_label"] == "Positive").sum()),
        "sample_neutral_reviews": int((sample_df["sentiment_label"] == "Neutral").sum()),
        "sample_negative_reviews": int((sample_df["sentiment_label"] == "Negative").sum()),
    }

    pd.DataFrame([{**full_summary, **sample_summary}]).to_csv(
        OUTPUT_DIR / "yelp_text_summary.csv",
        index=False,
    )

    save_top_terms(sample_df.loc[sample_df["sentiment_label"] == "Positive", "clean_text"], "positive")
    save_top_terms(sample_df.loc[sample_df["sentiment_label"] == "Negative", "clean_text"], "negative")
    save_topics(sample_df["clean_text"])
    save_sentiment_plot(sample_df)

    report_text = f"""# Yelp Text Analysis Outputs

Source businesses: {BUSINESS_FILE}
Source reviews: {REVIEW_FILE}

Full matched New Orleans review count: {full_summary['full_review_count']:,}
Unique New Orleans businesses: {full_summary['unique_businesses']:,}

Sample size used for text modeling: {sample_summary['sample_review_count']:,}
Sample positive reviews: {sample_summary['sample_positive_reviews']:,}
Sample neutral reviews: {sample_summary['sample_neutral_reviews']:,}
Sample negative reviews: {sample_summary['sample_negative_reviews']:,}

Sentiment labeling rule:
- Positive = 4 or 5 stars
- Neutral = 3 stars
- Negative = 1 or 2 stars
"""
    (OUTPUT_DIR / "README.md").write_text(report_text)

    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
