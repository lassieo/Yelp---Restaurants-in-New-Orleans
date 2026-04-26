import pandas as pd
import json
from datetime import datetime
from textblob import TextBlob

# -----------------------------
# STEP 1: Load datasets
# -----------------------------
restaurants = pd.read_csv("new_orleans_restaurants.csv")
reviews = pd.read_csv("new_orleans_reviews.csv")
features = pd.read_csv("new_feature_table.csv")

# -----------------------------
# STEP 2: Compute SENTIMENT
# -----------------------------
# Ensure text is string
reviews["text"] = reviews["text"].astype(str)

# Compute polarity for each review
reviews["polarity"] = reviews["text"].apply(lambda x: TextBlob(x).polarity)

# Compute average sentiment per business
sentiment_stats = reviews.groupby("business_id").agg(
    avg_sentiment=("polarity", "mean")
).reset_index()

# -----------------------------
# STEP 3: Compute REVIEW FREQUENCY
# -----------------------------
reviews["date"] = pd.to_datetime(reviews["date"])

review_stats = reviews.groupby("business_id").agg(
    total_reviews=("review_id", "count"),
    first_review=("date", "min"),
    last_review=("date", "max")
).reset_index()

# Compute active months
review_stats["active_months"] = (
    (review_stats["last_review"] - review_stats["first_review"]).dt.days / 30
)

# Avoid division by zero
review_stats["active_months"] = review_stats["active_months"].replace(0, 1)

# Compute review frequency
review_stats["review_frequency"] = (
    review_stats["total_reviews"] / review_stats["active_months"]
)

review_stats = review_stats[["business_id", "review_frequency"]]

# -----------------------------
# STEP 4: Load CHECK-IN DATA
# -----------------------------
checkin_data = []

with open("yelp_academic_dataset_checkin.json", "r") as f:
    for line in f:
        checkin_data.append(json.loads(line))

checkin_df = pd.DataFrame(checkin_data)

# -----------------------------
# STEP 5: Compute CHECK-IN COUNT
# -----------------------------
def count_checkins(date_str):
    if pd.isna(date_str):
        return 0
    return len(date_str.split(","))

checkin_df["checkin_count"] = checkin_df["date"].apply(count_checkins)
checkin_df = checkin_df[["business_id", "checkin_count"]]

# -----------------------------
# STEP 6: Merge ALL FEATURES
# -----------------------------
final_df = features.merge(review_stats, on="business_id", how="left")
final_df = final_df.merge(checkin_df, on="business_id", how="left")
final_df = final_df.merge(sentiment_stats, on="business_id", how="left")

# -----------------------------
# STEP 7: Handle Missing Values
# -----------------------------
final_df["review_frequency"] = final_df["review_frequency"].fillna(0)
final_df["checkin_count"] = final_df["checkin_count"].fillna(0)
final_df["avg_sentiment"] = final_df["avg_sentiment"].fillna(0)

# -----------------------------
# STEP 8: Final Feature Matrix
# -----------------------------
final_features = final_df[
    [
        "business_id",
        "stars",
        "review_count",
        "review_frequency",
        "avg_sentiment",
        "checkin_count",
        "price_range",
        "takeout",
        "delivery",
        "reservations",
        "outdoor_seating",
        "good_for_kids",
        "postal_code",
        "competition_density"
    ]
]

# -----------------------------
# STEP 9: Final Cleaning
# -----------------------------
final_features = final_features.drop_duplicates(subset="business_id")

binary_cols = [
    "takeout",
    "delivery",
    "reservations",
    "outdoor_seating",
    "good_for_kids"
]

for col in binary_cols:
    final_features[col] = final_features[col].astype(int)

# -----------------------------
# STEP 10: Save Output
# -----------------------------
final_features.to_csv("final_feature_matrix.csv", index=False)

print("✅ Final feature matrix created: final_feature_matrix.csv")
print(final_features.head())