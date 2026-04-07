import pandas as pd
import json
from datetime import datetime

# -----------------------------
# STEP 1: Load datasets
# -----------------------------
restaurants = pd.read_csv("new_orleans_restaurants.csv")
reviews = pd.read_csv("new_orleans_reviews.csv")

# -----------------------------
# STEP 2: Create SENTIMENT (temporary = 0)
# -----------------------------
restaurants["avg_sentiment"] = 0.0

# -----------------------------
# STEP 3: Compute REVIEW FREQUENCY
# -----------------------------
# Convert date column
reviews["date"] = pd.to_datetime(reviews["date"])

# Group reviews by business
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

# Keep only needed columns
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
# STEP 6: Load Week 2 Feature Table
# -----------------------------
features = pd.read_csv("week2_feature_table.csv")

# -----------------------------
# STEP 7: Merge EVERYTHING
# -----------------------------
final_df = features.merge(review_stats, on="business_id", how="left")
final_df = final_df.merge(checkin_df, on="business_id", how="left")
final_df["avg_sentiment"] = 0.0
# -----------------------------
# STEP 8: Handle Missing Values
# -----------------------------
final_df["review_frequency"] = final_df["review_frequency"].fillna(0)
final_df["checkin_count"] = final_df["checkin_count"].fillna(0)

# Sentiment already set to 0

# -----------------------------
# STEP 9: Final Feature Matrix
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
# STEP 10: Final Cleaning
# -----------------------------
# Remove duplicates
final_features = final_features.drop_duplicates(subset="business_id")

# Ensure correct types
binary_cols = ["takeout", "delivery", "reservations", "outdoor_seating", "good_for_kids"]
for col in binary_cols:
    final_features[col] = final_features[col].astype(int)

# -----------------------------
# STEP 11: Save Output
# -----------------------------
final_features.to_csv("final_feature_matrix.csv", index=False)

print("✅ Final feature matrix created: final_feature_matrix.csv")
print(final_features.head())