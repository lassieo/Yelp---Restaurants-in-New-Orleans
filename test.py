import pandas as pd
import json

# -----------------------------
# STEP 1: Load datasets
# -----------------------------
restaurants = pd.read_csv("new_orleans_restaurants.csv")

# Load business JSON (for attributes like price, takeout, etc.)
business_data = []
with open("yelp_academic_dataset_business.json", "r") as f:
    for line in f:
        business_data.append(json.loads(line))

business_df = pd.DataFrame(business_data)

# -----------------------------
# STEP 2: Filter business data to only restaurants in your dataset
# -----------------------------
business_df = business_df[business_df["business_id"].isin(restaurants["business_id"])]

# -----------------------------
# STEP 3: Extract attributes safely
# -----------------------------
def get_attribute(attr_dict, key):
    if isinstance(attr_dict, dict):
        return attr_dict.get(key)
    return None

# Extract price range
business_df["price_range"] = business_df["attributes"].apply(
    lambda x: get_attribute(x, "RestaurantsPriceRange2")
)

# Extract binary features
features = {
    "takeout": "RestaurantsTakeOut",
    "delivery": "RestaurantsDelivery",
    "reservations": "RestaurantsReservations",
    "outdoor_seating": "OutdoorSeating",
    "good_for_kids": "GoodForKids"
}

for new_col, attr_name in features.items():
    business_df[new_col] = business_df["attributes"].apply(
        lambda x: get_attribute(x, attr_name)
    )

# -----------------------------
# STEP 4: Clean binary features
# -----------------------------
def convert_to_binary(val):
    if val in [True, "True", "true", 1]:
        return 1
    else:
        return 0

for col in features.keys():
    business_df[col] = business_df[col].apply(convert_to_binary)

# Convert price to numeric
business_df["price_range"] = pd.to_numeric(business_df["price_range"], errors="coerce")

# -----------------------------
# STEP 5: Merge with restaurants CSV
# -----------------------------
merged_df = restaurants.merge(
    business_df[
        ["business_id", "price_range"] + list(features.keys())
    ],
    on="business_id",
    how="left"
)

# -----------------------------
# STEP 6: Build competition_density
# -----------------------------
competition = merged_df.groupby("postal_code").size().reset_index(name="competition_density")

merged_df = merged_df.merge(competition, on="postal_code", how="left")

# -----------------------------
# STEP 7: Handle missing values
# -----------------------------
# Fill price with median
merged_df["price_range"] = merged_df["price_range"].fillna(merged_df["price_range"].median())

# Fill binary features with 0
for col in features.keys():
    merged_df[col] = merged_df[col].fillna(0)
# -----------------------------
# STEP 8: Final feature table
# -----------------------------
feature_table = merged_df[
    [
        "business_id",
        "stars",
        "review_count",
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
# STEP 9: Save output
# -----------------------------
feature_table.to_csv("week2_feature_table.csv", index=False)

print("✅ Feature table created: feature_table.csv")
print(feature_table.head())