# ==============================
# 1. Imports
# ==============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
# ==============================
# 2. Load Data
# ==============================
# Use your engineered dataset
df = pd.read_csv("final_feature_matrix.csv")
# ==============================
# 3. Select Features (UPDATED)
# ==============================
features = [
    'review_count',
    'review_frequency',
    'avg_sentiment',
    'checkin_count',
    'price_range',
    'takeout',
    'delivery',
    'reservations',
    'outdoor_seating',
    'good_for_kids',
    'competition_density'
]

target = 'stars'

# Drop missing values
df = df[features + [target]].dropna()

X = df[features]
y = df[target]
# ==============================
# 4. Train/Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# ==============================
# 5. Train Random Forest
# ==============================
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42
)

rf.fit(X_train, y_train)
# ==============================
# 6. Predictions
# ==============================
y_pred = rf.predict(X_test)
# ==============================
# 7. Evaluation
# ==============================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.3f}")
print(f"R^2: {r2:.3f}")
# ==============================
# 8. Feature Importance
# ==============================
importances = rf.feature_importances_
feature_names = X.columns

# Sort
indices = np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(8,5))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), feature_names[indices], rotation=45)
plt.tight_layout()
plt.show()