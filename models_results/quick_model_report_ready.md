# Quick Model Results For Report

## Model Objective
This quick model was built to predict whether a New Orleans restaurant would be classified as successful on Yelp, where success is defined as having a star rating of **4.0 or higher**.

## Data Used
- Source file: `new_orleans_restaurants.csv`
- Malformed placeholder IDs (`#NAME?`) were removed before modeling
- Final modeling sample: **2,235 unique restaurants**

## Model Choice
A **logistic regression classifier** was used because the report frames restaurant success as a binary outcome: above or below the 4.0-star threshold. This model is simple, interpretable, and appropriate for a quick baseline-plus analysis.

## Features Included
The model used only structured business-level features available in the restaurant file:
- review count
- log review count
- latitude
- longitude
- open/closed status
- ZIP code
- restaurant category text converted into TF-IDF features

## Performance
- Majority-class baseline accuracy: **0.583**
- Model accuracy: **0.662**
- ROC AUC: **0.739**

These results show that the model performs meaningfully better than a naive baseline, although its predictive power is still moderate. This suggests that structured business features contain useful information, but they do not fully explain restaurant ratings on their own.

## Interpretation
The model indicates that location and business category matter, but they are not enough to predict Yelp success with high precision. This supports one of the central themes of the project: **visibility and restaurant type help explain ratings, but they do not fully capture customer satisfaction**.

The moderate performance also suggests that adding richer variables such as:
- review sentiment
- service-related text features
- tourism exposure measures
- business operations data

would likely improve prediction quality.

## Suggested Report Sentence
"A quick logistic regression model predicting whether a restaurant achieved at least 4.0 Yelp stars improved over the majority-class baseline from 58.3% to 66.2% accuracy, with a ROC AUC of 0.739, indicating moderate predictive power from structured business features alone."

## Important Limitation
This model should be presented as a **quick supporting analysis**, not as the final strongest predictive model. It does not include review sentiment or detailed behavioral features, and its coefficients should be interpreted as associative rather than causal.
