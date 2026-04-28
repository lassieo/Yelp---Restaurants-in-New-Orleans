# Quick Random Forest Results For Report

## Model Objective
This quick model was built to predict whether a New Orleans restaurant would be classified as successful on Yelp, where success is defined as having a star rating of **4.0 or higher**.

## Data Used
- Source file: `new_orleans_restaurants.csv`
- Malformed placeholder IDs (`#NAME?`) were removed before modeling
- Final modeling sample: **2,235 unique restaurants**

## Model Choice
A **Random Forest classifier** was used as a flexible nonlinear alternative to logistic regression. This model can capture interactions across review volume, location, ZIP code, open status, and restaurant category text without requiring those relationships to be strictly linear.

## Features Included
- review count
- log review count
- latitude
- longitude
- open/closed status
- ZIP code
- restaurant category text converted into TF-IDF features

## Performance
- Majority-class baseline accuracy: **0.583**
- Random Forest accuracy: **0.678**
- ROC AUC: **0.752**

These results are slightly stronger than the logistic regression model, suggesting that the relationship between structured business features and Yelp success is not fully linear.

## Interpretation
The Random Forest model indicates that review volume, geography, ZIP code, and category-related features all contribute to prediction quality. However, even this stronger model remains moderate rather than highly predictive, which supports the conclusion that structured business attributes explain only part of restaurant success on Yelp.

## Suggested Report Sentence
"A quick Random Forest classifier predicting whether a restaurant achieved at least 4.0 Yelp stars improved over the majority-class baseline from 58.3% to 67.8% accuracy, with a ROC AUC of 0.752, slightly outperforming logistic regression and suggesting mild nonlinear structure in the predictors."

## Important Limitation
This model should still be treated as a supporting analysis. It does not include review sentiment, check-in aggregates, tourism exposure, or business operations data, so its predictions reflect only a limited set of structured restaurant features.
