# Yelp Text Analysis Outputs

This folder contains the Yelp-specific text mining outputs used to support the New Orleans restaurant analysis.

## Source Data

- `new_orleans_restaurants.csv`
- `yelp_academic_dataset_review.json`

## Summary

- Full matched New Orleans review count: `469,322`
- Unique New Orleans businesses: `2,235`
- Sample size used for text modeling: `50,000`
- Sample positive reviews: `36,205`
- Sample neutral reviews: `5,595`
- Sample negative reviews: `8,200`

## Sentiment Labels

- Positive = 4 or 5 stars
- Neutral = 3 stars
- Negative = 1 or 2 stars

## Included Files

- `yelp_text_summary.csv`
- `top_terms_positive.csv`
- `top_terms_negative.csv`
- `yelp_topics.csv`
- `yelp_text_analysis_key_takeaways.md`
- `yelp_text_analysis_adapted.py`

## Not Included In GitHub

The large processed review sample CSV was kept out of the repository because it is better treated as a generated artifact than a source file.

- `new_orleans_review_sample_processed.csv`

## Why These Files Matter

These outputs help explain what customers are actually responding to in restaurant reviews. The text analysis adds context that the numeric star rating alone cannot provide, especially around service problems, wait-time issues, and the language associated with stronger reviews.