# Problem Statement And Success Metrics

New Orleans Restaurant Analysis | Yelp Open Dataset

## Dataset Snapshot

Before defining success, all decisions were grounded in what the dataset actually contains. The New Orleans restaurant subset of the Yelp Open Dataset was reviewed and the following key statistics were established.

- Total restaurant businesses: 6,215
- Currently open: 4,654 (75%)
- Permanently closed: 1,561 (25%)
- Mean star rating: 3.82
- Rated 4.0 stars or higher: 61.3%
- Rated below 3.0 stars: 12.2%
- Median review count: 22
- Mean review count: 100
- Max reviews: 7,568 (Acme Oyster)

The dataset shows a right-skewed ratings distribution: most restaurants cluster between 3.5 and 4.5 stars. Review counts are heavily right-skewed: the median restaurant has only 22 reviews, while the mean is pulled to 100 by a small number of high-traffic tourist restaurants concentrated in ZIP code 70130, the French Quarter and CBD. This skew has direct implications for how success should be defined and modeled.

Key data insight driving the success definition:

1,615 businesses (26%) have fewer than 10 reviews. Rating alone is unreliable for these, since a restaurant with 5 reviews and a 5.0 rating is not meaningfully "better" than one with 200 reviews and a 4.3 rating. Any success definition must account for evidence strength.

Additionally, 1,561 businesses (25%) are permanently closed. Success will be modeled on open businesses only (`n = 4,654`) to ensure business relevance.

## Problem Statement

This project uses the Yelp Open Dataset to identify which restaurant characteristics best predict overall Yelp rating in New Orleans, Louisiana. It combines structured business attributes such as price range, location, hours, and amenities with text-derived features from customer reviews and tips, including sentiment, keywords, and review length, and with behavioral signals from check-in data to build a predictive model of restaurant quality as measured by Yelp star rating.

A secondary focus tests whether these predictors differ between newer and established restaurants, capturing whether reputation-building follows different dynamics depending on business age.

Analysis is restricted to currently open businesses (`n = 4,654`) with a minimum evidence threshold of 10 or more reviews, yielding a modeling sample of approximately 3,040 businesses.

## Why New Orleans Is The Right City For This Question

- Tourism vs. local demand: ZIP 70130 (French Quarter / CBD) contains 1,511 businesses, nearly 25% of the entire dataset, and is dominated by tourist-facing restaurants. The surrounding ZIPs (70115, 70116, 70119) are more residential and local. This natural split makes it possible to test whether the drivers of success differ by audience type.
- Extreme review inequality: the top 10 most-reviewed restaurants all have 3,900+ reviews, while the median has 22. This makes New Orleans a useful test case for whether review volume independently predicts rating quality, or whether it simply reflects tourism exposure.
- COVID resilience dimension: New Orleans was one of the first U.S. cities hit by COVID-19 in early 2020 and has a large hospitality-dependent economy. The dataset spans the pre- and post-pandemic period, making it possible to test which restaurant attributes, such as takeout, delivery, and outdoor seating, predicted resilience, though this is framed as a supporting analysis rather than the primary focus.

## Success Metrics Definition

Restaurant success is defined using a two-part framework. The primary metric drives all modeling. The secondary metric controls for visibility and evidence quality.

### Primary Metric: Star Rating (Continuous)

- Variable: `stars` (Yelp overall business rating)
- Type: continuous, values 1.0, 1.5, 2.0 ... 5.0 in 0.5 increments
- Range in dataset: 1.0 to 5.0, mean = 3.82, median = 4.0
- Use in modeling: regression target for exact rating and classification target for high vs. low rating
- Classification split: high success = `stars >= 4.0` (61.3% of open businesses), low success = `stars < 3.5` (27.4%), and borderline `3.5` excluded from binary classification to sharpen the signal

Rationale: Star rating is the most direct and universally understood signal of perceived restaurant quality on Yelp. It is the variable customers use when deciding where to eat. Predicting it accurately, and identifying what drives it, has clear analytical value. The 4.0 cutoff for "high success" is grounded in the actual distribution: it represents the median and is a natural threshold above which restaurants are considered solidly well regarded.

### Secondary Metric: Review Count (Evidence Strength Control)

- Variable: `review_count`
- Type: continuous, used as a covariate and filter rather than a standalone target
- Minimum threshold: 10 reviews required for inclusion in the modeling sample
- Role in analysis: control variable in regression, secondary outcome for popularity analysis, and potential weighting or stratification variable
- Why not a primary metric: review count measures visibility and tourist traffic more than quality. The top 10 restaurants by count are all tourist-heavy French Quarter spots, not necessarily the city's best-regarded dining.

Rationale: Review count is deliberately not treated as a co-equal success metric. It is heavily confounded by location, since restaurants in the French Quarter receive tourist traffic that inflates counts independent of quality. Using it as a primary target would cause the model to predict tourism exposure rather than restaurant excellence. It is retained as a control variable and as a secondary descriptive outcome.

## Classification Thresholds For Modeling

| Class | Star Range | Count (open) | Share |
|---|---|---:|---:|
| High success | 4.0 - 5.0 | ~2,854 | 61.3% |
| Borderline (excluded from binary) | 3.5 | ~474 | 10.2% |
| Low success | 1.0 - 3.0 | ~1,326 | 28.5% |

Counts are approximate and applied to open businesses (`n = 4,654`) only. The final modeling sample with 10+ reviews is approximately 3,040 businesses.

## Finalized Research Questions

### RQ1: What does the New Orleans restaurant landscape look like?

- How are ratings and review counts distributed across the city?
- Which ZIP codes have the highest restaurant concentration, and how do ratings compare across ZIPs?
- How does the mix of restaurant subcategories vary by neighborhood?

### RQ2: Does price range predict restaurant success?

- Do higher-priced restaurants (`$$$$`) receive significantly higher star ratings?
- Does the price-rating relationship hold across all ZIP codes, or only outside the French Quarter?
- Is price range a significant predictor in the regression model after controlling for review count?

### RQ3: Does competition density affect ratings?

- Do restaurants in high-density ZIP codes, such as 70130 with 1,511 businesses, rate differently from low-density ZIPs?
- Is competition density a significant feature in the model, or is it absorbed by ZIP-level fixed effects?

### RQ4: What do reviews reveal about high- vs. low-rated restaurants?

- Which words, sentiments, and topics predict high versus low star ratings?
- Do higher-rated restaurants receive longer, more positive reviews?
- What negative themes appear most often in low-rated restaurants?

### RQ5: What combination of features best predicts overall star rating?

- Which model performs best: a linear regression baseline or a random forest?
- Which features, structured attributes, text-derived signals, or check-in behavior, contribute the most predictive power?
- Does adding text features or check-in features meaningfully improve prediction beyond structured attributes alone?

### RQ6: Do newer vs. established restaurants differ in what drives success?

- Are newer restaurants more volatile in ratings and sentiment?
- Does the importance of delivery, takeout, and check-in momentum differ by business age?

## Modeling Sample Definition

Final modeling sample criteria:

1. `is_open = 1` (currently open businesses only)
2. `review_count >= 10` (minimum evidence threshold)
3. `city = New Orleans` and category includes `Restaurants`

Estimated modeling sample: approximately 3,040 businesses.

Exploratory analysis, including visualizations and text mining, will use the full open dataset (`n = 4,654`) to preserve breadth. The review-count filter applies only to regression and classification modeling.
