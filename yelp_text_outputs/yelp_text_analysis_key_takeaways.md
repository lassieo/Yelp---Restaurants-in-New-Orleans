# Yelp Text Analysis: Key Analysis And Takeaways

## Key Analysis
The text analysis focused on New Orleans restaurant reviews from the Yelp dataset. The full matched dataset included **469,322 reviews** across **2,235 New Orleans businesses**. For text modeling, a **50,000-review sample** was used.

Sentiment was defined using Yelp star ratings:

- **Positive** = 4 or 5 stars
- **Neutral** = 3 stars
- **Negative** = 1 or 2 stars

In the 50,000-review sample:

- **36,205 reviews** were positive
- **5,595 reviews** were neutral
- **8,200 reviews** were negative

## Positive Review Patterns
The most common words in positive reviews included:

- `great`
- `good`
- `service`
- `delicious`
- `best`
- `amazing`
- `shrimp`

This suggests that positive Yelp reviews often focus on **food quality**, **strong service**, and **memorable local dishes**.

## Negative Review Patterns
The most common words in negative reviews included:

- `service`
- `ordered`
- `order`
- `time`
- `table`
- `didn`
- `minutes`

This suggests that negative Yelp reviews often focus on **wait times**, **service problems**, **order issues**, and **disappointing restaurant experiences**.

## Topic Modeling Themes
The topic model identified several recurring themes:

- Specific foods, including shrimp, fried chicken, oysters, sandwiches, pizza, burgers, and po-boys
- Service and staff experiences
- Wait time and table/server issues
- Positive recommendation language, including words like `great`, `amazing`, `friendly`, and `recommend`

## Main Takeaway
The text analysis shows that Yelp star ratings are strongly connected to what customers write in their reviews. Positive reviews emphasize **delicious food**, **good service**, and **local favorites**, while negative reviews focus more on **slow service**, **order problems**, **waiting**, and **poor table/server experiences**.

This analysis helps explain why restaurants receive high or low ratings beyond just the numeric star score.
