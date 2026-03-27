## 5. Feature Engineering

### 5.1 Overview  
The goal of feature engineering in this project is to transform raw Yelp data into meaningful variables that help explain and predict restaurant success. Success is defined using a combination of star ratings and review volume. Features were derived from both structured restaurant data and review-level data, then aggregated to the business level to support modeling.

---

### 5.2 Data Sources  
Feature construction utilized two primary datasets:

- **Restaurant dataset (`new_orleans_restaurants.csv`)** containing business-level attributes such as location, categories, ratings, and review counts.  
- **Review dataset (`new_orleans_reviews.csv`)** containing individual user reviews, including text, ratings, engagement metrics, and timestamps.

These datasets were joined using the `business_id` key and aggregated to produce a single row per restaurant.

---

### 5.3 Feature Categories  

#### 5.3.1 Core Performance Features  
These features directly capture restaurant performance and serve as key predictors or targets in modeling:

- **Star Rating (`stars`)**  
  Represents the average customer rating of a restaurant. This variable is used as a primary target in regression models.

- **Review Count (`review_count`)**  
  Measures the total number of reviews received by a restaurant and serves as a proxy for popularity.

---

#### 5.3.2 Location-Based Features  

- **Postal Code (`postal_code`)**  
  Used as a proxy for geographic segmentation and neighborhood-level analysis.

- **Competition Density (`competition_density`)**  
  Defined as the number of restaurants within the same postal code. This feature captures market saturation and local competition.

---

#### 5.3.3 Category Features  

- **Restaurant Categories (`categories`)**  
  The category field was transformed into categorical indicators representing cuisine types (e.g., Italian, Chinese, Fast Food).  
  These were encoded using one-hot encoding to allow the model to identify patterns across different restaurant types.

---

#### 5.3.4 Engagement Features (Review-Level Aggregation)  

From the review dataset, user interaction signals were aggregated at the business level:

- **Average Useful Votes (`avg_useful`)**  
- **Average Funny Votes (`avg_funny`)**  
- **Average Cool Votes (`avg_cool`)**

These features reflect customer engagement and perceived value of reviews.

---

#### 5.3.5 Text-Derived Features  

Although computed in collaboration with the text analytics component, the following features were incorporated into the feature set:

- **Average Sentiment Score (`avg_sentiment`)**  
  Calculated using sentiment analysis (e.g., VADER or TextBlob) across all reviews for each restaurant.

- **Average Review Length (`avg_review_length`)**  
  Measured as the average number of words per review, capturing depth of customer feedback.

---

#### 5.3.6 Temporal Features (Optional Enhancements)  

- **Review Frequency (`reviews_per_month`)**  
  Derived from the review timestamp to capture growth trends and sustained engagement over time.

---

### 5.4 Data Transformation  

Several preprocessing steps were applied to ensure model readiness:

- Aggregation of review-level data to the business level  
- Handling of missing values through imputation or removal  
- Encoding of categorical variables (e.g., categories)  
- Conversion of boolean and numeric fields into consistent formats  
- Optional log transformations applied to skewed variables such as review counts  

---

### 5.5 Feature Table  

The final feature dataset consists of one row per restaurant, with variables including:

- Business identifiers and location attributes  
- Performance metrics (ratings, review counts)  
- Competition and category indicators  
- Aggregated engagement and sentiment features  

This structured feature matrix serves as the input for both regression and classification models in subsequent analysis.

---

### 5.6 Limitations  

Some potentially valuable features, such as price range, business attributes (e.g., WiFi, parking), and check-in data, were not available in the filtered CSV datasets. These may be incorporated in future work using the full Yelp JSON datasets.

---

### 5.7 Summary  

The engineered features provide a comprehensive representation of restaurant characteristics, customer engagement, and textual sentiment. This enables robust modeling of restaurant success and supports actionable business insights.
