# Yelp Restaurants in New Orleans

This repository contains our team project for DSBA 6211 (Advanced Business Analytics), at University of North Carolina at Charlotte, on restaurant ratings in New Orleans using the Yelp Academic Dataset.

At a basic level, we wanted to answer a question that sounds simple but turns out not to be simple at all: what actually helps a restaurant earn strong Yelp ratings in a city where visibility, tourism, and customer experience are all tangled together?

New Orleans ended up being a great place to study that question. Some parts of the city bring in a huge amount of tourist traffic, especially around ZIP code `70130`, while other areas feel much more neighborhood-driven. That gave us a useful way to think about the difference between a restaurant that is popular because a lot of people pass through it and a restaurant that is genuinely well regarded.

## What we studied

We focused on New Orleans restaurants in the Yelp Academic Dataset and used a mix of:

- business-level information
- review text
- review timestamps
- check-in activity

The project combined exploratory analysis, text analysis, feature engineering, and predictive modeling. We also included a pandemic-related section to look at how review activity and check-ins changed during COVID.

## Main question

The project centers on one main idea:

What seems to drive restaurant success in New Orleans, and what does “success” actually mean in a dataset like Yelp?

For this project, success was tied mainly to Yelp rating rather than raw popularity. Review volume still mattered, but more as a measure of evidence and visibility than as a standalone definition of quality.

## Why New Orleans

We chose New Orleans because the city has a strong mix of tourism-heavy restaurants and local neighborhood spots. That made it easier to ask more interesting questions than we could have in a more uniform market.

A restaurant in a tourism-heavy area may collect a large number of reviews simply because of location. But that does not always mean it has the best ratings. That tension shows up clearly in this dataset and became one of the most useful themes in the project.

## Dataset

The project uses the Yelp Academic Dataset, mainly these files:

- `yelp_academic_dataset_business.json`
- `yelp_academic_dataset_review.json`
- `yelp_academic_dataset_checkin.json`
- `yelp_academic_dataset_tip.json`
- `yelp_academic_dataset_user.json`

From there, we filtered the data down to New Orleans restaurants and built cleaned working files for different parts of the analysis.

Depending on the stage of the workflow, that included:

- restaurant-level datasets
- review-level datasets with cleaned timestamps and text
- check-in summaries
- text-analysis outputs
- visuals for the report and presentation

## What we did

The project was split into a few connected parts.

### Data cleaning and merging

We started by filtering the Yelp data to New Orleans restaurants and then merging business, review, tip, and check-in information. A big part of this stage was cleaning dates and making sure the downstream analysis would work consistently across the team.

### Exploratory analysis and visualization

We looked at rating distributions, review-count skew, restaurant concentration by ZIP code, and differences between tourist-heavy and non-tourist areas. This part of the project helped us understand the structure of the city before moving into modeling.

### Review text analysis

We used review text to look at sentiment, common themes, and the kinds of words that show up in stronger versus weaker reviews. This helped us move past the numeric star rating and get closer to what customers were actually responding to.

### Feature engineering and modeling

We built a feature set using restaurant attributes, location-based features, review-derived signals, and check-in behavior. The goal was to see which types of information were most useful in predicting overall rating.

### Pandemic analysis

We also looked at how review volume, sentiment, and check-in patterns changed before COVID, during the initial shock period, and during recovery. This was a supporting section rather than the main focus of the project, but it added useful context.

## A few things we found

Some of the clearest patterns were:

- Restaurant activity is very uneven across the city, with some ZIP codes dominating review volume.
- High review volume and high rating are not the same thing.
- Positive reviews tend to talk about food quality, service, and memorable dishes.
- Negative reviews are much more likely to mention slow service, order problems, wait times, and table-side issues.
- Review counts are heavily skewed, so median values often tell a more honest story than means.
- During the COVID shock period, both reviews and check-ins fell sharply, but average ratings stayed fairly steady.

## Team roles

This was a team project, and the work was divided across different parts of the analysis:

- Laasya Venugopal:
  Project lead, framing, final integration
- Will Moore:
  Data cleaning, EDA support, pandemic analysis
- Bobby Deasy:
  Visualizations, ZIP-level trends, presentation visuals
- Christian Busk:
  Text analysis, sentiment, topic and keyword interpretation
- Kanniese Chen:
  Feature engineering, predictive modeling, model interpretation

## Repository contents

The exact file structure changed as the project developed, but the repo includes work in areas like:

```text
.
├── notebooks/                # cleaning, EDA, visualization, and modeling notebooks
├── cleaned datasets/         # filtered restaurant and review files
├── text outputs/             # sentiment, top-term, and topic-model outputs
├── figures/                  # charts for the report and presentation
├── planning docs/            # project framing and team workflow notes
└── README.md
```

In the more recent GitHub version of the project, parts of the workflow were also brought together into a single master notebook so the team could work from one end-to-end pipeline.

## Running the project

If you want to explore the notebooks locally, the usual workflow is:

1. Download the Yelp Academic Dataset.
2. Place the required JSON files in the project data folder.
3. Create and activate a Python environment.
4. Install dependencies if a `requirements.txt` file is included in the repo.
5. Start with the cleaning and exploratory notebooks, then move into text analysis, feature engineering, modeling, and visuals.

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

If the repo version you are using includes the unified master notebook, that is the best place to begin.

## A few things to keep in mind

There are a couple of caveats that matter when reading the results:

- Review counts are very skewed, so averages can be misleading on their own.
- Tourist-heavy areas can dominate citywide patterns.
- Yelp ratings capture customer perception, not direct business performance.
- Pandemic-era changes in review activity should not be treated as direct revenue measures.
- Some planning documents and later working files reflect slightly different dataset snapshots, so the final write-up should stay consistent about which version is being used.

## Why this matters

The point of the project was not just to predict ratings for the sake of prediction. We were more interested in what the data could say about restaurant experience in a real city.

In practice, that means questions like:

- What do customers praise most when they leave strong reviews?
- What kinds of service failures show up most often in lower-rated reviews?
- How much of restaurant visibility is driven by location rather than quality?
- Do review language and check-in behavior add useful information beyond basic business attributes?

Those are the kinds of questions that make the project useful from both a data perspective and a business perspective.

## Dataset and usage note

This project uses the Yelp Academic Dataset for academic work. Any reuse of the raw data should follow Yelp's dataset terms and usage guidelines.
