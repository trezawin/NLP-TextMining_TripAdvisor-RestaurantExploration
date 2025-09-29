# NLP Project: Restaurant Reviews in Berlin

## Authors
Eliott Barat-Chabeauti, Eugène Delecour, Kanyana Ntagungira, Treza Bawn Win, Xuejing Cao

## Overview
This project explores restaurant reviews from **TripAdvisor in Berlin** using **Natural Language Processing (NLP)**.
We applied text mining, unsupervised learning, and supervised learning to analyze sentiment, discover themes, and predict ratings.
We also addressed dataset imbalance challenges.

## Project Steps

### 1. Data Collection & Cleaning
- Webscraping reviews (including French-to-English translation).
- Cleaning: lowercasing, punctuation removal, stopword filtering, lemmatization (NLTK).
- Handling missing values (visit type, contribution count).
- Vectorization: TF-IDF for machine learning input.

### 2. Exploratory Analysis
- Sentiment trends by cuisine and visit type.
- Word clouds and n-grams (e.g., “highly recommend,” “service excellent”).
- Seasonal patterns: dips (2010 scandals), peaks (July & October tourist season).
- Cuisine-specific differences (e.g., pizza and Mexican had wider rating spreads).

### 3. Unsupervised Learning
- **LDA Topic Modeling**: uncovered themes like *ambiance, delicious food, German cuisine*.
- **K-Means Clustering + t-SNE**: identified cohesive vs overlapping feedback themes.
- **SBERT Embeddings**: semantic search for “romantic dining,” improving personalization.

### 4. Supervised Learning
- **Sentiment Classification**: Random Forest + TF-IDF, 97% accuracy (weak recall for negatives).
- **Rating Prediction**:
  - Baseline: Random Forest Regressor, R² ≈ 0.52.
  - Enhanced: XGBoost + Word2Vec embeddings, slight improvement (R² ≈ 0.54).

### 5. Dataset Imbalance
- Problem: 88% of data belonged to a single class (P3).
- Effect: Models biased toward majority class, poor performance on minority classes.
- Solutions: oversampling, undersampling, class weighting (explored methods).

## Key Insights
- Most reviews are positive, but **negative reviews cluster around business/solo visits**.
- Couples’ reviews are the most positive and romantic.
- German cuisine reviews highlight *beer* and *tradition*.
- Semantic search (SBERT) enables better recommendations than keyword matching.

## Tools & Libraries
- Python: `nltk`, `scikit-learn`, `deep-translator`, `pyLDAvis`, `Word2Vec`, `XGBoost`, `SBERT`.
- Visualization: `matplotlib`, `seaborn`, `t-SNE`.
