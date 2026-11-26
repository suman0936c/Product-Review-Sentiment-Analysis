# Product Review Sentiment Analysis Project Report

## Submitted for

Applied Deep Learning Training, Central University of Haryana

## Abstract

This project classifies real product reviews as Positive or Negative. It uses public Amazon Unlocked Mobile Phones reviews, simple text cleaning, TF-IDF vectorization, Logistic Regression, and a basic Artificial Neural Network. The project compares both models using standard classification metrics and provides a Streamlit application for predictions on new reviews.

## Problem Statement

Online stores contain a large number of customer reviews. Reading every review manually is difficult. This project automatically identifies whether a review expresses a positive or negative opinion.

## Objectives

- Prepare real review text and ratings for binary classification.
- Visualize review sentiment, ratings, review length, and common words.
- Train and compare Logistic Regression and an ANN.
- Build a small web application for new-review prediction.

## Dataset

The source is the Amazon Unlocked Mobile Phones Reviews dataset published by PromptCloud on Kaggle. The source file includes `Reviews` and `Rating` columns. Ratings 1–2 are Negative, ratings 4–5 are Positive, and rating 3 is removed. The project uses a balanced random sample of 30,000 real reviews after cleaning.

## Data Preparation

The program removes rows with missing review text or ratings and removes duplicate reviews. It converts each review to lowercase, removes links, punctuation, numbers, special characters, and repeated spaces. It preserves negation words such as `not`, `no`, and `never` because they affect sentiment. The final target uses 0 for Negative and 1 for Positive.

## Exploratory Data Analysis

The project saves five charts in the `images` folder: sentiment distribution, rating distribution, review-length distribution, common positive words, and common negative words. These charts help explain the class balance, source ratings, review size, and frequent vocabulary.

## Feature Extraction

Computers cannot directly use text as model input. TF-IDF converts words into numerical features. A word receives more importance when it is useful in one review but uncommon across reviews. The vectorizer uses at most 3,000 features and is fitted only on training text to avoid data leakage.

## Models

Logistic Regression provides a simple and strong baseline for sparse TF-IDF features. The ANN contains a 64-neuron ReLU layer, Dropout 0.3, a 32-neuron ReLU layer, and a one-neuron Sigmoid output. It uses Adam optimization, binary crossentropy loss, five epochs, and batch size 32.

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 90.08% | 91.17% | 88.77% | 89.95% |
| Artificial Neural Network | 88.40% | 87.02% | 90.27% | 88.61% |

Logistic Regression produced the best accuracy and F1-score on the test set. The ANN had higher recall for positive reviews. Both models demonstrate that TF-IDF features can support useful sentiment classification with simple methods.

## Streamlit Application

The application asks a user to enter a product review. It applies the same cleaning function and saved TF-IDF vectorizer, then loads the saved ANN to show Positive or Negative sentiment with a probability.

## Conclusion

The project demonstrates the complete beginner-friendly NLP workflow: collect real review data, clean text, convert it to numerical features, train two binary classifiers, evaluate their actual results, and deploy a prediction interface. Logistic Regression performed best for this dataset, while the ANN demonstrates a basic deep-learning approach.

## Future Scope

Future work could use a larger balanced sample, include a neutral class, improve preprocessing, or compare additional basic models. These changes are outside the current project scope.
