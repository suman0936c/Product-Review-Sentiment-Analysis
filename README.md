# Product Review Sentiment Analysis

Applied Deep Learning project for Central University of Haryana (CUH).

## Overview

This project predicts whether a mobile-phone product review is **Positive** or **Negative**. It compares Logistic Regression with a simple Artificial Neural Network (ANN).

## Dataset

Source: [Amazon Unlocked Mobile Phones Reviews](https://www.kaggle.com/datasets/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones). Ratings 1–2 become Negative and ratings 4–5 become Positive. Rating 3 is excluded because it is neutral or mixed. A balanced sample of 30,000 genuine reviews keeps training practical on a normal laptop.

## Method

The project removes missing and duplicate reviews, cleans text while preserving negation words, and converts text to 3,000 TF-IDF features. TF-IDF is fitted only on training data. It trains Logistic Regression and an ANN with Dense 64 ReLU, Dropout 0.3, Dense 32 ReLU, and a Sigmoid output. The ANN uses Adam, binary crossentropy, five epochs, and batch size 32.

## Actual Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 90.08% | 91.17% | 88.77% | 89.95% |
| Artificial Neural Network | 88.40% | 87.02% | 90.27% | 88.61% |

Logistic Regression achieved the higher accuracy and F1-score. The ANN achieved slightly higher positive-review recall.

## Run

```powershell
cd "Product-Review-Sentiment-Analysis"
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\streamlit.exe run app.py
```

`train.py` creates the cleaned dataset, EDA charts, saved ANN, saved TF-IDF vectorizer, confusion matrices, and metrics. `app.py` uses the saved ANN and the same vectorizer for new predictions.
