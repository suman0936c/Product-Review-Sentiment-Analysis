"""Train simple ML and ANN sentiment models on real product reviews."""

import os
import re
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
MAX_REVIEWS = 30000
MAX_FEATURES = 3000
os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def clean_text(text):
    """Use simple cleaning while keeping words such as not and never."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join("images", filename), dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_model(name, true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="binary", zero_division=0
    )
    matrix = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted sentiment")
    plt.ylabel("Actual sentiment")
    save_plot(f"{name.lower().replace(' ', '_')}_confusion_matrix.png")
    print(f"\n{name} classification report:\n")
    print(classification_report(true_labels, predicted_labels,
                                target_names=["Negative", "Positive"], zero_division=0))
    return {"Model": name, "Accuracy": accuracy, "Precision": precision,
            "Recall": recall, "F1-score": f1}


def main():
    raw_df = pd.read_csv("data/reviews.csv", usecols=["Rating", "Reviews"])
    raw_df.columns = ["rating", "review_text"]
    raw_df = raw_df.dropna(subset=["rating", "review_text"])
    raw_df = raw_df.drop_duplicates(subset="review_text")
    raw_df = raw_df[raw_df["rating"].isin([1, 2, 4, 5])].copy()
    raw_df["sentiment"] = (raw_df["rating"] >= 4).astype(int)
    raw_df["clean_review"] = raw_df["review_text"].apply(clean_text)
    raw_df = raw_df[raw_df["clean_review"].str.len() > 0].copy()

    count_per_class = min(MAX_REVIEWS // 2, raw_df["sentiment"].value_counts().min())
    negative = raw_df[raw_df["sentiment"] == 0].sample(count_per_class, random_state=RANDOM_STATE)
    positive = raw_df[raw_df["sentiment"] == 1].sample(count_per_class, random_state=RANDOM_STATE)
    clean_df = pd.concat([negative, positive], ignore_index=True).sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)
    clean_df["review_length"] = clean_df["clean_review"].str.split().str.len()
    clean_df.to_csv("data/cleaned_reviews.csv", index=False)
    print("Cleaned data shape:", clean_df.shape)

    # Required beginner-friendly EDA charts.
    labels = clean_df["sentiment"].map({0: "Negative", 1: "Positive"})
    plt.figure(figsize=(6, 4)); sns.countplot(x=labels, hue=labels, legend=False, palette="Set2")
    plt.title("Number of Positive and Negative Reviews"); plt.xlabel("Sentiment"); plt.ylabel("Number of reviews")
    save_plot("sentiment_distribution.png")
    plt.figure(figsize=(6, 4)); sns.countplot(x=clean_df["rating"], color="#4C78A8")
    plt.title("Rating Distribution"); plt.xlabel("Rating"); plt.ylabel("Number of reviews")
    save_plot("rating_distribution.png")
    plt.figure(figsize=(7, 4)); sns.histplot(clean_df["review_length"], bins=40, color="#59A14F")
    plt.xlim(0, clean_df["review_length"].quantile(0.99)); plt.title("Review Length Distribution")
    plt.xlabel("Words in review"); plt.ylabel("Number of reviews")
    save_plot("review_length_distribution.png")
    for label, text, color in [("positive", clean_df.loc[clean_df.sentiment == 1, "clean_review"], "#2E8B57"),
                               ("negative", clean_df.loc[clean_df.sentiment == 0, "clean_review"], "#C44E52")]:
        words = " ".join(text).split()
        common = pd.Series(words).value_counts().head(15).sort_values()
        plt.figure(figsize=(8, 5)); common.plot(kind="barh", color=color)
        plt.title(f"Most Common Words in {label.title()} Reviews"); plt.xlabel("Frequency")
        save_plot(f"common_words_{label}.png")

    X_train, X_test, y_train, y_test = train_test_split(
        clean_df["clean_review"], clean_df["sentiment"], test_size=0.2,
        random_state=RANDOM_STATE, stratify=clean_df["sentiment"]
    )
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    with open("models/tfidf_vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)

    logistic_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logistic_model.fit(X_train_tfidf, y_train)
    logistic_predictions = logistic_model.predict(X_test_tfidf)
    results = [evaluate_model("Logistic Regression", y_test, logistic_predictions)]

    ann_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_tfidf.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    ann_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = ann_model.fit(X_train_tfidf.toarray(), y_train, validation_split=0.1,
                            epochs=5, batch_size=32, verbose=2)
    ann_probabilities = ann_model.predict(X_test_tfidf.toarray(), verbose=0).ravel()
    ann_predictions = (ann_probabilities >= 0.5).astype(int)
    results.append(evaluate_model("Artificial Neural Network", y_test, ann_predictions))
    ann_model.save("models/ann_model.keras")

    history_df = pd.DataFrame(history.history)
    plt.figure(figsize=(7, 4)); plt.plot(history_df["accuracy"], label="Training accuracy")
    plt.plot(history_df["val_accuracy"], label="Validation accuracy")
    plt.title("ANN Training Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    save_plot("ann_training_accuracy.png")
    plt.figure(figsize=(7, 4)); plt.plot(history_df["loss"], label="Training loss")
    plt.plot(history_df["val_loss"], label="Validation loss")
    plt.title("ANN Training Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    save_plot("ann_training_loss.png")

    results_df = pd.DataFrame(results)
    results_df.to_csv("models/model_comparison.csv", index=False)
    print("\nModel comparison:\n", results_df.round(4))


if __name__ == "__main__":
    main()
