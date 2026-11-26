"""Simple Streamlit interface for product review sentiment prediction."""

import os
import pickle
import re

import streamlit as st
import tensorflow as tf


def clean_text(text):
    """Apply the same simple cleaning used during training."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource
def load_resources():
    with open("models/tfidf_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)
    model = tf.keras.models.load_model("models/ann_model.keras")
    return vectorizer, model


st.set_page_config(page_title="Product Review Sentiment Analysis", page_icon="💬")
st.title("Product Review Sentiment Analysis")
st.write("Enter a product review to predict whether its sentiment is Positive or Negative.")

if not (os.path.exists("models/tfidf_vectorizer.pkl") and os.path.exists("models/ann_model.keras")):
    st.warning("Models are not ready yet. Run `python train.py` once from the project folder.")
    st.stop()

review = st.text_area("Enter Product Review", placeholder="This phone has excellent battery life.", height=130)
if st.button("Predict Sentiment"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        vectorizer, model = load_resources()
        features = vectorizer.transform([clean_text(review)]).toarray()
        probability = float(model.predict(features, verbose=0)[0][0])
        if probability >= 0.5:
            st.success(f"Prediction: Positive ({probability:.1%} positive probability)")
        else:
            st.error(f"Prediction: Negative ({1 - probability:.1%} negative probability)")
