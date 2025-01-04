import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load the saved logistic regression model and TF-IDF vectorizer
with open("logistic_regression_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Streamlit app configuration
st.set_page_config(page_title="Spam Email Detection", layout="centered")

# Title and Description
st.title("Spam Email Detection System")
st.write("Enter an email's text below to check if it's Spam or Not.")

# Input field
email_text = st.text_area("Enter the email content here:", height=200)

if st.button("Detect"):
    if email_text.strip():  # Check if the input is not empty
        # Preprocess and vectorize the input
        email_vector = tfidf_vectorizer.transform([email_text])
        
        # Predict using the loaded model
        prediction = model.predict(email_vector)[0]
        prediction_label = "Spam" if prediction == 1 else "Not Spam"
        
        # Display the result with different colors based on prediction
        if prediction == 1:
            # If Spam, display as error (red)
            st.error(f"The email is classified as: **{prediction_label}**")
        else:
            # If Not Spam, display as success (green)
            st.success(f"The email is classified as: **{prediction_label}**")
    else:
        st.error("Please enter valid email content to analyze.")

# Footer
st.write("---")
st.write("Spam Email Detection System| Powered by Machine Learning")
