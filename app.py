import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Load the trained model from the file
with open('ComplaintModel.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

# Load the TF-IDF vectorizer from the file
with open('TfidfVectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Load the LabelEncoder from the file
with open('LabelEncoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

# Assuming you have a list of department names corresponding to numerical labels
department_names = ["Department_A", "Department_B", "Department_C", ...]

def preprocess_text(text):
    if isinstance(text, str):  # Check if text is a string
        text = text.lower()
        text = ' '.join([word for word in word_tokenize(text) if word.isalnum()])
        return text
    else:
        return ''  # Replace NaN or non-string values with an empty string

st.title("Complaint Department Prediction")

# User input for the complaint
complaint_input = st.text_area("Enter your complaint here:")

# Preprocess the complaint
preprocessed_complaint = preprocess_text(complaint_input)

if st.button("Submit"):
    if not preprocessed_complaint:
        st.warning("Please enter a complaint.")
    else:
        # TF-IDF Vectorization
        complaint_tfidf = tfidf_vectorizer.transform([preprocessed_complaint])

        # Make prediction
        predicted_department = clf.predict(complaint_tfidf)

        # Map numerical label to department name
        predicted_department_name = encoder.inverse_transform(predicted_department)

        # Display the predicted department
        st.success(f"Predicted Department: {predicted_department_name[0]}")
