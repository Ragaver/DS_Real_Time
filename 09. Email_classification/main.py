import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib


# Load model and vectorizer
model = joblib.load('model.pkl')        
v = joblib.load('vectorizer.pkl')        


# Streamlit UI
st.title("Predictive Model for Email Classification")
txt = st.text_input('Enter the email subject:')

if txt:
    # Text preprocessing
    lemmatizer = WordNetLemmatizer()
    message = re.sub('[^a-zA-Z0-9]', ' ', txt).lower().split()
    message = [lemmatizer.lemmatize(word, pos='v') for word in message if word not in stopwords.words('english')]
    message = ' '.join(message)
    
    # Transform input using loaded vectorizer
    x = v.transform([message]).toarray()
    
    # Predict
    pred = model.predict(x)[0]
    st.write(f'📧 Email type is: **{pred}**')
else:
    st.warning('Please enter a subject line to classify.')
