import streamlit as st
import joblib
import pandas as pd
import language_tool_python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Load LanguageTool for grammar checking
import language_tool_python

tool = language_tool_python.LanguageTool('en-US')

# Load or Train Sentiment Analysis Model
@st.cache_resource
def train_model():
    # Load dataset (replace with your custom dataset)
    df = pd.read_csv("sentiment_data.csv")  # Ensure you have a dataset file

    X, y = df['text'], df['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create model pipeline
    model = make_pipeline(CountVectorizer(), MultinomialNB())

    # Train model
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, 'sentiment_model.pkl')

    # Evaluate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

# Load trained model
try:
    model = joblib.load('sentiment_model.pkl')
except:
    model, accuracy = train_model()
    st.write(f"Model Trained with Accuracy: {accuracy:.2f}")

# Streamlit UI
st.title(" Sentiment Analysis App 🔍 ")

st.subheader("📌 Enter a sentence for sentiment analysis")

# User Input
user_text = st.text_area("Enter Text Here:")

if st.button("Analyze Sentiment"):
    if user_text:
        # Correct grammar
        corrected_text = tool.correct(user_text)
        st.write("✅ **Grammar Corrected Text:**", corrected_text)

        # Predict sentiment
        sentiment = model.predict([corrected_text])[0]

        # Display sentiment result
        if sentiment == "positive":
            st.success("😊 Positive Sentiment")
        elif sentiment == "negative":
            st.error("☹️ Negative Sentiment")
        else:
            st.warning("😐 Neutral Sentiment")
    else:
        st.warning("⚠️ Please enter some text!")

