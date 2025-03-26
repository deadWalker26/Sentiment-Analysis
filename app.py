import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    compound_score = sentiment_scores['compound']

    sentiment = (
        "Positive 😊" if compound_score > 0.05 
        else "Negative 💀" if compound_score < -0.05 
        else "Neutral 😶"
    )

    subjectivity_score = TextBlob(text).sentiment.subjectivity
    return compound_score, sentiment, subjectivity_score

# Streamlit app layout
st.title("📃 Sentiment Analysis App")
st.write("🔍 Enter text below and analyze its sentiment:")

# Text input
raw_text = st.text_area("Input text here")

# Analyze button
if st.button("Analyze"):
    if raw_text:
        sentiment_score, sentiment, subjectivity_score = analyze_sentiment(raw_text)

        # Create a detailed output
        output = f"""
        ## **📊 Sentiment Analysis Result**
        - **Sentiment:** {sentiment}
        - **Sentiment Score (VADER):** {sentiment_score:.2f}
        - **Subjectivity Score (TextBlob):** {subjectivity_score:.2f}
        
        ### **Explanation:**
        - **Sentiment Score:** The score ranges from **-1 to 1**:
            - **1** = Strong **positive** sentiment.
            - **-1** = Strong **negative** sentiment.
            - **0** = **Neutral** sentiment.
        
        - **Subjectivity Score:** 
            - Closer to **1** = More **subjective** (personal opinions, feelings).
            - Closer to **0** = More **objective** (factual statements).

        ### **Summary:**
        The analyzed text has a **{sentiment.lower()}** sentiment with a score of **{sentiment_score:.2f}**.
        It is classified as **{"more subjective" if subjectivity_score > 0.5 else "more objective"}**.
        """
        
        st.markdown(output)

        # Show sentiment image based on sentiment_score
        if sentiment_score > 0:
            st.image("Positive_sentiment.jpg")
        elif sentiment_score == 0:
            st.image("Neutral_sentiment.jpg")
        else:
            st.image("Negative_sentiment.jpg")

    else:
        st.warning("⚠️ Please enter some text for analysis.")
