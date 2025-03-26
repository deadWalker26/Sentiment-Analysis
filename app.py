import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER
sia = SentimentIntensityAnalyzer()


# Function to analyze sentiment
def analyze_sentiment(text):
    analysis = sia.polarity_scores(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

# Streamlit app layout
st.title("Sentiment Analysis App🙊")
st.write("Enter text to analyze its sentiment📃:")

# Create a session state variable to track typing
if 'is_typing' not in st.session_state:
    st.session_state.is_typing = False

# Text input
raw_text = st.text_area("Input text here", on_change=lambda: setattr(st.session_state, 'is_typing', True))

# Show typing indicator
if st.session_state.is_typing:
    st.markdown("Loading...")

# Analyze button
if st.button("Analyze"):
    if raw_text:
        sentiment_score, subjectivity_score = analyze_sentiment(raw_text)
        sentiment = "Positive😊" if sentiment_score > 0 else "Negative💀" if sentiment_score < 0 else "Neutral😶"
        
        # Create a detailed output
        output = f"""
        **Sentiment Analysis Result:**
        - **Sentiment:** {sentiment}
        - **Sentiment Score:** {sentiment_score:.2f}
        - **Subjectivity Score:** {subjectivity_score:.2f}
        
        ### Explanation:
        - **Sentiment Score:** The sentiment score ranges from -1 to 1.
            - A score closer to **1** indicates a **positive sentiment** (e.g., "Ashutosh is Legit!").
            - A score closer to **-1** indicates a **negative sentiment** (e.g., "I hate this world!").
            - A score around **0** indicates a **neutral sentiment** (e.g., "This College is okay.").
        
        - **Subjectivity Score:** The subjectivity score ranges from 0 to 1.
            - A score closer to **1** indicates that the text is **more subjective** (e.g., "I think this is the best movie ever!").
            - A score closer to **0** indicates that the text is **more objective** (e.g., "The movie was released in 2020.").
        
        ### Summary:
        Based on the analysis of the text you provided, the sentiment is classified as **{sentiment}**. 
        The sentiment score of **{sentiment_score:.2f}** indicates that the text leans towards being **{sentiment.lower()}**. 
        This suggests that the emotions conveyed in the text are predominantly **{sentiment.lower()}** in nature. 
        The subjectivity score of **{subjectivity_score:.2f}** indicates that the text is **{"more subjective" if subjectivity_score > 0.5 else "more objective"}**. 
        This means that the text contains personal opinions or feelings rather than just factual information based on the above paragraph. 😜
        """
        
        st.markdown(output)

        # Show sentiment image based on sentiment_score
        if sentiment_score > 0:
            st.image("Positive_sentiment.jpg")
        elif sentiment_score == 0:
            st.image("Neutral_sentiment.jpg")
        else:
            st.image("Negative_sentiment.jpg")
        
        st.session_state.is_typing = False
    else:
        st.warning("Please enter some text for analysis.")
