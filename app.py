import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import language_tool_python  # For grammar checking
import re  # For regular expressions

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Use LanguageTool API instead of the local Java server
grammar_tool = language_tool_python.LanguageToolPublicAPI('en-US')

text = "This is a example of bad grammar."
matches = grammar_tool.check(text)

for match in matches:
    print(match)


# Function to analyze sentiment using VADER
def analyze_sentiment(text):
    if not text or not isinstance(text, str):
        return None
    scores = analyzer.polarity_scores(text)
    return scores

# Function to check grammar
def check_grammar(text):
    matches = grammar_tool.check(text)
    return matches

# Function to check if the text is nonsensical
def is_nonsensical(text):
    # Check if the text contains only random characters or is too short
    if len(text) < 5 or not re.search(r'[a-zA-Z]', text):
        return True
    return False

# Streamlit UI
st.title('Sentiment Analysis App')
st.write('Enter a paragraph of text to analyze its sentiment.')

# Text input from user
user_input = st.text_area("Input Text", height=200)

# Button to submit the input
if st.button('Analyze'):
    if user_input:
        # Check if the input is nonsensical
        if is_nonsensical(user_input):
            st.error("Invalid input. Please enter a valid text.")
        else:
            # Check grammar
            grammar_issues = check_grammar(user_input)
            if grammar_issues:
                st.warning("Grammar issues detected:")
                for issue in grammar_issues:
                    st.write(f"- {issue.message} (Suggestion: {issue.replacements})")
            
            # Analyze sentiment
            sentiment_scores = analyze_sentiment(user_input)
            if sentiment_scores is None:
                st.error("Invalid input. Please enter a valid text.")
            else:
                st.success(f"Sentiment Analysis Results: {sentiment_scores}")
                st.write(f"Positive: {sentiment_scores['pos']*100:.2f}%")
                st.write(f"Negative: {sentiment_scores['neg']*100:.2f}%")
                st.write(f"Neutral: {sentiment_scores['neu']*100:.2f}%")
                st.write(f"Overall Sentiment Score: {sentiment_scores['compound']:.2f}")
    else:
        st.warning("Please enter some text to analyze.")
