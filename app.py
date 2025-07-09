import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import language_tool_python
from language_tool_python.utils import RateLimitError
from langdetect import detect, DetectorFactory
import re

DetectorFactory.seed = 0

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Try to initialize grammar tool
try:
    grammar_tool = language_tool_python.LanguageToolPublicAPI('en-US')
except RateLimitError:
    grammar_tool = None
    st.warning("⚠️ Grammar check is temporarily unavailable due to API rate limits.")

# Function to analyze sentiment
def analyze_sentiment(text):
    if not text or not isinstance(text, str):
        return None
    return analyzer.polarity_scores(text)

# Function to check grammar
def check_grammar(text):
    if grammar_tool is None:
        return None  # grammar check disabled
    try:
        return grammar_tool.check(text)
    except RateLimitError:
        st.warning("⚠️ You’ve hit the grammar check rate limit. Try again later.")
        return None

# Function to validate input
def is_nonsensical(text):
    return len(text) < 5 or not re.search(r'[a-zA-Z]', text)

# Streamlit UI
st.title("🔍 Sentiment Analysis App")
st.write("Enter some text to analyze sentiment and grammar 📃")

user_input = st.text_area("Input Text", height=200)

if st.button("Analyze"):
    if user_input:
        if is_nonsensical(user_input):
            st.error("Invalid input! Please enter meaningful text. 🤔")
        else:
            # Detect language
            try:
                language = detect(user_input)
                st.write(f"Detected Language: `{language}`")
            except Exception:
                st.error("Language detection failed.")

            # Check grammar
            grammar_issues = check_grammar(user_input)
            if grammar_issues:
                if len(grammar_issues) > 0:
                    st.warning("⚠️ Grammar issues found:")
                    for issue in grammar_issues:
                        st.write(f"- {issue.message} (Suggestion: {issue.replacements})")
                else:
                    st.success("✅ No grammar issues found.")
            elif grammar_tool is None:
                st.info("Grammar checking is disabled due to API rate limits.")

            # Sentiment
            sentiment = analyze_sentiment(user_input)
            if sentiment:
                st.subheader("📊 Sentiment Results")
                st.write(f"😊 Positive: {sentiment['pos']*100:.2f}%")
                st.write(f"💀 Negative: {sentiment['neg']*100:.2f}%")
                st.write(f"🤐 Neutral: {sentiment['neu']*100:.2f}%")
                st.write(f"🧠 Compound Score: {sentiment['compound']:.2f}")
            else:
                st.error("Sentiment analysis failed.")
    else:
        st.warning("Please enter some text. 🙄")
