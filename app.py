import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import language_tool_python
from language_tool_python.utils import RateLimitError
from langdetect import detect, DetectorFactory
from textblob import TextBlob
import re

DetectorFactory.seed = 0

# Supported grammar check languages by LanguageTool (not exhaustive)
SUPPORTED_LANGUAGES = {
    'en': 'en-US',
    'es': 'es',
    'fr': 'fr',
    'de': 'de',
    'hi': 'hi',
}

# Initialize VADER (only for English)
vader = SentimentIntensityAnalyzer()

# Detect language code
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'

# Create grammar tool based on language
def get_grammar_tool(lang_code):
    if lang_code not in SUPPORTED_LANGUAGES:
        return None
    try:
        return language_tool_python.LanguageToolPublicAPI(SUPPORTED_LANGUAGES[lang_code])
    except RateLimitError:
        st.warning("⚠️ Rate limit hit for grammar check. Skipping...")
        return None
    except Exception as e:
        st.error(f"Grammar tool error: {str(e)}")
        return None

# Grammar check
def check_grammar(tool, text):
    if not tool:
        return []
    try:
        return tool.check(text)
    except RateLimitError:
        st.warning("Grammar rate limit reached.")
        return []

# Sentiment (English = VADER, else TextBlob)
def analyze_sentiment(text, lang_code):
    if lang_code == 'en':
        return vader.polarity_scores(text)
    else:
        try:
            blob = TextBlob(text)
            return {"polarity": blob.sentiment.polarity}
        except:
            return {"polarity": 0}

# Text validation
def is_nonsensical(text):
    return len(text) < 5 or not re.search(r'[a-zA-Z]', text)

# UI
st.title("🌍 Multilingual Sentiment & Grammar Analyzer")
user_input = st.text_area("✍️ Enter your text (English, Spanish, French, Hinglish...)")

if st.button("Analyze"):
    if user_input:
        if is_nonsensical(user_input):
            st.error("Please enter meaningful text.")
        else:
            lang = detect_language(user_input)
            st.write(f"🌐 Detected Language: `{lang}`")

            # Grammar checking
            grammar_tool = get_grammar_tool(lang)
            issues = check_grammar(grammar_tool, user_input)
            if issues:
                st.warning("✏️ Grammar Issues:")
                for issue in issues:
                    st.write(f"- {issue.message} (Suggestion: {issue.replacements})")
            else:
                st.success("✅ No grammar issues found.")

            # Sentiment
            sentiment = analyze_sentiment(user_input, lang)
            st.subheader("🧠 Sentiment Analysis")
            if lang == 'en':
                st.write(f"😊 Positive: {sentiment['pos']*100:.2f}%")
                st.write(f"💀 Negative: {sentiment['neg']*100:.2f}%")
                st.write(f"🤐 Neutral: {sentiment['neu']*100:.2f}%")
                st.write(f"🧪 Compound: {sentiment['compound']:.2f}")
            else:
                score = sentiment['polarity']
                label = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
                st.write(f"{label} ({score:.2f})")
    else:
        st.warning("Input something to analyze. 🙄")

