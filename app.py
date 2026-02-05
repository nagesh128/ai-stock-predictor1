import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide"
)

nltk.download('vader_lexicon')

NEWS_API_KEY = "YOUR_NEWS_API_KEY"

# ---------------- CUSTOM UI STYLE ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #eef2f3, #8e9eab);
}
[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}
.stButton>button {
    background: linear-gradient(to right, #11998e, #38ef7d);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("📈 AI Powered Stock Price Predictor")
st.markdown("### Smart Prediction using Machine Learning + News Sentiment Analysis")
st.write("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙ Stock Settings")

market = st.sidebar.selectbox(
    "Select Market",
    ["India 🇮🇳", "USA 🇺🇸"]
)

stock_input = st.sidebar.text_input("Enter Stock Symbol", "RELIANCE")

predict_btn = st.sidebar.button("🚀 Predict Now")

# ---------------- NEWS FUNCTION ----------------
def get_news_sentiment(stock_name):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": stock_name,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
    except:
        return 0

    if data.get("status") != "ok":
        return 0

    articles = data.get("articles", [])[:10]
    if not articles:
        return 0

    analyzer = SentimentIntensityAnalyzer()
    sentiments = []

    for article in articles:
        title = article.get("title")
        if title:
            score = analyzer.polarity_scores(title)
            sentiments.append(score["compound"])

    if not sentiments:
        return 0

    return sum(sentiments) / len(sentiments)

# ---------------- PREDICTION SECTION ----------------
if predict_btn:

    stock_input = stock_input.strip().upper()

    if market == "India 🇮🇳":
        stock = stock_input + ".NS"
    else:
        stock = stock_input

    company_name = stock_input

    with st.spinner("Fetching News Sentiment & Training Model..."):

        news_sentiment = float(get_news_sentiment(company_name))

        ticker = yf.Ticker(stock)
        df = ticker.history(period="1y")

        if df.empty:
            st.error("Invalid stock symbol or no data found.")
            st.stop()

        df["Sentiment"] = news_sentiment
        df["Return"] = df["Close"].pct_change()
        df["MA_5"] = df["Close"].rolling(5).mean()
        df["Prev_Return"] = df["Return"].shift(1)

        df = df.dropna()

        if len(df) < 10:
            st.error("Not enough data to train model.")
            st.stop()

        y = df["Return"]
        X = df[["MA_5", "Prev_Return", "Sentiment"]]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        latest_data = df.iloc[-1][["MA_5", "Prev_Return", "Sentiment"]].values.reshape(1, -1)
        prediction = model.predict(latest_data)[0]

        last_price = df["Close"].iloc[-1]
        predicted_price = float(last_price * (1 + prediction))

    # ---------------- METRICS ----------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📰 News Sentiment", round(news_sentiment, 4))

    with col2:
        st.metric("📈 Predicted Return", f"{prediction:.5f}")

    with col3:
        st.metric("💰 Predicted Price", f"₹{predicted_price:.2f}")

    if news_sentiment > 0.2:
        st.success("🟢 Market Sentiment: Positive")
    elif news_sentiment < -0.2:
        st.error("🔴 Market Sentiment: Negative")
    else:
        st.warning("🟡 Market Sentiment: Neutral")

    st.write("---")
    st.subheader("📉 Last 1 Year Closing Price")
    st.line_chart(df["Close"])