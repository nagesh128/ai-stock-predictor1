import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Analysis of News Articles To Predict Stock Prices",
    page_icon="📈",
    layout="wide"
)

nltk.download('vader_lexicon')

NEWS_API_KEY = "b418468058fe451db789c005ebd1d12c"

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

# -------- STOCK DROPDOWN --------
if market == "India 🇮🇳":
    stock_input = st.sidebar.selectbox(
        "Select Stock",
        [
            "RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK",
            "SBIN","ITC","LT","AXISBANK","KOTAKBANK"
        ]
    )
else:
    stock_input = st.sidebar.selectbox(
        "Select Stock",
        [
            "AAPL","TSLA","GOOGL","MSFT","AMZN","META","NVDA"
        ]
    )

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
        currency = "₹"
    else:
        stock = stock_input
        currency = "$"

    with st.spinner("Fetching Data & Training Model..."):

        news_sentiment = float(get_news_sentiment(stock_input))

        ticker = yf.Ticker(stock)

        df = ticker.history(period="1y")

        if df.empty:
            st.error("Invalid stock symbol or no data found.")
            st.stop()

        df.reset_index(inplace=True)

        # -------- Feature Engineering --------
        df["Sentiment"] = news_sentiment
        df["Return"] = df["Close"].pct_change()
        df["MA_5"] = df["Close"].rolling(5).mean()
        df["Prev_Return"] = df["Return"].shift(1)

        df.dropna(inplace=True)

        y = df["Return"]

        X = df[["MA_5","Prev_Return","Sentiment"]]

        X_train, X_test, y_train, y_test = train_test_split(
            X,y,test_size=0.2,shuffle=False
        )

        model = RandomForestRegressor(n_estimators=100,random_state=42)

        model.fit(X_train,y_train)

        # -------- Predictions --------
        y_pred = model.predict(X_test)

        actual_price = df["Close"].iloc[-len(y_test):].values

        predicted_price_series = actual_price * (1 + y_pred)

        latest_data = df.iloc[-1][["MA_5","Prev_Return","Sentiment"]].values.reshape(1,-1)

        prediction = model.predict(latest_data)[0]

        last_price = df["Close"].iloc[-1]

        predicted_price = float(last_price * (1 + prediction))

        # -------- Metrics --------
        r2 = r2_score(y_test,y_pred)

        mse = mean_squared_error(y_test,y_pred)

    # ---------------- METRICS ----------------
    col1,col2,col3 = st.columns(3)

    col1.metric("📰 News Sentiment",round(news_sentiment,4))

    col2.metric("📈 Predicted Return",f"{prediction:.5f}")

    col3.metric("💰 Predicted Price",f"{currency}{predicted_price:.2f}")

    st.write(f"**R² Score:** {r2:.4f}")

    st.write(f"**Mean Squared Error:** {mse:.6f}")

    st.write("---")

    # -------- Original vs Predicted --------
    st.subheader("📊 Original vs Predicted Price")

    fig_compare,ax_compare = plt.subplots(figsize=(12,6))

    ax_compare.plot(actual_price,label="Original Price")

    ax_compare.plot(predicted_price_series,label="Predicted Price")

    ax_compare.legend()

    ax_compare.grid(True)

    st.pyplot(fig_compare)

    # -------- Candlestick Chart --------
    st.write("---")

    st.subheader("📊 Real Stock Price (Candlestick Chart)")

    fig = go.Figure(data=[go.Candlestick(

        x=df["Date"],

        open=df["Open"],

        high=df["High"],

        low=df["Low"],

        close=df["Close"]

    )])

    st.plotly_chart(fig,use_container_width=True)

    # -------- Closing Price --------
    st.write("---")

    st.subheader("📈 Closing Price Over Time")

    st.line_chart(df.set_index("Date")["Close"])

    # -------- Opening Price --------
    st.write("---")

    st.subheader("📈 Opening Price Over Time")

    st.line_chart(df.set_index("Date")["Open"])

    # -------- High Price --------
    st.write("---")

    st.subheader("📈 High Price Over Time")

    st.line_chart(df.set_index("Date")["High"])

    # -------- Volume --------
    st.write("---")

    st.subheader("📊 Volume Over Time")

    st.bar_chart(df.set_index("Date")["Volume"])

