import streamlit as st
import yfinance as yf
from GoogleNews import GoogleNews
import torch
from transformers import pipeline
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Senti-Bridge: Market Truth Engine", layout="wide")

# --- 1. THE "EYES": DATA FETCHING FUNCTIONS ---

def get_stock_data(ticker_symbol):
    """Fetches live price and % change from yfinance"""
    stock = yf.Ticker(ticker_symbol)
    history = stock.history(period="1d")
    
    if history.empty:
        return None, 0.0
    
    current_price = history['Close'].iloc[-1]
    # Calculate simple % change from open
    open_price = history['Open'].iloc[-1]
    change_percent = ((current_price - open_price) / open_price) * 100
    
    return round(current_price, 2), round(change_percent, 2)

def get_news_headlines(query, max_results=5):
    """Fetches news headlines using GoogleNews"""
    googlenews = GoogleNews(lang='en', region='US') # or 'IN' for India specific
    googlenews.search(query)
    results = googlenews.result()
    # Return just the titles for analysis
    return [entry['title'] for entry in results[:max_results]]

def get_mock_social_sentiment():
    """
    Simulates Reddit sentiment for the Hackathon demo.
    (Use this if you don't have time to set up PRAW/Reddit API keys)
    """
    import random
    # In a real app, use PRAW to fetch r/wallstreetbets comments
    hype_score = random.uniform(0.3, 0.9) 
    trending_words = ["Buy the dip", "YOLO", "Rocket", "Moon", "Panic"]
    return hype_score, random.choice(trending_words)

# --- 2. THE "BRAIN": AI SENTIMENT ANALYSIS ---

@st.cache_resource # Caches the model so it doesn't reload every time
def load_sentiment_model():
    """Loads FinBERT - specialized for financial text"""
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_news_sentiment(headlines, sentiment_pipeline):
    """Passes headlines through FinBERT and returns an average score"""
    if not headlines:
        return 0, "Neutral"
    
    total_score = 0
    
    for headline in headlines:
        result = sentiment_pipeline(headline)[0]
        # Convert label to numeric score
        score = result['score']
        if result['label'] == 'negative':
            score *= -1
        elif result['label'] == 'neutral':
            score = 0
        
        total_score += score

    avg_score = total_score / len(headlines)
    
    # Classify the final average
    if avg_score > 0.2: return avg_score, "Bullish"
    if avg_score < -0.2: return avg_score, "Bearish"
    return avg_score, "Neutral"

# --- 3. THE "VOICE": LLM REASONING (Simulated for Demo) ---
def generate_ai_explanation(stock_name, price_change, sentiment_label, headlines):
    """
    Generates the 'Why'. Connect this to Gemini API/OpenAI API for the win.
    Here we use a dynamic template for the demo.
    """
    intro = f"**AI Analyst Report for {stock_name}:**\n\n"
    
    if price_change < 0 and sentiment_label == "Bullish":
        reason = "🚨 **DIVERGENCE DETECTED:** Prices are dropping despite positive news. This suggests a 'Bear Trap' or market overreaction to external noise. Institutional sentiment remains strong."
    elif price_change > 0 and sentiment_label == "Bearish":
        reason = "⚠️ **RISK ALERT:** Price is rising on weak fundamentals. This rally appears driven by retail hype rather than facts. Caution advised."
    elif price_change > 0 and sentiment_label == "Bullish":
        reason = "✅ **TREND CONFIRMED:** Price and News are in sync. The rally is supported by strong fundamental reporting."
    else:
        reason = "📉 **MARKET WEAKNESS:** Negative news is dragging the price down. The trend is confirmed by fundamentals."
        
    evidence = f"\n\n**Key Driver:** '{headlines[0] if headlines else 'Market Volatility'}'"
    return intro + reason + evidence

# --- 4. THE UI: STREAMLIT DASHBOARD ---

def main():
    st.title("🧠 Senti-Bridge: Live Market Truth Engine")
    st.markdown("### *Detecting the gap between News (Fundamentals) and Price (Reality)*")
    
    # Sidebar for Asset Selection
    with st.sidebar:
        st.header("Control Panel")
        selected_asset = st.selectbox("Select Asset", ["S&P 500", "Nifty 50", "Reliance Ind.", "Tesla"])
        
        # Map friendly names to Ticker Symbols
        tickers = {
            "S&P 500": "^GSPC",
            "Nifty 50": "^NSEI",
            "Reliance Ind.": "RELIANCE.NS",
            "Tesla": "TSLA"
        }
        symbol = tickers[selected_asset]
        
        if st.button("Run Live Analysis"):
            st.rerun()

    # Load Model (Happens once)
    sentiment_pipeline = load_sentiment_model()

    # Main Dashboard Columns
    col1, col2, col3 = st.columns(3)

    # 1. Fetch Data
    price, change = get_stock_data(symbol)
    headlines = get_news_headlines(selected_asset)
    sent_score, sent_label = analyze_news_sentiment(headlines, sentiment_pipeline)
    social_score, social_word = get_mock_social_sentiment()

    # 2. Display Core Metrics
    with col1:
        st.metric(label=f"{selected_asset} Price", value=f"{price}", delta=f"{change}%")
    
    with col2:
        # Color logic for sentiment
        color = "normal"
        if sent_label == "Bullish": color = "normal" # Streamlit handles green automatically for delta
        st.metric(label="Institutional Sentiment (FinBERT)", value=f"{sent_score:.2f}", delta=sent_label)

    with col3:
        st.metric(label="Social Hype (Retail)", value=f"{social_score:.2f}", delta=social_word, delta_color="off")

    st.divider()

    # 3. The "Why" Section (The Winner)
    st.subheader("🤖 AI Explanation Engine")
    
    explanation = generate_ai_explanation(selected_asset, change, sent_label, headlines)
    st.info(explanation)

    # 4. Global Correlation Visual (The Comparison)
    st.subheader("Global Correlation: S&P 500 vs Nifty 50")
    
    # Mock data for visual - In real app, fetch history for both
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=[1, 2, 3, 2.5, 4], mode='lines', name='S&P 500 (Leading)'))
    fig.add_trace(go.Scatter(y=[0.8, 1.5, 2.8, 2.2, 3.5], mode='lines', name='Nifty 50 (Following)', line=dict(dash='dot')))
    fig.update_layout(title="Lead-Lag Analysis", template="plotly_dark", height=300)
    st.plotly_chart(fig, use_container_width=True)

    # 5. Raw Data Expander
    with st.expander("See Raw News Source Data"):
        for i, head in enumerate(headlines):
            st.write(f"{i+1}. {head}")

if __name__ == "__main__":
    main()