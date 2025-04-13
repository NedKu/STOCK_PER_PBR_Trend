import streamlit as st
from _7line_webpage_PER_PBR_interactive_Web import analyze_volume_breakout, create_interactive_chart

st.set_page_config(layout="wide", page_title="Stock Analysis Tool")

st.title("📈 Interactive Stock Analysis Tool")

# Input area for stock symbols
stock_input = st.text_area(
    "Enter stock symbols (one per line):",
    "2330.TW\n6285.TW",
    help="Enter stock symbols (e.g., 2330.TW, AAPL, MSFT). One symbol per line."
)

analyze_button = st.button("Analyze Stocks")

if analyze_button:
    stocks = [s.strip() for s in stock_input.split('\n') if s.strip()]
    
    with st.spinner('Analyzing stocks...'):
        for ticker in stocks:
            st.subheader(f"Analysis for {ticker}")
            
            result = analyze_volume_breakout(ticker)
            if result:
                create_interactive_chart(result)
            else:
                st.warning(f"No significant volume breakout found for {ticker}")
