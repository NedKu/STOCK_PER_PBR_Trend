import streamlit as st
from stock_analyzer import analyze_volume_breakout, create_interactive_chart
st.set_page_config(layout="wide", page_title="Stock Analysis Tool")
st.title("📈 Interactive Stock Analysis Tool")
stock_input = st.text_area(
    "Enter stock symbols (one per line):",
    "2330.TW\n6285.TW",
    help="Enter stock symbols (e.g., 2330.TW, AAPL, MSFT). One symbol per line."
)
if st.button("Analyze Stocks"):
    stocks = [s.strip() for s in stock_input.split('\n') if s.strip()]
    with st.spinner('Analyzing stocks...'):
        for ticker in stocks:
            st.subheader(f"Analysis for {ticker}")
            result = analyze_volume_breakout(ticker)
            if result is not None:
                create_interactive_chart(result)
            else:
                st.warning(f"No significant volume breakout found for {ticker}")
