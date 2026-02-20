import streamlit as st

from stock_per_pbr_trend.app_streamlit import run


def main() -> None:
    st.set_page_config(layout="wide", page_title="Stock PER/PBR Trend")
    run()


if __name__ == "__main__":
    main()
