import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import time
from scipy.stats import norm
try:
    import streamlit as st
except ImportError:
    pass


def analyze_volume_breakout(ticker, period='5y', avg_vol_period=90, std_dev_period=252*3.5, high_volume_threshold=1.4):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        min_data_points = max(std_dev_period + avg_vol_period + 30, 500)
        if len(hist) < min_data_points:
            print(f"Warning: Insufficient data ({len(hist)}) for {ticker}. Minimum required: {min_data_points}. Skipping.")
            return None

        hist['MA_3.5Y'] = hist['Close'].rolling(window=int(std_dev_period), min_periods=int(std_dev_period)).mean()
        hist['MA_Volume_3M'] = hist['Volume'].rolling(window=avg_vol_period, min_periods=avg_vol_period).mean()

        rolling_std = hist['Close'].rolling(window=int(std_dev_period), min_periods=int(std_dev_period)).std()

        recent_data = hist.iloc[-30:]
        recent_high = recent_data['High'].max()
        recent_low = recent_data['Low'].min()

        avg_3m_vol = hist['MA_Volume_3M'].iloc[-30:].mean()
        high_volume_days = recent_data[recent_data['Volume'] > avg_3m_vol * high_volume_threshold]

        if len(high_volume_days) > 0:
            return {
                'ticker': ticker,
                'recent_high': recent_high,
                'recent_low': recent_low,
                'avg_volume_3m': hist['MA_Volume_3M'].iloc[-1],
                'recent_volumes': high_volume_days['Volume'].tolist(),
                'data': hist
            }
        else:
            return None

    except yf.exceptions.YFinanceError as e:
        print(f"YFinance Error processing {ticker}: {e}")
        time.sleep(2)
        return None
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        time.sleep(2)
        return None



def calculate_confidence_intervals(data, window_size=182):
    data = data.resample('W').last()
    data['Close'] = data['Close'].ffill()  # Changed from fillna(method='ffill')

    def linear_regression(x, y):
        if len(x) < 2:
            return np.nan, np.nan
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return m, c

    data['slope'] = np.nan
    data['intercept'] = np.nan
    data['regression'] = np.nan
    data['difference'] = np.nan

    for i in range(len(data)):
        if i < window_size - 1:
            y = data['Close'][:i + 1]
            x = np.arange(len(y)) + 1
        else:
            y = data['Close'][i - window_size + 1:i + 1]
            x = np.arange(len(y)) + 1

        slope, intercept = linear_regression(x, y)
        data.loc[data.index[i], 'slope'] = slope
        data.loc[data.index[i], 'intercept'] = intercept

        if i < window_size - 1:
            regression = slope * (i + 1) + intercept
        else:
            regression = slope * window_size + intercept

        data.loc[data.index[i], 'regression'] = regression
        data.loc[data.index[i], 'difference'] = data.loc[data.index[i], 'Close'] - regression

    data['rolling_std_diff'] = data['difference'].rolling(window=window_size, min_periods=1).std()

    x_values = [0.975, 0.875, 0.125, 0.025, 0.002, 0.999]
    for x in x_values:
        data[f'CI_{x}'] = norm.ppf(x, loc=data['regression'], scale=data['rolling_std_diff'])
    return data



def create_interactive_chart(stock):
    df = stock['data'].copy()

    # Prepare weekly data for CI chart
    df_weekly = df.resample('W').last()
    df_weekly['Close'] = df_weekly['Close'].ffill()  # Changed from fillna(method='ffill')
    df_weekly = calculate_confidence_intervals(df_weekly)

    df['MA_3.5Y'] = df['Close'].rolling(window=182, min_periods=182).mean()
    mas = [5, 10, 20, 50, 150, 200]
    for ma in mas:
        df[f'MA{ma}'] = df['Close'].rolling(window=ma, min_periods=ma).mean()


    df_plot = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA_3.5Y', 'MA_Volume_3M'] + [f'MA{ma}' for ma in mas]].tail(252 * 5)
    df_weekly_plot = df_weekly[['Close', 'regression'] + [f'CI_{x}' for x in [0.975, 0.875, 0.125, 0.025, 0.002, 0.999]]].tail(252 * 5 // 5)


    # Main Chart (Daily Data)
    fig_main = go.Figure()

    fig_main.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'],
                                     low=df_plot['Low'], close=df_plot['Close'], name='Stock Price'))

    fig_main.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA_3.5Y'], name='MA 3.5Y', line=dict(color='green', width=2)))
    for ma in mas:
        fig_main.add_trace(go.Scatter(x=df_plot.index, y=df_plot[f'MA{ma}'], name=f'MA{ma}', line=dict(width=1)))

    fig_main.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], name='Volume', yaxis='y2', marker_color='rgba(30, 30, 30, 0.8)'))
    fig_main.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA_Volume_3M'], name='3M MA Volume', yaxis='y2', line=dict(color='green', width=2)))


    recent_high_volume_days = stock['recent_volumes']
    for vol in recent_high_volume_days:
        vol_index = df_plot[df_plot['Volume'] == vol].index
        if not vol_index.empty:
            fig_main.add_trace(go.Scatter(x=vol_index, y=df_plot.loc[vol_index, 'High'], mode='markers',
                                         marker=dict(color='green', size=10, symbol='star'), name='High Volume Day'))

    fig_main.update_layout(height=800, title=f'{stock["ticker"]} Stock Analysis', xaxis_rangeslider_visible=False,
                          yaxis=dict(title="Stock Price"),
                          yaxis2=dict(title="Volume", overlaying='y', side='right'),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))



    # Confidence Interval Chart (Weekly Data)
    fig_ci = go.Figure()

    fig_ci.add_trace(go.Scatter(x=df_weekly_plot.index, y=df_weekly_plot['Close'], name='Close Price'))
    fig_ci.add_trace(go.Scatter(x=df_weekly_plot.index, y=df_weekly_plot['regression'], name='3.5Y Regression', line=dict(color='green', width=2)))

    x_values = [0.975, 0.875, 0.125, 0.025, 0.002, 0.999]
    for x in x_values:
        fig_ci.add_trace(go.Scatter(x=df_weekly_plot.index, y=df_weekly_plot[f'CI_{x}'], name=f'CI {x:.3f}'))

    fig_ci.update_layout(title=f'{stock["ticker"]} with Confidence Intervals (Weekly)',
                      yaxis_title='Price', xaxis_title='Date', hovermode='x')

    try:
        st.plotly_chart(fig_main, use_container_width=True)
        st.plotly_chart(fig_ci, use_container_width=True)
    except NameError:
        fig_main.show()
        fig_ci.show()

    # Add PER/PBR analysis if data available
    info = yf.Ticker(stock['ticker']).info
    trailing_eps = info.get('trailingEps')
    book_value = info.get('bookValue')
    
    if trailing_eps and book_value:
        df['PER'] = df['Close'] / trailing_eps
        df['PBR'] = df['Close'] / book_value
        
        # Prepare weekly data for ratio analysis
        df_weekly = df.resample('W').last()
        df_weekly['PER'] = df_weekly['PER'].ffill()  # Changed from fillna(method='ffill')
        df_weekly['PBR'] = df_weekly['PBR'].ffill()  # Changed from fillna(method='ffill')
        
        # Calculate confidence intervals for ratios
        df_weekly = calculate_ratio_confidence_intervals(df_weekly)
        
        # Create PER and PBR charts
        create_ratio_charts(stock['ticker'], df_weekly)

    return

def calculate_ratio_confidence_intervals(data, window_size=182):
    for ratio in ['PER', 'PBR']:
        data[f'{ratio}_slope'] = np.nan
        data[f'{ratio}_intercept'] = np.nan
        data[f'{ratio}_regression'] = np.nan
        data[f'{ratio}_difference'] = np.nan
        
        for i in range(len(data)):
            if i < window_size - 1:
                y = data[ratio][:i + 1]
                x = np.arange(len(y)) + 1
            else:
                y = data[ratio][i - window_size + 1:i + 1]
                x = np.arange(len(y)) + 1
            
            if len(x) >= 2:
                A = np.vstack([x, np.ones(len(x))]).T
                slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
                
                data.loc[data.index[i], f'{ratio}_slope'] = slope
                data.loc[data.index[i], f'{ratio}_intercept'] = intercept
                
                regression = slope * (window_size if i >= window_size - 1 else i + 1) + intercept
                data.loc[data.index[i], f'{ratio}_regression'] = regression
                data.loc[data.index[i], f'{ratio}_difference'] = data.loc[data.index[i], ratio] - regression
        
        data[f'{ratio}_rolling_std'] = data[f'{ratio}_difference'].rolling(window=window_size, min_periods=1).std()
        
        x_values = [0.975, 0.875, 0.125, 0.025]
        for x in x_values:
            data[f'{ratio}_CI_{x}'] = norm.ppf(x, loc=data[f'{ratio}_regression'], 
                                              scale=data[f'{ratio}_rolling_std'])
    return data

def create_ratio_charts(ticker, data):
    for ratio in ['PER', 'PBR']:
        fig = go.Figure()
        
        # Main ratio line
        fig.add_trace(go.Scatter(x=data.index, y=data[ratio], 
                               name=ratio, line=dict(color='blue')))
        
        # Regression line
        fig.add_trace(go.Scatter(x=data.index, y=data[f'{ratio}_regression'],
                               name=f'{ratio} Regression', line=dict(color='green', width=2)))
        
        # Confidence intervals
        colors = ['rgba(255,170,0,0.2)', 'rgba(255,170,0,0.1)']
        x_values = [0.975, 0.875, 0.125, 0.025]
        
        for i, (upper, lower) in enumerate(zip(x_values[:2], x_values[2:])):
            fig.add_trace(go.Scatter(x=data.index, y=data[f'{ratio}_CI_{upper}'],
                                   line=dict(color='orange', dash='dash'), name=f'+{i+1}σ'))
            fig.add_trace(go.Scatter(x=data.index, y=data[f'{ratio}_CI_{lower}'],
                                   line=dict(color='orange', dash='dash'), name=f'-{i+1}σ',
                                   fill='tonexty', fillcolor=colors[i]))

        fig.update_layout(
            title=f'{ticker} {ratio} Statistical Analysis',
            yaxis_title=ratio,
            xaxis_title='Date',
            showlegend=True,
            hovermode='x',
            height=600,
            template='plotly_white'
        )
        
        try:
            st.plotly_chart(fig, use_container_width=True)
        except NameError:
            fig.show()

def main():
    print("This script is meant to be imported by the Streamlit app.")
    print("Please run stock_analyzer_app.py instead.")

if __name__ == "__main__":
    main()