from __future__ import annotations

import pandas as pd
import plotly.graph_objs as go


def create_price_chart(ticker: str, df_plot: pd.DataFrame, mas: list[int], recent_volumes: list[float]) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df_plot.index,
            open=df_plot["Open"],
            high=df_plot["High"],
            low=df_plot["Low"],
            close=df_plot["Close"],
            name="Price",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot["MA_3.5Y"],
            name="MA 3.5Y",
            line=dict(color="green", width=2),
        )
    )

    for ma in mas:
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot[f"MA{ma}"],
                name=f"MA{ma}",
                line=dict(width=1),
            )
        )

    fig.add_trace(
        go.Bar(
            x=df_plot.index,
            y=df_plot["Volume"],
            name="Volume",
            yaxis="y2",
            marker_color="rgba(30, 30, 30, 0.8)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot["MA_Volume_3M"],
            name="3M MA Volume",
            yaxis="y2",
            line=dict(color="green", width=2),
        )
    )

    # High volume markers
    for vol in recent_volumes:
        vol_index = df_plot[df_plot["Volume"] == vol].index
        if not vol_index.empty:
            fig.add_trace(
                go.Scatter(
                    x=vol_index,
                    y=df_plot.loc[vol_index, "High"],
                    mode="markers",
                    marker=dict(color="green", size=10, symbol="star"),
                    name="High Volume Day",
                )
            )

    fig.update_layout(
        height=800,
        title=f"{ticker} Stock Analysis",
        xaxis_rangeslider_visible=False,
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Volume", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_ci_chart(ticker: str, df_weekly_plot: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_weekly_plot.index, y=df_weekly_plot["Close"], name="Close Price"))
    fig.add_trace(
        go.Scatter(
            x=df_weekly_plot.index,
            y=df_weekly_plot["regression"],
            name="3.5Y Regression",
            line=dict(color="green", width=2),
        )
    )

    for q in [0.975, 0.875, 0.125, 0.025, 0.002, 0.999]:
        fig.add_trace(
            go.Scatter(
                x=df_weekly_plot.index,
                y=df_weekly_plot[f"CI_{q}"],
                name=f"CI {q:.3f}",
            )
        )

    fig.update_layout(
        title=f"{ticker} with Confidence Intervals (Weekly)",
        yaxis_title="Price",
        xaxis_title="Date",
        hovermode="x",
    )

    return fig


def create_ratio_chart(ticker: str, ratio: str, weekly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=weekly.index, y=weekly[ratio], name=ratio, line=dict(color="blue")))
    fig.add_trace(
        go.Scatter(
            x=weekly.index,
            y=weekly[f"{ratio}_regression"],
            name=f"{ratio} Regression",
            line=dict(color="green", width=2),
        )
    )

    qs = [0.975, 0.875, 0.125, 0.025, 0.002, 0.999]
    colors = ["rgba(255,170,0,0.2)", "rgba(255,170,0,0.1)", "rgba(255,170,0,0.05)"]

    for i, (upper, lower) in enumerate(zip(qs[:3], qs[3:])):
        fig.add_trace(
            go.Scatter(
                x=weekly.index,
                y=weekly[f"{ratio}_CI_{upper}"],
                line=dict(color="orange", dash="dash"),
                name=f"+{i + 1}σ",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=weekly.index,
                y=weekly[f"{ratio}_CI_{lower}"],
                line=dict(color="orange", dash="dash"),
                name=f"-{i + 1}σ",
                fill="tonexty",
                fillcolor=colors[i],
                showlegend=True,
            )
        )

    fig.update_layout(
        title=f"{ticker} {ratio} Statistical Analysis",
        yaxis_title=ratio,
        xaxis_title="Date",
        showlegend=True,
        hovermode="x",
        height=600,
        template="plotly_white",
    )

    return fig
