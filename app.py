from flask import Flask, render_template, request
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.simplefilter("ignore", ValueWarning)
warnings.simplefilter("ignore", ConvergenceWarning)

app = Flask(__name__)

# -------------------------
# Stock master list
# -------------------------
STOCK_OPTIONS = [
    {"symbol": "AAPL", "name": "Apple"},
    {"symbol": "MSFT", "name": "Microsoft"},
    {"symbol": "GOOGL", "name": "Alphabet"},
    {"symbol": "AMZN", "name": "Amazon"},
    {"symbol": "TSLA", "name": "Tesla"},
    {"symbol": "META", "name": "Meta"},
    {"symbol": "NVDA", "name": "NVIDIA"},
    {"symbol": "NFLX", "name": "Netflix"},
    {"symbol": "AMD", "name": "AMD"},
    {"symbol": "INTC", "name": "Intel"},
    {"symbol": "JPM", "name": "JPMorgan Chase"},
    {"symbol": "BAC", "name": "Bank of America"},
    {"symbol": "WMT", "name": "Walmart"},
    {"symbol": "DIS", "name": "Disney"},
    {"symbol": "KO", "name": "Coca-Cola"},
    {"symbol": "PEP", "name": "PepsiCo"},
    {"symbol": "NKE", "name": "Nike"},
    {"symbol": "TCS.NS", "name": "TCS"},
    {"symbol": "INFY.NS", "name": "Infosys"},
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank"},
    {"symbol": "SBIN.NS", "name": "State Bank of India"},
    {"symbol": "TATAMOTORS.NS", "name": "Tata Motors"},
    {"symbol": "LT.NS", "name": "Larsen & Toubro"},
]


# -------------------------
# Utility functions
# -------------------------
def adf_test(series):
    series = pd.Series(series).dropna()
    result = adfuller(series)
    return {
        "adf_stat": round(result[0], 4),
        "p_value": round(result[1], 6),
        "stationary": "Yes" if result[1] < 0.05 else "No"
    }


def evaluate_forecast(actual, predicted):
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    return round(rmse, 4), round(mae, 4)


def find_best_arima_order(train):
    """
    Manual ARIMA order selection using AIC
    """
    best_aic = float("inf")
    best_order = None

    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    model = ARIMA(
                        train,
                        order=(p, d, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except Exception:
                    continue

    if best_order is None:
        best_order = (1, 1, 1)
        best_aic = None

    return best_order, best_aic


def build_candlestick_chart(df, symbol):
    """
    TradingView-style candlestick chart with MA50 + MA200
    """
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candlestick"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MA50"],
            mode="lines",
            name="MA50",
            line=dict(width=2)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MA200"],
            mode="lines",
            name="MA200",
            line=dict(width=2)
        )
    )

    fig.update_layout(
        title=f"{symbol} Price Chart",
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Price",
        height=620,
        xaxis_rangeslider_visible=True,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def build_forecast_chart(df_close, future_mean, future_ci, symbol, forecast_days):
    """
    Forecast chart: historical close + future forecast + confidence interval
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_close.index,
            y=df_close.values,
            mode="lines",
            name="Historical Close",
            line=dict(width=2)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future_mean.index,
            y=future_mean.values,
            mode="lines",
            name=f"{forecast_days}-Day Forecast",
            line=dict(width=3)
        )
    )

    # upper band
    fig.add_trace(
        go.Scatter(
            x=future_ci.index,
            y=future_ci.iloc[:, 1],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        )
    )

    # lower band with fill
    fig.add_trace(
        go.Scatter(
            x=future_ci.index,
            y=future_ci.iloc[:, 0],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name="Confidence Interval",
            hoverinfo="skip"
        )
    )

    fig.update_layout(
        title=f"{symbol} Forecast Chart",
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Price",
        height=520,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")

    return fig.to_html(full_html=False, include_plotlyjs=False)


def resolve_symbol(form_symbol, form_search):
    """
    Use dropdown symbol if selected; otherwise try search text.
    Search supports ticker or company name.
    """
    if form_symbol and form_symbol.strip():
        return form_symbol.strip().upper()

    if not form_search:
        return "AAPL"

    query = form_search.strip().lower()

    # exact symbol match
    for stock in STOCK_OPTIONS:
        if stock["symbol"].lower() == query:
            return stock["symbol"]

    # exact company name match
    for stock in STOCK_OPTIONS:
        if stock["name"].lower() == query:
            return stock["symbol"]

    # partial match
    for stock in STOCK_OPTIONS:
        if query in stock["symbol"].lower() or query in stock["name"].lower():
            return stock["symbol"]

    # fallback: allow user-entered ticker
    return form_search.strip().upper()


# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", stock_options=STOCK_OPTIONS)


@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        selected_symbol = request.form.get("symbol", "")
        search_text = request.form.get("stock_search", "")
        forecast_days = int(request.form.get("forecast_days", 30))

        symbol = resolve_symbol(selected_symbol, search_text)

        # -------------------------
        # Download stock data
        # -------------------------
        raw = yf.download(symbol, period="5y", auto_adjust=False)

        if raw.empty:
            return render_template(
                "index.html",
                stock_options=STOCK_OPTIONS,
                error=f"No data found for {symbol}. Try another stock symbol."
            )

        # Flatten MultiIndex columns if yfinance returns them
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        required_cols = ["Open", "High", "Low", "Close"]
        for col in required_cols:
            if col not in raw.columns:
                return render_template(
                    "index.html",
                    stock_options=STOCK_OPTIONS,
                    error=f"Downloaded data for {symbol} is missing required column: {col}"
                )

        if "Volume" not in raw.columns:
            raw["Volume"] = 0

        # Keep OHLCV for charting + Close for modeling
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

        # Convert everything to numeric
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows where OHLC are missing
        df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

        # Datetime index + business frequency
        df.index = pd.to_datetime(df.index)
        df = df.asfreq("B")

        # Forward fill price columns for business-day continuity
        df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].ffill()
        df["Volume"] = df["Volume"].fillna(0)

        # Moving averages
        df["MA50"] = df["Close"].rolling(50).mean()
        df["MA200"] = df["Close"].rolling(200).mean()

        # -------------------------
        # Train / Test split
        # -------------------------
        close_series = df["Close"].copy()
        train_size = int(len(close_series) * 0.8)
        train = close_series[:train_size]
        test = close_series[train_size:]

        # ADF tests
        adf_original = adf_test(train)
        train_diff = train.diff().dropna()
        adf_diff = adf_test(train_diff)

        # -------------------------
        # ARIMA
        # -------------------------
        best_order, best_aic = find_best_arima_order(train)

        arima_model = ARIMA(
            train,
            order=best_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        arima_result = arima_model.fit()

        # Forecast test period
        test_forecast = arima_result.get_forecast(steps=len(test))
        forecast_mean = pd.Series(test_forecast.predicted_mean, index=test.index)

        forecast_ci = test_forecast.conf_int()
        forecast_ci.index = test.index

        # Benchmarks
        naive_forecast = np.repeat(train.iloc[-1], len(test))
        rolling_mean_value = train.tail(30).mean()
        rolling_mean_forecast = np.repeat(rolling_mean_value, len(test))

        arima_rmse, arima_mae = evaluate_forecast(test, forecast_mean)
        naive_rmse, naive_mae = evaluate_forecast(test, naive_forecast)
        rolling_rmse, rolling_mae = evaluate_forecast(test, rolling_mean_forecast)

        # -------------------------
        # Future Forecast
        # -------------------------
        future_forecast = arima_result.get_forecast(steps=forecast_days)
        future_mean = pd.Series(future_forecast.predicted_mean)

        future_ci = future_forecast.conf_int()

        last_date = df.index[-1]
        future_index = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_days
        )

        future_mean.index = future_index
        future_ci.index = future_index

        # -------------------------
        # SARIMA comparison
        # -------------------------
        try:
            sarima_model = SARIMAX(
                train,
                order=best_order,
                seasonal_order=(1, 0, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            sarima_result = sarima_model.fit(disp=False)
            sarima_aic = round(sarima_result.aic, 4)
        except Exception:
            sarima_aic = "Not Available"

        # -------------------------
        # Forecast table
        # -------------------------
        forecast_table = pd.DataFrame({
            "Date": future_mean.index.strftime("%Y-%m-%d"),
            "Forecast": np.round(future_mean.values, 2),
            "Lower CI": np.round(future_ci.iloc[:, 0].values, 2),
            "Upper CI": np.round(future_ci.iloc[:, 1].values, 2)
        })

        # -------------------------
        # Summary cards
        # -------------------------
        latest_close = round(float(df["Close"].dropna().iloc[-1]), 2)
        latest_open = round(float(df["Open"].dropna().iloc[-1]), 2)
        latest_high = round(float(df["High"].dropna().iloc[-1]), 2)
        latest_low = round(float(df["Low"].dropna().iloc[-1]), 2)

        # -------------------------
        # Charts
        # -------------------------
        candlestick_chart = build_candlestick_chart(df, symbol)
        forecast_chart = build_forecast_chart(
            df["Close"], future_mean, future_ci, symbol, forecast_days
        )

        # -------------------------
        # Results dictionary
        # -------------------------
        results = {
            "symbol": symbol,
            "forecast_days": forecast_days,
            "rows": len(df),
            "latest_close": latest_close,
            "latest_open": latest_open,
            "latest_high": latest_high,
            "latest_low": latest_low,
            "best_order": best_order,
            "best_aic": round(best_aic, 4) if best_aic is not None else "Not Available",
            "sarima_aic": sarima_aic,
            "adf_original": adf_original,
            "adf_diff": adf_diff,
            "arima_rmse": arima_rmse,
            "arima_mae": arima_mae,
            "naive_rmse": naive_rmse,
            "naive_mae": naive_mae,
            "rolling_rmse": rolling_rmse,
            "rolling_mae": rolling_mae,
            "forecast_table": forecast_table.to_dict(orient="records"),
            "candlestick_chart": candlestick_chart,
            "forecast_chart": forecast_chart,
        }
        print("RESULTS KEYS:", results.keys())
        print("SYMBOL:", results["symbol"])
        print("FORECAST TABLE LENGTH:", len(results["forecast_table"]))

        return render_template(
            "result.html",
            results=results,
            stock_options=STOCK_OPTIONS
        )

    except Exception as e:
        return render_template(
            "index.html",
            stock_options=STOCK_OPTIONS,
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)