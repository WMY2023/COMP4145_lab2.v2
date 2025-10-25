import pandas as pd
import numpy as np
from typing import Optional


def calculate_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    return data


def identify_golden_cross(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data['Signal'] = 0
    data['GoldenCross'] = (data['MA50'] > data['MA200']) & (data['MA50'].shift(1) <= data['MA200'].shift(1))
    return data


def implement_strategy(data: pd.DataFrame, target_pct: float = 0.15, max_holding_days: int = 60, stop_loss_pct: float = 0.07) -> pd.DataFrame:
    """Golden Cross trading strategy with configurable take-profit and stop-loss.

    Parameters
    - data: DataFrame with at least 'Close' and 'GoldenCross' columns
    - target_pct: take-profit as a decimal (e.g., 0.15 for 15%)
    - max_holding_days: maximum days to hold before forced sell
    - stop_loss_pct: stop-loss as a decimal (e.g., 0.07 for 7% drop)
    """
    positions = []
    # skip the early rows where MA200 may be NaN
    data = data.copy()
    if 'GoldenCross' not in data.columns:
        data = identify_golden_cross(calculate_moving_averages(data))

    data = data.dropna(subset=['MA200']).copy()
    buy_dates = data[data['GoldenCross'] == True].index.tolist()

    for buy_date in buy_dates:
        # ensure buy_date is in index
        if buy_date not in data.index:
            continue
        buy_price = data.loc[buy_date, 'Close']
        target_price = buy_price * (1.0 + target_pct)
        stop_price = buy_price * (1.0 - stop_loss_pct)
        max_sell_date = buy_date + pd.Timedelta(days=max_holding_days)

        # consider sell window from next trading day through max_sell_date
        sell_period = data.loc[buy_date:max_sell_date].copy()
        if sell_period.empty:
            continue

        # locate all candidate events
        target_hits = sell_period[sell_period['Close'] >= target_price]
        stop_hits = sell_period[sell_period['Close'] <= stop_price]

        # choose earliest event (stop or target) if any, otherwise last available date
        candidate_events = []
        if not target_hits.empty:
            candidate_events.append((target_hits.index[0], target_hits.loc[target_hits.index[0], 'Close'], 'Target reached'))
        if not stop_hits.empty:
            candidate_events.append((stop_hits.index[0], stop_hits.loc[stop_hits.index[0], 'Close'], 'Stop loss'))

        if candidate_events:
            # pick earliest by timestamp
            candidate_events.sort(key=lambda x: x[0])
            sell_date, sell_price, sell_reason = candidate_events[0]
        else:
            # fallback: sell at last available date in sell_period
            last_date = sell_period.index[-1]
            sell_date = last_date
            sell_price = sell_period.loc[last_date, 'Close']
            sell_reason = 'Max holding period'

        holding_days = (sell_date - buy_date).days
        profit_pct = (sell_price / buy_price - 1) * 100

        positions.append({
            'BuyDate': buy_date,
            'BuyPrice': buy_price,
            'SellDate': sell_date,
            'SellPrice': sell_price,
            'HoldingDays': holding_days,
            'ProfitPct': profit_pct,
            'SellReason': sell_reason
        })

    return pd.DataFrame(positions)


def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    data = data.copy()
    data['BB_Middle'] = data['Close'].rolling(window=window).mean()
    data['BB_Std'] = data['Close'].rolling(window=window).std()
    data['BB_Upper'] = data['BB_Middle'] + num_std * data['BB_Std']
    data['BB_Lower'] = data['BB_Middle'] - num_std * data['BB_Std']
    return data


def implement_bollinger_strategy(data: pd.DataFrame, window: int = 20, num_std: int = 2, max_holding_days: int = 60) -> pd.DataFrame:
    data = data.copy()
    data = calculate_bollinger_bands(data, window=window, num_std=num_std)
    positions = []
    data = data.dropna(subset=['BB_Lower', 'BB_Middle'])
    buy_signals = (data['Close'] < data['BB_Lower']) & (data['Close'].shift(1) >= data['BB_Lower'].shift(1))
    buy_dates = data[buy_signals].index.tolist()

    for buy_date in buy_dates:
        buy_price = data.loc[buy_date, 'Close']
        max_sell_date = buy_date + pd.Timedelta(days=max_holding_days)
        sell_period = data.loc[buy_date:max_sell_date].copy()
        sell_candidates = sell_period[sell_period['Close'] >= sell_period['BB_Middle']]
        if not sell_candidates.empty:
            sell_date = sell_candidates.index[0]
            sell_price = sell_candidates.loc[sell_date, 'Close']
            sell_reason = 'BB middle reached'
        else:
            sell_dates = sell_period.index.tolist()
            if sell_dates:
                sell_date = sell_dates[-1]
                sell_price = sell_period.loc[sell_date, 'Close']
                sell_reason = 'Max holding period'
            else:
                continue

        holding_days = (sell_date - buy_date).days
        profit_pct = (sell_price / buy_price - 1) * 100

        positions.append({
            'BuyDate': buy_date,
            'BuyPrice': buy_price,
            'SellDate': sell_date,
            'SellPrice': sell_price,
            'HoldingDays': holding_days,
            'ProfitPct': profit_pct,
            'SellReason': sell_reason
        })

    return pd.DataFrame(positions)


def calculate_obv(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if 'Volume' not in data.columns:
        data['Volume'] = 0
    sign = np.sign(data['Close'].diff()).fillna(0)
    data['OBV'] = (sign * data['Volume']).cumsum()
    return data


def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    data = data.copy()
    if not all(col in data.columns for col in ['High', 'Low', 'Close']):
        data['ATR'] = np.nan
        return data
    prev_close = data['Close'].shift(1)
    tr1 = data['High'] - data['Low']
    tr2 = (data['High'] - prev_close).abs()
    tr3 = (data['Low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data['TR'] = tr
    data['ATR'] = data['TR'].rolling(window=window).mean()
    return data


def obv_strategy(data: pd.DataFrame, ma_window: int = 20, max_holding_days: int = 60) -> pd.DataFrame:
    """Generate simple OBV-based buy/sell positions.

    Buy when OBV crosses above its rolling MA, sell when OBV crosses below its MA or after max_holding_days.
    """
    df = calculate_obv(data)
    df = df.dropna(subset=['OBV'])
    df['OBV_MA'] = df['OBV'].rolling(window=ma_window).mean()

    buy_signals = (df['OBV'] > df['OBV_MA']) & (df['OBV'].shift(1) <= df['OBV_MA'].shift(1))
    buy_dates = df[buy_signals].index.tolist()

    positions = []
    for buy_date in buy_dates:
        buy_price = df.loc[buy_date, 'Close']
        max_sell_date = buy_date + pd.Timedelta(days=max_holding_days)
        sell_period = df.loc[buy_date:max_sell_date].copy()

        # Sell when OBV crosses below its MA
        sell_candidates = sell_period[(sell_period['OBV'] < sell_period['OBV_MA']) & (sell_period['OBV'].shift(1) >= sell_period['OBV_MA'].shift(1))]
        if not sell_candidates.empty:
            sell_date = sell_candidates.index[0]
            sell_price = sell_candidates.loc[sell_date, 'Close']
            sell_reason = 'OBV cross down'
        else:
            sell_dates = sell_period.index.tolist()
            if sell_dates:
                sell_date = sell_dates[-1]
                sell_price = sell_period.loc[sell_date, 'Close']
                sell_reason = 'Max holding period'
            else:
                continue

        holding_days = (sell_date - buy_date).days
        profit_pct = (sell_price / buy_price - 1) * 100
        positions.append({
            'BuyDate': buy_date,
            'BuyPrice': buy_price,
            'SellDate': sell_date,
            'SellPrice': sell_price,
            'HoldingDays': holding_days,
            'ProfitPct': profit_pct,
            'SellReason': sell_reason
        })

    return pd.DataFrame(positions)


def atr_strategy(data: pd.DataFrame, atr_window: int = 14, entry_mult: float = 1.5, exit_mult: float = 1.0, max_holding_days: int = 60) -> pd.DataFrame:
    """Simple ATR breakout strategy:

    - Buy when today's close > yesterday's close + entry_mult * ATR
    - Sell when price drops below entry_price - exit_mult * ATR or after max_holding_days
    """
    df = calculate_atr(data, window=atr_window)
    df = df.dropna(subset=['ATR'])

    # Entry rule
    entry_mask = df['Close'] > (df['Close'].shift(1) + entry_mult * df['ATR'])
    buy_dates = df[entry_mask].index.tolist()

    positions = []
    for buy_date in buy_dates:
        buy_price = df.loc[buy_date, 'Close']
        atr_at_entry = df.loc[buy_date, 'ATR']
        max_sell_date = buy_date + pd.Timedelta(days=max_holding_days)
        sell_period = df.loc[buy_date:max_sell_date].copy()

        # Exit when price drops below entry_price - exit_mult * ATR
        sell_candidates = sell_period[sell_period['Close'] <= (buy_price - exit_mult * atr_at_entry)]
        if not sell_candidates.empty:
            sell_date = sell_candidates.index[0]
            sell_price = sell_candidates.loc[sell_date, 'Close']
            sell_reason = 'Stop (ATR)'
        else:
            sell_dates = sell_period.index.tolist()
            if sell_dates:
                sell_date = sell_dates[-1]
                sell_price = sell_period.loc[sell_date, 'Close']
                sell_reason = 'Max holding period'
            else:
                continue

        holding_days = (sell_date - buy_date).days
        profit_pct = (sell_price / buy_price - 1) * 100
        positions.append({
            'BuyDate': buy_date,
            'BuyPrice': buy_price,
            'SellDate': sell_date,
            'SellPrice': sell_price,
            'HoldingDays': holding_days,
            'ProfitPct': profit_pct,
            'SellReason': sell_reason
        })

    return pd.DataFrame(positions)


def simple_forecast(data: pd.DataFrame, days: int = 5, window: int = 20, method: str = 'linear') -> pd.Series:
    """Produce a simple linear forecast for the next `days` trading days using a linear fit on the
    last `window` close prices. Returns a pandas Series indexed by business days after the last index.
    This is a lightweight, illustrative predictor (not a production model).
    """
    df = data.copy()
    if df.empty or 'Close' not in df.columns:
        return pd.Series(dtype=float)

    x = np.arange(len(df))
    y = df['Close'].values
    # use only recent window
    if len(y) >= max(5, window):
        x = x[-window:]
        y = y[-window:]

    # choose method
    slope, intercept = 0.0, float(y[-1])
    if method == 'linear':
        try:
            coeffs = np.polyfit(x, y, 1)
            slope, intercept = coeffs[0], coeffs[1]
        except Exception:
            slope, intercept = 0.0, float(y[-1])
    elif method == 'ma':
        # forecast as repeating the last moving average value
        try:
            ma = pd.Series(y).rolling(window=min(len(y), window)).mean().iloc[-1]
            slope, intercept = 0.0, float(ma if not pd.isna(ma) else y[-1])
        except Exception:
            slope, intercept = 0.0, float(y[-1])
    elif method == 'ema':
        try:
            ema = pd.Series(y).ewm(span=min(len(y), window), adjust=False).mean().iloc[-1]
            slope, intercept = 0.0, float(ema if not pd.isna(ema) else y[-1])
        except Exception:
            slope, intercept = 0.0, float(y[-1])
    else:
        # default fallback
        try:
            coeffs = np.polyfit(x, y, 1)
            slope, intercept = coeffs[0], coeffs[1]
        except Exception:
            slope, intercept = 0.0, float(y[-1])

    last_idx = df.index[-1]
    # create business-day index for forecast
    try:
        future_index = pd.bdate_range(start=last_idx + pd.Timedelta(days=1), periods=days)
    except Exception:
        future_index = pd.date_range(start=last_idx + pd.Timedelta(days=1), periods=days)

    # map future x values
    start_x = (len(data) - 1) if len(data) > 0 else 0
    future_x = np.arange(start_x + 1, start_x + 1 + days)
    preds = intercept + slope * future_x
    return pd.Series(data=preds, index=future_index)


def generate_suggestion(data: pd.DataFrame, forecast: pd.Series, atr_window: int = 14) -> dict:
    """Generate a simple suggestion (Buy/Sell/Hold) based on forecast and ATR volatility.

    Logic:
    - Compute predicted_pct = (last_forecast / last_close - 1) * 100
    - Compute ATR% = (ATR / last_close) * 100 (if ATR available)
    - If predicted_pct > 1.5 * ATR% => Suggest BUY
    - If predicted_pct < -1.5 * ATR% => Suggest SELL
    - Else => Hold
    Returns a dict with keys: predicted_pct, atr_pct, recommendation, reason
    """
    result = {
        'predicted_pct': None,
        'atr_pct': None,
        'recommendation': 'Hold',
        'reason': 'Insufficient data'
    }

    if data.empty or 'Close' not in data.columns or forecast.empty:
        return result

    last_close = float(data['Close'].iloc[-1])
    last_pred = float(forecast.iloc[-1])
    predicted_pct = (last_pred / last_close - 1) * 100
    result['predicted_pct'] = predicted_pct

    # ATR
    if 'ATR' not in data.columns:
        data = calculate_atr(data, window=atr_window)
    atr = data['ATR'].iloc[-1] if 'ATR' in data.columns and not pd.isna(data['ATR'].iloc[-1]) else None
    if atr is not None and last_close > 0:
        atr_pct = (atr / last_close) * 100
        result['atr_pct'] = atr_pct
    else:
        # fallback threshold
        result['atr_pct'] = 1.0

    threshold = 1.5 * (result['atr_pct'] or 1.0)
    if predicted_pct > threshold:
        result['recommendation'] = 'Consider BUY'
        result['reason'] = f'Predicted +{predicted_pct:.2f}% > threshold {threshold:.2f}% (ATR-based)'
    elif predicted_pct < -threshold:
        result['recommendation'] = 'Consider SELL'
        result['reason'] = f'Predicted {predicted_pct:.2f}% < -{threshold:.2f}% (ATR-based)'
    else:
        result['recommendation'] = 'Hold'
        result['reason'] = f'Predicted {predicted_pct:.2f}% within threshold ±{threshold:.2f}%'

    return result


def simple_forecast(data: pd.DataFrame, days: int = 5, method: str = 'linear', window: int = 60, degree: int = 1) -> pd.DataFrame:
    """Produce a simple forecast for the 'Close' price.

    Methods supported:
    - 'ma': repeat the moving average of last `window` closes
    - 'ema': use exponentially weighted moving average as level forecast
    - 'linear': fit a polynomial (degree) to the last `window` closes and extrapolate

    Returns a DataFrame with a DatetimeIndex for the next `days` periods and a 'Forecast' column.
    """
    df = data.copy()
    if df.empty or 'Close' not in df.columns:
        return pd.DataFrame()

    # infer delta between last two points (fallback to 1 day)
    if len(df.index) >= 2:
        delta = df.index[-1] - df.index[-2]
    else:
        delta = pd.Timedelta(days=1)

    # choose the history window for fitting
    hist = df['Close'].dropna()
    if window <= 0:
        window = min(60, len(hist))
    hist = hist.iloc[-window:]
    if hist.empty:
        return pd.DataFrame()

    last_ts = df.index[-1]
    freq = delta
    future_index = [last_ts + (i + 1) * freq for i in range(days)]

    if method == 'ma':
        val = hist.mean()
        forecast_values = [val] * days
    elif method == 'ema':
        span = max(3, min(window, 30))
        val = hist.ewm(span=span, adjust=False).mean().iloc[-1]
        forecast_values = [val] * days
    else:
        # linear / polynomial fit on integer x
        x = np.arange(len(hist))
        y = hist.values
        deg = max(1, int(degree))
        # if not enough points, fallback to MA
        if len(x) <= deg:
            val = hist.mean()
            forecast_values = [val] * days
        else:
            coeffs = np.polyfit(x, y, deg)
            poly = np.poly1d(coeffs)
            x_future = np.arange(len(hist), len(hist) + days)
            forecast_values = poly(x_future).tolist()

    forecast_df = pd.DataFrame({'Forecast': forecast_values}, index=pd.to_datetime(future_index))
    return forecast_df
