import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ==========================================================
# Technical indicator helper functions
# ==========================================================
def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average (EMA)."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    # Deal with divisions by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))

    return rsi_val

def momentum(series: pd.Series, period: int) -> pd.Series:
    """Momentum (MOM)."""
    return series.diff(period)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    Moving Average Convergence Divergence (MACD).
    Returns MACD line, Signal line and MACD histogram.
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def stochastic_oscillator(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    period: int = 14,
    D_period: int = 3,
):
    """
    Stochastic Oscillator (%K, %D).
    """
    lowests = low.rolling(window=period, min_periods=period).min()
    highests = high.rolling(window=period, min_periods=period).max()

    # Deal with divisions by zero
    K = 100 * ((close - lowests) / (highests - lowests).replace(0, np.nan))
    D = K.rolling(window=D_period, min_periods=D_period).mean()

    return K, D

def proc(series: pd.Series, period: int) -> pd.Series:
    """Percentage Rate of Change (PROC)."""
    proc_val = 100 * (series / series.shift(period) - 1)

    # Deal with divisions by zero
    proc_val = proc_val.replace([np.inf, -np.inf], np.nan)

    return proc_val


# ==========================================================
# 1. Load initial datasheets
# ==========================================================
print("Step 1/6: Loading data...")

file_path = '../data/20251111_JUNCTION_training.xlsx'
groups_data = pd.read_excel(file_path, sheet_name='groups')
consumption_data = pd.read_excel(file_path, sheet_name='training_consumption')
price_data = pd.read_excel(file_path, sheet_name='training_prices')

# Ensure datetime and drop timezone -> make them naive
consumption_data["measured_at"] = (
    pd.to_datetime(consumption_data["measured_at"], utc=True)
      .dt.tz_convert(None)
)
price_data["measured_at"] = (
    pd.to_datetime(price_data["measured_at"], utc=True)
      .dt.tz_convert(None)
)

# Prepare price data
price_data = price_data.rename(columns={
    "measured_at": "timestamp",
    "eur_per_mwh": "price",
})
price_data["price"] = price_data["price"].fillna(price_data["price"].mean())

# Identify user group columns (all except the timestamp)
group_cols = [c for c in consumption_data.columns if c != "measured_at"]
if len(group_cols) == 0:
    raise ValueError("No user group columns found in 'training_consumption' sheet.")

print(f"  Loaded data for {len(group_cols)} groups.")
print("Step 1/6: Done.\n")


# ==========================================================
# 2. Build feature dataframe for all groups (stacked)
# ==========================================================
print("Step 2/6: Building features per group...")

all_dfs = []

for i, g in enumerate(group_cols, start=1):
    print(f"  Processing group {i}/{len(group_cols)}: {g}")

    # Base df for one group
    df_g = pd.DataFrame(
        {
            "timestamp": consumption_data["measured_at"],
            "consumption": consumption_data[g].astype(float),
            "group_id": g,
        }
    )

    # Merge prices
    df_g = df_g.merge(
        price_data[["timestamp", "price"]],
        on="timestamp",
        how="left",
    )

    # Sort for time-series features
    df_g = df_g.sort_values("timestamp")

    # ------------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------------
    df_g["lag_1"]   = df_g["consumption"].shift(1)
    df_g["lag_24"]  = df_g["consumption"].shift(24)
    df_g["lag_168"] = df_g["consumption"].shift(168)  # one-week lag

    # ------------------------------------------------------------------
    # Technical indicators for consumption (only)
    # ------------------------------------------------------------------
    cons = df_g["consumption"]

    df_g["cons_ema_12"]  = ema(cons, 12)
    df_g["cons_ema_24"]  = ema(cons, 24)
    df_g["cons_rsi_14"]  = rsi(cons, 14)
    df_g["cons_mom_12"]  = momentum(cons, 12)
    df_g["cons_proc_12"] = proc(cons, 12)

    cons_macd, cons_macd_signal, cons_macd_hist = macd(cons, fast=12, slow=26, signal=9)
    df_g["cons_macd"]        = cons_macd
    df_g["cons_macd_signal"] = cons_macd_signal
    df_g["cons_macd_hist"]   = cons_macd_hist

    # For stochastic oscillator, approximate high/low as rolling highs/lows of the series itself
    cons_high_14 = cons.rolling(window=14, min_periods=14).max()
    cons_low_14  = cons.rolling(window=14, min_periods=14).min()
    cons_k_14, cons_d_14 = stochastic_oscillator(
        close=cons,
        high=cons_high_14,
        low=cons_low_14,
        period=14,
        D_period=3,
    )
    df_g["cons_stoch_k_14"] = cons_k_14
    df_g["cons_stoch_d_14"] = cons_d_14

    # Drop rows where any of the lag or indicator features are NaN
    before_rows = len(df_g)
    df_g = df_g.dropna()
    after_rows = len(df_g)

    print(f"    Rows before dropna: {before_rows}, after dropna: {after_rows}")

    all_dfs.append(df_g)

# Concatenate all groups
df_all = pd.concat(all_dfs, ignore_index=True)
print(f"Step 2/6: Done. Combined feature dataframe shape: {df_all.shape}\n")


# ==========================================================
# 3. Time-based and price features
# ==========================================================
print("Step 3/6: Adding time-based features...")

df_all["hour"]        = df_all["timestamp"].dt.hour
df_all["day_of_week"] = df_all["timestamp"].dt.dayofweek
df_all["is_weekend"]  = (df_all["day_of_week"] >= 5).astype(int)

# Price aligned with the target hour (no price indicators)
df_all["price_t"] = df_all["price"]

print("Step 3/6: Done.\n")


# ==========================================================
# 4. Train/test split
# ==========================================================
print("Step 4/6: Train/test split...")

cutoff = pd.Timestamp("2024-09-28 23:00:00")

feature_cols = [
    "group_id",
    "lag_1",
    "lag_24",
    "lag_168",
    "hour",
    "day_of_week",
    "is_weekend",
    "price_t",

    # Consumption indicators only
    "cons_ema_12",
    "cons_ema_24",
    "cons_rsi_14",
    "cons_mom_12",
    "cons_proc_12",
    "cons_macd",
    "cons_macd_signal",
    "cons_macd_hist",
    "cons_stoch_k_14",
    "cons_stoch_d_14",
]

train_mask = df_all["timestamp"] <= cutoff
test_mask  = df_all["timestamp"] >= cutoff

X_train = df_all.loc[train_mask, feature_cols].copy()
y_train = df_all.loc[train_mask, "consumption"].copy()

X_test = df_all.loc[test_mask, feature_cols].copy()
y_test = df_all.loc[test_mask, "consumption"].copy()

print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"  X_test  shape: {X_test.shape}, y_test  shape: {y_test.shape}")
print("Step 4/6: Done.\n")


# ==========================================================
# 5. Scale selected numeric features
# ==========================================================
print("Step 5/6: Scaling numeric features...")

scaler = StandardScaler()

cols_to_scale = [
    "lag_1",
    "lag_24",
    "lag_168",
    "price_t",

    "cons_ema_12",
    "cons_ema_24",
    "cons_rsi_14",
    "cons_mom_12",
    "cons_proc_12",
    "cons_macd",
    "cons_macd_signal",
    "cons_macd_hist",
    "cons_stoch_k_14",
    "cons_stoch_d_14",
]

X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])

print("Step 5/6: Done.\n")


# ==========================================================
# 6. Save datasets and scaler (to data/third)
# ==========================================================
print("Step 6/6: Saving outputs to ../data/third ...")

output_dir = "../data/third"
os.makedirs(output_dir, exist_ok=True)

joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)

X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

print("All done. Files saved:")
print(f"  {output_dir}/scaler.pkl")
print(f"  {output_dir}/X_train.csv")
print(f"  {output_dir}/y_train.csv")
print(f"  {output_dir}/X_test.csv")
print(f"  {output_dir}/y_test.csv")