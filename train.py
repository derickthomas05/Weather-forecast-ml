import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

HORIZON = 24  # predict the temperature this many hours ahead

df = (pd.read_csv("history.csv", parse_dates=["timestamp"])
        .sort_values("timestamp").reset_index(drop=True))

# ---- Features: only information available at prediction time ----
hour = df["timestamp"].dt.hour
doy = df["timestamp"].dt.dayofyear
df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
df["doy_cos"] = np.cos(2 * np.pi * doy / 365)
df["temp_lag_3"] = df["temp_f"].shift(3)
df["temp_lag_6"] = df["temp_f"].shift(6)
df["temp_lag_24"] = df["temp_f"].shift(24)
df["temp_change_3h"] = df["temp_f"] - df["temp_lag_3"]

# ---- Target: temperature HORIZON hours in the future ----
df["target"] = df["temp_f"].shift(-HORIZON)
df = df.dropna().reset_index(drop=True)

FEATURES = ["temp_f", "humidity_pct", "wind_mph", "pressure_hpa",
            "hour_sin", "hour_cos", "doy_sin", "doy_cos",
            "temp_lag_3", "temp_lag_6", "temp_lag_24", "temp_change_3h"]

# ---- Time-based split (no shuffling, so the test set is the "future") ----
split = int(len(df) * 0.8)
train, test = df.iloc[:split], df.iloc[split:]
X_train, y_train = train[FEATURES], train["target"]
X_test, y_test = test[FEATURES], test["target"]
print(f"Train rows: {len(train)} | Test rows: {len(test)}")

def score(name, pred):
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    return {"model": name, "MAE_F": round(mae, 2), "RMSE_F": round(rmse, 2)}

results, preds = [], {}

# Baseline: "tomorrow at this hour will be the same as right now"
preds["Baseline (same as now)"] = X_test["temp_f"].values

lin = LinearRegression().fit(X_train, y_train)
preds["Linear Regression"] = lin.predict(X_test)

rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5,
                           n_jobs=-1, random_state=42).fit(X_train, y_train)
preds["Random Forest"] = rf.predict(X_test)

for name, p in preds.items():
    results.append(score(name, p))

res = pd.DataFrame(results).sort_values("MAE_F")
print("\nResults on the held-out test set (lower is better):")
print(res.to_string(index=False))
res.to_csv("results.csv", index=False)

imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nRandom Forest feature importance:")
print(imp.round(3).to_string())

# ---- Plot: last 7 days of the test set ----
n = 24 * 7
t = test["timestamp"].iloc[-n:] + pd.Timedelta(hours=HORIZON)  # time being predicted
plt.figure(figsize=(11, 4.5))
plt.plot(t, y_test.iloc[-n:], label="Actual", linewidth=2)
plt.plot(t, preds["Random Forest"][-n:], label="Random Forest", alpha=0.85)
plt.plot(t, preds["Baseline (same as now)"][-n:], label="Baseline", alpha=0.6, linestyle="--")
plt.title(f"{HORIZON}-hour-ahead temperature forecast (last 7 days of test data)")
plt.ylabel("Temperature (F)")
plt.legend()
plt.tight_layout()
plt.savefig("forecast_plot.png", dpi=150)
print("\nSaved results.csv and forecast_plot.png")