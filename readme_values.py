import pandas as pd

hist = pd.read_csv("history.csv", parse_dates=["timestamp"]).sort_values("timestamp")
res = pd.read_csv("results.csv")

# Rows used for modeling (first 24 and last 24 rows are lost to lags and the 24-hour target)
ts = hist["timestamp"].iloc[24:-24].reset_index(drop=True)
split = int(len(ts) * 0.8)

print("=== DATA SECTION ===")
print(f"Start date: {hist['timestamp'].min().date()}")
print(f"End date:   {hist['timestamp'].max().date()}")
print(f"Total rows: {len(hist)}")
print(f"Test set:   {len(ts) - split} rows, {ts.iloc[split].date()} to {ts.iloc[-1].date()}")

print("\n=== SAMPLE ROWS (paste into the Data table) ===")
cols = ["timestamp", "temp_f", "humidity_pct", "wind_mph", "pressure_hpa"]
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for _, r in hist[cols].iloc[[0, 1, 2, 3, 4]].iterrows():
    print(f"| {r['timestamp']} | {r['temp_f']:.1f} | {r['humidity_pct']:.0f} | {r['wind_mph']:.1f} | {r['pressure_hpa']:.1f} |")

print("\n=== RESULTS TABLE (paste into the Results section) ===")
print("| Model | MAE (F) | RMSE (F) |")
print("|---|---|---|")
for _, r in res.iterrows():
    print(f"| {r['model']} | {r['MAE_F']:.2f} | {r['RMSE_F']:.2f} |")

base = res.loc[res["model"].str.startswith("Baseline"), "MAE_F"].iloc[0]
best = res.sort_values("MAE_F").iloc[0]
print("\n=== KEY FINDING / RESUME NUMBERS ===")
print(f"Best model: {best['model']} (MAE {best['MAE_F']:.2f} F)")
print(f"Baseline MAE: {base:.2f} F")
change = (base - best["MAE_F"]) / base * 100
if best["model"].startswith("Baseline"):
    print("The baseline was the best, so no model beat it.")
else:
    print(f"The best model's MAE is {change:.1f}% lower than the baseline.")