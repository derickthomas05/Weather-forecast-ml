from datetime import date, timedelta
import requests
import pandas as pd

URL = "https://archive-api.open-meteo.com/v1/archive"
end = date.today() - timedelta(days=7)   # archive data lags by a few days
start = end - timedelta(days=730)        # about 2 years of history

params = {
    "latitude": 42.73, "longitude": -73.69,   # Troy, NY
    "start_date": start.isoformat(), "end_date": end.isoformat(),
    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
    "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
    "timezone": "America/New_York",
}

r = requests.get(URL, params=params, timeout=60)
r.raise_for_status()

df = pd.DataFrame(r.json()["hourly"]).rename(columns={
    "time": "timestamp", "temperature_2m": "temp_f",
    "relative_humidity_2m": "humidity_pct", "wind_speed_10m": "wind_mph",
    "surface_pressure": "pressure_hpa"})
df = df.dropna()
df.to_csv("history.csv", index=False)
print(f"Saved {len(df)} rows from {start} to {end} to history.csv")