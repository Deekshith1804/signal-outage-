import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from streamlit_folium import folium_static
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut
import certifi
import ssl

# ---------------------- Page Setup ---------------------- #
st.set_page_config(page_title="GNSS Outage Predictor", layout="wide")
st.title("📡 GNSS Outage Prediction Based on Weather & Ionospheric Data")

# ---------------------- Sidebar ---------------------- #
st.sidebar.header("🛠️ Simulation Controls")
n_rows = st.sidebar.slider("Number of Data Points", 100, 2000, 1000, step=100)
start_date = st.sidebar.date_input("Select Start Date", value=datetime(2025, 6, 9))
location_input = st.sidebar.text_input("Enter Location (e.g., Tirupati)", value="India")

# ---------------------- Geocode Location ---------------------- #
geolocator = Nominatim(user_agent="gnss_app", ssl_context=ssl.create_default_context(cafile=certifi.where()))
try:
    location = geolocator.geocode(location_input, timeout=10)
    if location:
        map_center = [location.latitude, location.longitude]
        lat, lon = location.latitude, location.longitude
    else:
        st.sidebar.warning("❌ Location not found. Using default center (India).")
        map_center = [20, 80]
        lat, lon = map_center
except (GeocoderUnavailable, GeocoderTimedOut):
    st.sidebar.warning("⚠️ Geocoding service unavailable. Using default center (India).")
    map_center = [20, 80]
    lat, lon = map_center

# ---------------------- Generate Synthetic Data ---------------------- #
dates = pd.date_range(start=start_date, periods=n_rows, freq="H")
data = pd.DataFrame({
    "timestamp": dates,
    "latitude": np.random.uniform(lat - 5, lat + 5, n_rows),
    "longitude": np.random.uniform(lon - 5, lon + 5, n_rows),
    "rain": np.random.uniform(0, 50, n_rows),
    "cloud_cover": np.random.uniform(0, 100, n_rows),
    "TEC": np.random.uniform(1, 100, n_rows),
    "geomagnetic": np.random.uniform(10, 500, n_rows),
})

# ---------------------- Outage Rule ---------------------- #
data["outage"] = (
    (data["TEC"] > 70) &
    (data["geomagnetic"] > 300) &
    (data["cloud_cover"] > 70)
).astype(int)

# ---------------------- GNSS Outage at Location ---------------------- #
nearest_point = data.iloc[(data[["latitude", "longitude"]] - [lat, lon]).pow(2).sum(axis=1).idxmin()]
gnss_outage_status = "🔴 GNSS Outage Risk" if nearest_point["outage"] == 1 else "🟢 No Outage Risk"
st.sidebar.markdown("---")
st.sidebar.subheader("📍 GNSS Outage Status at Location")
st.sidebar.write(f"**{gnss_outage_status}**")
st.sidebar.write(f"**TEC:** {nearest_point['TEC']:.2f}")
st.sidebar.write(f"**Geomagnetic:** {nearest_point['geomagnetic']:.2f}")
st.sidebar.write(f"**Cloud Cover:** {nearest_point['cloud_cover']:.2f}")

# ---------------------- Show Raw Data ---------------------- #
if st.checkbox("Show Raw Data"):
    st.dataframe(data.head())

# ---------------------- Train Model ---------------------- #
features = ["rain", "cloud_cover", "TEC", "geomagnetic"]
X_train, X_test, y_train, y_test = train_test_split(data[features], data["outage"], test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------------- Model Performance ---------------------- #
st.subheader("📈 Model Performance")
st.text(classification_report(y_test, y_pred))

# ---------------------- Heatmap + Marker ---------------------- #
st.subheader("🌍 GNSS Outage Locations Map")
heat_data = data[data["outage"] == 1][["latitude", "longitude"]].values.tolist()

m = folium.Map(location=map_center, zoom_start=6)
HeatMap(heat_data).add_to(m)

popup_text = f"""
<b>Location:</b> {location_input}<br>
<b>Coordinates:</b> {lat:.2f}, {lon:.2f}<br>
<b>TEC:</b> {nearest_point['TEC']:.2f}<br>
<b>Geomagnetic:</b> {nearest_point['geomagnetic']:.2f}<br>
<b>Cloud Cover:</b> {nearest_point['cloud_cover']:.2f}<br>
<b>Status:</b> {gnss_outage_status}
"""

folium.Marker(
    location=[lat, lon],
    popup=folium.Popup(popup_text, max_width=300),
    icon=folium.Icon(color="red" if nearest_point["outage"] == 1 else "green", icon="signal", prefix="fa")
).add_to(m)

folium_static(m)

# ---------------------- Download CSV ---------------------- #
st.download_button(
    label="⬇️ Download Dataset as CSV",
    data=data.to_csv(index=False),
    file_name="gnss_outage_dataset.csv",
    mime="text/csv"
)

