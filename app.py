import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
st.set_page_config(
    page_title="Climate Intelligence Dashboard",
    layout="wide"
)

 

@st.cache_data

def load_data():

    data = pd.read_csv("cleaned_weather_data.csv")

    data["last_updated"] = pd.to_datetime(

        data["last_updated"],

        errors="coerce"

    )

    return data.dropna()

 

weather = load_data()

 

st.title("Climate Intelligence Dashboard")

 

st.caption(

    "Educational prototype - not an official forecast"

)

 

cities = sorted(weather["location_name"].unique())

 

city = st.selectbox("Choose a city", cities)

 

city_data = weather[

    weather["location_name"] == city

].sort_values("last_updated")
 
average_temp = city_data["temperature_celsius"].mean()
average_humidity = city_data["humidity"].mean()
total_rainfall = city_data["precip_mm"].sum()
 
col1, col2, col3 = st.columns(3)
col1.metric("Average temperature", f"{average_temp:.1f} C")
col2.metric("Average humidity", f"{average_humidity:.1f}%")
col3.metric("Total rainfall", f"{total_rainfall:.1f} mm")
 
if average_temp >= 35 and average_humidity >= 60:
    risk = "High heat risk"
elif average_temp >= 30:
    risk = "Medium heat risk"
else:
    risk = "Low heat risk"
 
st.subheader("Risk indicator")
st.info(risk)
 
st.subheader("Temperature trend")
fig, ax = plt.subplots()
ax.plot(
    city_data["last_updated"],
    city_data["temperature_celsius"],
    marker="o",
    color="#2E74B5"
)
ax.set_xlabel("Date")
ax.set_ylabel("Temperature (C)")
plt.xticks(rotation=45)
st.pyplot(fig)

# Machine Learning Temperature Estimator
st.subheader("Machine Learning Temperature Estimator")

st.write("Enter weather information to estimate the temperature.")

# User inputs
humidity = st.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=55.0
)

precip = st.number_input(
    "Precipitation (mm)",
    min_value=0.0,
    value=1.2
)

wind = st.number_input(
    "Wind Speed (km/h)",
    min_value=0.0,
    value=12.0
)

pressure = st.number_input(
    "Air Pressure (mb)",
    min_value=0.0,
    value=1012.0
)

# Train the machine learning model
features = [
    "humidity",
    "precip_mm",
    "wind_kph",
    "pressure_mb"
]

model_data = weather[
    features + ["temperature_celsius"]
].dropna()

X = model_data[features]
y = model_data["temperature_celsius"]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Predict temperature
if st.button("Estimate Temperature"):

    sample_input = pd.DataFrame([{
        "humidity": humidity,
        "precip_mm": precip,
        "wind_kph": wind,
        "pressure_mb": pressure
    }])

    prediction = model.predict(sample_input)[0]

    st.success(
        f"Estimated Temperature: {prediction:.2f} °C"
    )
