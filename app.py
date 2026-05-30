import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest

# -----------------------------
# Generate Sample IoT Data
# -----------------------------

np.random.seed(42)

samples = 500

temperature = np.random.normal(
    loc=30,
    scale=3,
    size=samples
)

humidity = np.random.normal(
    loc=60,
    scale=10,
    size=samples
)

pressure = np.random.normal(
    loc=1013,
    scale=8,
    size=samples
)

# Create anomalies

temperature[50] = 55
temperature[120] = 5

humidity[200] = 98

pressure[330] = 1080

data = pd.DataFrame({

    "Temperature":temperature,
    "Humidity":humidity,
    "Pressure":pressure
})

# Save Dataset

data.to_csv(
    "iot_sensor_data.csv",
    index=False
)

print("Dataset Created")

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv(
    "iot_sensor_data.csv"
)

print(df.head())

# -----------------------------
# Statistical Analysis
# -----------------------------

print("\nStatistics")

print(df.describe())

# -----------------------------
# Anomaly Detection
# -----------------------------

model = IsolationForest(
    contamination=0.02,
    random_state=42
)

df["Anomaly"] = model.fit_predict(df)

# Convert values

df["Anomaly"] = df["Anomaly"].map({

    1:"Normal",
   -1:"Anomaly"

})

print("\nAnomaly Counts")

print(
df["Anomaly"].value_counts()
)

# -----------------------------
# Visualization
# -----------------------------

plt.figure(figsize=(10,5))

plt.plot(
    df["Temperature"],
    label="Temperature"
)

plt.title(
    "Temperature Sensor Data"
)

plt.xlabel(
    "Time"
)

plt.ylabel(
    "Temperature"
)

plt.legend()

plt.grid()

plt.show()

# Scatter plot anomaly visualization

plt.figure(figsize=(8,5))

colors = []

for value in df["Anomaly"]:

    if value=="Anomaly":
        colors.append("red")
    else:
        colors.append("blue")

plt.scatter(

    df.index,
    df["Temperature"],
    c=colors

)

plt.title(
"Detected Sensor Anomalies"
)

plt.xlabel(
"Sample"
)

plt.ylabel(
"Temperature"
)

plt.show()

# -----------------------------
# User Prediction Section
# -----------------------------

while True:

    print("\nEnter Sensor Values")

    temp = input("Temperature: ")

    if temp=="exit":
        break

    hum = float(
        input("Humidity: ")
    )

    pres = float(
        input("Pressure: ")
    )

    sample = np.array([[
        float(temp),
        hum,
        pres
    ]])

    result = model.predict(sample)

    if result[0]==1:

        print("Sensor Status = Normal")

    else:

        print("Sensor Status = Anomaly")