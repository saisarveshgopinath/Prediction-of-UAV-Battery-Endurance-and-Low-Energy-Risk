
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="UAV Battery ML Dashboard", layout="wide")

DATA_FILE = "UAV_Battery_SOC_SOH_Dataset.csv"
SOH_MODEL = "outputs/models/soh_model.pkl"
STATE_MODEL = "outputs/models/battery_state_model.pkl"

st.title("🚁 UAV Battery ML Monitoring Dashboard")
st.caption("Machine-learning based battery health and state monitoring")

df = pd.read_csv(DATA_FILE)
soh_model = joblib.load(SOH_MODEL)
state_model = joblib.load(STATE_MODEL)

st.sidebar.header("Battery Selection")
idx = st.sidebar.number_input("Dataset row", min_value=0, max_value=len(df)-1, value=0, step=1)
row = df.iloc[idx]

c1,c2,c3,c4 = st.columns(4)
c1.metric("SOC", f"{row['soc_pct']:.2f}%")
c2.metric("Actual SOH", f"{row['soh_pct']:.2f}%")
c3.metric("Voltage", f"{row['voltage_v']:.2f} V")
c4.metric("Temperature", f"{row['temperature_c']:.2f} °C")

features_soh = df.drop(columns=[
    "soh_pct","soc_pct","capacity_fade_pct","resistance_growth_pct",
    "battery_id","uav_id","battery_state_category"
], errors="ignore")
input_soh = row[features_soh.columns].to_frame().T
pred_soh = float(soh_model.predict(input_soh)[0])

features_state = df.drop(columns=[
    "battery_state_category","soc_pct","soh_pct",
    "capacity_fade_pct","resistance_growth_pct",
    "battery_id","uav_id"
], errors="ignore")
input_state = row[features_state.columns].to_frame().T
pred_state = state_model.predict(input_state)[0]

c5,c6,c7,c8 = st.columns(4)
c5.metric("Predicted SOH", f"{pred_soh:.2f}%")
c6.metric("Predicted State", str(pred_state))
c7.metric("Cycle Count", f"{row['cycle_count']:.0f}")
c8.metric("Flight Duration", f"{row['flight_duration_min']:.2f} min")

st.subheader("Battery Parameters")
st.dataframe(row.to_frame("Value"), use_container_width=True)

col1,col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="cycle_count", y="soh_pct", alpha=0.35, ax=ax)
    ax.scatter(row["cycle_count"], row["soh_pct"], s=100)
    ax.set_title("SOH vs Cycle Count")
    st.pyplot(fig)
with col2:
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="capacity_fade_pct", y="soh_pct", alpha=0.35, ax=ax)
    ax.scatter(row["capacity_fade_pct"], row["soh_pct"], s=100)
    ax.set_title("SOH vs Capacity Fade")
    st.pyplot(fig)

st.subheader("Battery State Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x="battery_state_category",
              order=df["battery_state_category"].value_counts().index, ax=ax)
ax.tick_params(axis="x", rotation=25)
st.pyplot(fig)

st.info("The classification model was trained without SOC, SOH, capacity fade, resistance growth, battery IDs, or the target state label to reduce target leakage.")
