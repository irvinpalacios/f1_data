import fastf1
from fastf1 import plotting
import streamlit as st
import pandas as pd
import plotly.express as px

# Enable FastF1 cache for faster reloads
fastf1.Cache.enable_cache('./cache')

st.set_page_config(page_title="F1 Telemetry Comparator", layout="wide")
st.title("🏎️ F1 Driver Telemetry Comparator")

# Sidebar controls
year = st.sidebar.selectbox("Select Year", list(range(2022, 2025)))
schedule = fastf1.get_event_schedule(year)
race_name = st.sidebar.selectbox("Select Grand Prix", schedule['EventName'].tolist())
session_type = st.sidebar.selectbox("Session Type", ["R", "Q", "FP1", "FP2", "FP3"])

load_button = st.sidebar.button("Load Session")

if load_button:
    with st.spinner("Loading session data..."):
        try:
            event = fastf1.get_event(year, race_name)
            session = event.get_session(session_type)
            session.load()
        except Exception as e:
            st.error(f"Error loading session: {e}")
            st.stop()

    st.success("Session Loaded!")

    # List drivers in the session
    drivers = session.drivers
    driver_info = session.results

    driver1 = st.selectbox("Select Driver 1", drivers)
    driver2 = st.selectbox("Select Driver 2", drivers)

    if st.button("Compare Telemetry"):
        lap1 = session.laps.pick_driver(driver1).pick_fastest()
        lap2 = session.laps.pick_driver(driver2).pick_fastest()

        tel1 = lap1.get_telemetry().add_distance()
        tel2 = lap2.get_telemetry().add_distance()

        st.subheader("Fastest Lap Comparison")
        st.write(f"**{driver1} Fastest Lap:** {lap1['LapTime']} • **{driver2} Fastest Lap:** {lap2['LapTime']}")

        # Plotting helper function
        def plot_line(df1, df2, col, title):
            fig = px.line(
                pd.DataFrame({
                    "Distance (m)": pd.concat([df1["Distance"], df2["Distance"]], axis=0),
                    col: pd.concat([df1[col], df2[col]], axis=0),
                    "Driver": [driver1]*len(df1) + [driver2]*len(df2)
                }),
                x="Distance (m)", y=col, color="Driver", title=title
            )
            st.plotly_chart(fig, use_container_width=True)

        # Telemetry plots
        plot_line(tel1, tel2, "Speed", "Speed Comparison (km/h)")
        col1, col2 = st.columns(2)
        with col1:
            plot_line(tel1, tel2, "Throttle", "Throttle (%)")
            plot_line(tel1, tel2, "Brake", "Brake (%)")
        with col2:
            plot_line(tel1, tel2, "nGear", "Gear")
