"""Streamlit Race Playback app using FastF1 telemetry."""
import time
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import fastf1
from fastf1 import plotting


# Enable on-disk cache for FastF1
fastf1.Cache.enable_cache("./cache", ignore_version=True)

st.set_page_config(page_title="F1 Race Playback", layout="wide")


# ---------------------------
# Data loading helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def load_session_data(season: int, rnd: int):
    """Load a race session and prepare telemetry for playback."""
    session = fastf1.get_session(season, rnd, "R")
    session.load(telemetry=True, laps=True, weather=False)

    drivers = session.drivers
    colors = {drv: plotting.driver_color(drv) for drv in drivers}

    telemetry_rows = []
    lap_time_lookup: Dict[str, Dict[int, float]] = {}

    for drv in drivers:
        driver_laps = session.laps.pick_driver(drv)
        lap_time_lookup[drv] = {}

        for _, lap in driver_laps.iterrows():
            lap_number = int(lap["LapNumber"])
            lap_time = lap["LapTime"]
            if pd.notnull(lap_time):
                lap_time_lookup[drv][lap_number] = lap_time.total_seconds()

            tel = lap.get_telemetry().add_distance()
            if tel.empty:
                continue

            lap_distance = float(tel["Distance"].max())
            lap_start = float(lap["LapStartTime"].total_seconds())

            chunk = tel[["SessionTime", "X", "Y", "Speed", "Distance"]].copy()
            chunk["Driver"] = drv
            chunk["LapNumber"] = lap_number
            chunk["TimeIntoLap"] = chunk["SessionTime"].dt.total_seconds() - lap_start
            chunk["SessionTime"] = chunk["SessionTime"].dt.total_seconds()
            chunk["LapTotalDistance"] = lap_distance
            telemetry_rows.append(chunk)

    if not telemetry_rows:
        raise RuntimeError("No telemetry available for this race")

    telemetry = pd.concat(telemetry_rows, ignore_index=True)
    telemetry.sort_values(["Driver", "SessionTime"], inplace=True)

    # Build resampled timeline at 10 Hz for smoother playback
    start_t = telemetry["SessionTime"].min()
    end_t = telemetry["SessionTime"].max()
    timeline = np.arange(start_t, end_t, 0.1)

    resampled_frames = []
    for drv, df in telemetry.groupby("Driver"):
        drv_df = df.set_index("SessionTime").reindex(timeline)
        drv_df[["X", "Y", "Speed", "Distance", "TimeIntoLap", "LapTotalDistance"]] = (
            drv_df[["X", "Y", "Speed", "Distance", "TimeIntoLap", "LapTotalDistance"]]
            .interpolate()
            .bfill()
            .ffill()
        )
        drv_df["LapNumber"] = drv_df["LapNumber"].interpolate().round().astype(int)
        drv_df["Driver"] = drv
        drv_df["SessionTime"] = timeline
        resampled_frames.append(drv_df.reset_index(drop=True))

    positions = pd.concat(resampled_frames, ignore_index=True)

    # Winner laps for quick lap jump
    winner_id = int(session.results.sort_values("Position").iloc[0]["DriverId"])
    winner_laps = session.laps.pick_driver(winner_id)
    lap_jump = {
        int(row["LapNumber"]): float(row["LapStartTime"].total_seconds())
        for _, row in winner_laps.iterrows()
        if pd.notnull(row["LapStartTime"])
    }

    return session, positions, colors, lap_time_lookup, lap_jump


def compute_order_frame(frame: pd.DataFrame, lap_time_lookup: Dict[str, Dict[int, float]]):
    """Compute running order and gaps from a frame of telemetry."""
    frame = frame.copy()
    frame["LapTotalDistance"] = frame["LapTotalDistance"].replace(0, np.nan).bfill()
    frame["Progress"] = (frame["LapNumber"] - 1) + (
        frame["Distance"] / frame["LapTotalDistance"].replace(0, np.nan)
    )
    frame = frame.sort_values("Progress", ascending=False)
    leader_progress = frame.iloc[0]["Progress"]
    leader_driver = frame.iloc[0]["Driver"]
    leader_lap_num = int(frame.iloc[0]["LapNumber"])

    leader_lap_time = lap_time_lookup.get(leader_driver, {}).get(leader_lap_num, np.nan)
    fallback_lap_time = np.nanmean([
        t for ldict in lap_time_lookup.values() for t in ldict.values()
    ]) or 90.0
    base_lap_time = leader_lap_time if not np.isnan(leader_lap_time) else fallback_lap_time

    tracker_rows = []
    for pos, (_, row) in enumerate(frame.iterrows(), start=1):
        progress_gap = leader_progress - row["Progress"]
        gap_seconds = progress_gap * base_lap_time

        if pos == 1:
            gap_to_ahead = 0.0
        else:
            prev_row = frame.iloc[pos - 2]
            ahead_gap = prev_row["Progress"] - row["Progress"]
            gap_to_ahead = ahead_gap * base_lap_time

        tracker_rows.append(
            {
                "Pos": pos,
                "Driver": row["Driver"],
                "Lap": int(row["LapNumber"]),
                "Lap Time (s)": round(float(row["TimeIntoLap"]), 2),
                "Gap Leader (s)": round(gap_seconds, 2),
                "Gap Ahead (s)": round(gap_to_ahead, 2),
            }
        )

    return pd.DataFrame(tracker_rows)


def get_frame_at_time(positions: pd.DataFrame, t: float):
    """Slice the resampled telemetry at a given timestamp."""
    idx = positions.iloc[(positions["SessionTime"] - t).abs().groupby(positions["Driver"]).idxmin()]
    return idx


# ---------------------------
# UI helpers
# ---------------------------

def render_track(frame: pd.DataFrame, colors: Dict[str, str]):
    fig = px.scatter(
        frame,
        x="X",
        y="Y",
        color="Driver",
        color_discrete_map=colors,
        hover_data={"Driver": True, "LapNumber": True, "Speed": True},
        size_max=12,
    )
    fig.update_traces(marker={"size": 10, "opacity": 0.9})
    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def initialize_state():
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "playback_time" not in st.session_state:
        st.session_state.playback_time = 0.0
    if "last_update" not in st.session_state:
        st.session_state.last_update = time.time()
    if "playback_speed" not in st.session_state:
        st.session_state.playback_speed = 1.0
    if "selected_lap" not in st.session_state:
        st.session_state.selected_lap = 1


# ---------------------------
# Main app
# ---------------------------

def main():
    initialize_state()

    st.title("🏎️ F1 Race Playback")
    st.markdown(
        "Use the controls in the sidebar to choose a season and race, then press **Play** to animate the race."
    )

    with st.sidebar:
        st.header("Race Selection")
        season = st.selectbox("Season", list(range(2020, 2025)), index=4)
        schedule = fastf1.get_event_schedule(season)
        race_options = schedule.loc[schedule["EventFormat"] == "conventional", "EventName"].tolist()
        race_name = st.selectbox("Grand Prix", race_options)

        session_loaded = "session_data" in st.session_state and st.session_state.get("loaded_key") == (season, race_name)
        load_btn = st.button("Load race", type="primary")

    if load_btn or session_loaded:
        if load_btn:
            with st.spinner("Loading session and telemetry…"):
                event_row = schedule[schedule["EventName"] == race_name].iloc[0]
                round_number = int(event_row["RoundNumber"])
                (
                    session,
                    positions,
                    colors,
                    lap_time_lookup,
                    lap_jump,
                ) = load_session_data(season, round_number)
                st.session_state.session_data = (session, positions, colors, lap_time_lookup, lap_jump)
                st.session_state.loaded_key = (season, race_name)
                st.session_state.playback_time = 0.0
                st.session_state.last_update = time.time()
        else:
            session, positions, colors, lap_time_lookup, lap_jump = st.session_state.session_data

        total_time = positions["SessionTime"].max()

        with st.sidebar:
            st.header("Playback")
            cols = st.columns(2)
            play_label = "Pause" if st.session_state.playing else "Play"
            if cols[0].button(play_label, use_container_width=True):
                st.session_state.playing = not st.session_state.playing
                st.session_state.last_update = time.time()

            if cols[1].button("Stop", use_container_width=True):
                st.session_state.playing = False
                st.session_state.playback_time = 0.0

            st.session_state.playback_speed = st.select_slider(
                "Speed",
                options=[0.5, 1.0, 2.0, 4.0],
                value=st.session_state.playback_speed,
            )

            # Lap skip
            if lap_jump:
                st.session_state.selected_lap = st.slider(
                    "Skip to lap", min_value=1, max_value=max(lap_jump.keys()), value=st.session_state.selected_lap
                )
                if st.button("Jump to lap"):
                    jump_time = lap_jump.get(st.session_state.selected_lap, 0.0)
                    st.session_state.playback_time = jump_time
                    st.session_state.last_update = time.time()

            manual_time = st.slider(
                "Playback time (s)", 0.0, float(total_time), value=float(st.session_state.playback_time), step=1.0
            )
            if not st.session_state.playing:
                st.session_state.playback_time = manual_time

        # Update playback time when playing
        if st.session_state.playing:
            now = time.time()
            delta = now - st.session_state.last_update
            st.session_state.last_update = now
            st.session_state.playback_time = min(
                st.session_state.playback_time + delta * st.session_state.playback_speed,
                float(total_time),
            )
            if st.session_state.playback_time >= total_time:
                st.session_state.playing = False

        current_frame = get_frame_at_time(positions, st.session_state.playback_time)
        order_frame = compute_order_frame(current_frame, lap_time_lookup)

        track_col, tracker_col = st.columns([2, 1])
        with track_col:
            st.subheader("Track map")
            st.plotly_chart(render_track(current_frame, colors), use_container_width=True)
            st.caption(
                f"Time: {st.session_state.playback_time:.1f}s • Lap: {int(current_frame['LapNumber'].max())}"
            )

        with tracker_col:
            st.subheader("Driver tracker")
            st.dataframe(order_frame, use_container_width=True, hide_index=True)

        # Trigger rerun for animation
        if st.session_state.playing:
            time.sleep(0.05)
            st.experimental_rerun()

    else:
        st.info("Select a race in the sidebar and press **Load race** to begin.")


if __name__ == "__main__":
    main()
