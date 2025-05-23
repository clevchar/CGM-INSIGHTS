import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

sns.set(style="whitegrid")
st.set_page_config(page_title="CGM Insights Dashboard", layout="wide")
st.title("📈 Glucose Insights from CGM Data")

# Add user input for glucose range
st.sidebar.header("Customize Glucose Range")
target_low = st.sidebar.slider("Low Glucose Threshold (mg/dL)", min_value=40, max_value=120, value=70)
target_high = st.sidebar.slider("High Glucose Threshold (mg/dL)", min_value=140, max_value=250, value=180)

file = st.file_uploader("Upload your CGM CSV file", type=["csv"])

@st.cache_data
def cached_load(file):
    return load_glucose_data(file)


def load_glucose_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    df['Timestamp'] = pd.to_datetime(df['Timestamp (YYYY-MM-DDThh:mm:ss)'], errors='coerce')
    df['Glucose Value (mg/dL)'] = pd.to_numeric(df['Glucose Value (mg/dL)'], errors='coerce')
    df = df.dropna(subset=['Timestamp', 'Glucose Value (mg/dL)'])
    df = df[df['Event Type'] == 'EGV']
    df = df.sort_values('Timestamp')
    df.set_index('Timestamp', inplace=True)
    return df

def plot_daily_glucose_profiles_enhanced(df):
    df_daily = df.copy().reset_index()
    df_daily['Rounded Time'] = df_daily['Timestamp'].dt.round('5min').dt.time
    df_daily['Date'] = df_daily['Timestamp'].dt.date
    pivot = df_daily.pivot_table(index='Rounded Time', columns='Date', values='Glucose Value (mg/dL)')
    arbitrary_date = pd.Timestamp('2025-01-01')
    times = [pd.Timestamp.combine(arbitrary_date, t) for t in pivot.index]
    daily_mean = pivot.mean(axis=1, skipna=True).copy()
    smoothed_mean = savgol_filter(daily_mean.interpolate().bfill().ffill(), window_length=5, polyorder=2)
    fig, ax = plt.subplots(figsize=(15, 6))
    for col in pivot.columns:
        day_vals = pivot[col]
        if day_vals.isnull().all(): continue
        interpolated = day_vals.interpolate().bfill().ffill()
        diff = abs(interpolated - smoothed_mean)
        darkness = max(0.1, min(1.0, 1.0 - diff.mean() / 100))
        for i in range(len(times) - 1):
            val, next_val = interpolated.iloc[i], interpolated.iloc[i+1]
            if pd.isna(val) or pd.isna(next_val): continue
            color = "gray"
            if val > target_high or val < target_low: color = "red"
            elif val > target_high - 20 or val < target_low + 20: color = "gold"
            ax.plot(times[i:i+2], [val, next_val], color=color if color != "gray" else str(0.3 * darkness), linewidth=0.6, linestyle='--')
    ax.plot(times, smoothed_mean, color='darkblue', linewidth=3, label='Smoothed Daily Average')
    ax.set_title("Daily Glucose Patterns (Color-Coded Extremes + Proximity-Weighted Lines)")
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.yaxis.set_ticks(range(0, int(pivot.max().max()) + 50, 25))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    fig.tight_layout()
    st.pyplot(fig)

def plot_glucose_weeks_concatenated(df):
    df = df.copy().reset_index()
    df['Week'] = df['Timestamp'].dt.isocalendar().week
    df['Day'] = df['Timestamp'].dt.dayofweek
    df['Rounded Time'] = df['Timestamp'].dt.round('5min').dt.time
    grouped = df.groupby(['Week', 'Day', 'Rounded Time'])['Glucose Value (mg/dL)'].mean()
    week_profiles = {}
    for week in grouped.index.get_level_values(0).unique():
        values = []
        for day in range(7):
            try:
                times = grouped.loc[week, day]
                values.extend(times.values)
            except: continue
        if len(values) > 200:
            week_profiles[week] = pd.Series(values)
    max_len = max(len(s) for s in week_profiles.values())
    all_weeks_df = pd.DataFrame({k: v.reindex(range(max_len)) for k, v in week_profiles.items()})
    avg_week = all_weeks_df.mean(axis=1).interpolate().bfill().ffill()
    smoothed_avg = savgol_filter(avg_week, window_length=25, polyorder=2)
    fig, ax = plt.subplots(figsize=(16, 6))
    for col in all_weeks_df.columns:
        week_vals = all_weeks_df[col].interpolate().bfill().ffill()
        diff = abs(week_vals - smoothed_avg)
        darkness = max(0.1, min(1.0, 1.0 - diff.mean() / 100))
        for i in range(len(week_vals) - 1):
            val, next_val = week_vals.iloc[i], week_vals.iloc[i + 1]
            if pd.isna(val) or pd.isna(next_val): continue
            color = "gray"
            if val > target_high or val < target_low: color = "red"
            elif val > target_high - 20 or val < target_low + 20: color = "gold"
            ax.plot([i, i + 1], [val, next_val], color=color if color != "gray" else str(0.3 * darkness), linewidth=0.6, linestyle='--')
    ax.plot(range(len(smoothed_avg)), smoothed_avg, color='darkblue', linewidth=3, label='Smoothed Weekly Average')
    weekday_ticks = np.linspace(0, len(smoothed_avg), 8)
    weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', '']
    ax.set_xticks(weekday_ticks)
    ax.set_xticklabels(weekday_labels)
    ax.set_yticks(range(0, int(all_weeks_df.max().max()) + 50, 25))
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title("Full Weekly Glucose Curves (Each Line = 1 Week)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)

def plot_weekday_glucose_profiles_enhanced(df):
    df_day = df.copy().reset_index()
    df_day['Rounded Time'] = df_day['Timestamp'].dt.round('5min').dt.time
    df_day['Weekday'] = df_day['Timestamp'].dt.day_name()
    df_day['Date'] = df_day['Timestamp'].dt.date
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in weekday_order:
        df_subset = df_day[df_day['Weekday'] == day]
        pivot = df_subset.pivot_table(index='Rounded Time', columns='Date', values='Glucose Value (mg/dL)')
        if pivot.empty:
            continue
        arbitrary_date = pd.Timestamp('2025-01-01')
        times = [pd.Timestamp.combine(arbitrary_date, t) for t in pivot.index]
        daily_mean = pivot.mean(axis=1, skipna=True)
        smoothed_mean = savgol_filter(daily_mean.interpolate().bfill().ffill(), window_length=5, polyorder=2)
        fig, ax = plt.subplots(figsize=(15, 6))
        for col in pivot.columns:
            day_vals = pivot[col]
            if day_vals.isnull().all():
                continue
            interpolated = day_vals.interpolate().bfill().ffill()
            diff = abs(interpolated - smoothed_mean)
            darkness = max(0.1, min(1.0, 1.0 - diff.mean() / 100))
            for i in range(len(times) - 1):
                val = interpolated.iloc[i]
                next_val = interpolated.iloc[i + 1]
                if pd.isna(val) or pd.isna(next_val): continue
                color = "gray"
                if val > target_high or val < target_low:
                    color = "red"
                elif val > target_high - 20 or val < target_low + 20:
                    color = "gold"
                ax.plot(times[i:i + 2], [val, next_val],
                        color=color if color != "gray" else str(0.3 * darkness),
                        linewidth=0.6, linestyle='--')
        ax.plot(times, smoothed_mean, color='darkblue', linewidth=3, label='Smoothed Average')
        ax.set_title(f"Glucose Profiles on {day} (Color-Coded Extremes)")
        ax.set_xlabel("Time of Day")
        ax.set_ylabel("Glucose (mg/dL)")
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax.set_yticks(range(0, int(pivot.max().max()) + 50, 25))
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
if file:
    df = cached_load(file)
    df_interp = df[["Glucose Value (mg/dL)"]].resample("5min").mean().interpolate()
    glucose = df_interp["Glucose Value (mg/dL)"]

    # --- Metrics ---
    avg_glucose = round(glucose.mean(), 2)
    gmi = round(3.31 + 0.02392 * glucose.mean(), 2)
    tir = round(100 * glucose.between(target_low, target_high).mean(), 2)
    above = round(100 * (glucose > target_high).mean(), 2)
    below = round(100 * (glucose < target_low).mean(), 2)
    std = round(glucose.std(), 2)
    var = round(glucose.var(), 2)

    # Spike/Dip Hours
    df_interp["Hour"] = df_interp.index.hour
    high_counts = df_interp[glucose > 200]["Hour"].value_counts().sort_values(ascending=False).head(5).to_dict()
    low_counts = df_interp[glucose < 80]["Hour"].value_counts().sort_values(ascending=False).head(5).to_dict()

    # Stability
    hourly_stats = df_interp.groupby(df_interp.index.hour)["Glucose Value (mg/dL)"].agg(['mean', 'std'])
    stable_hours = hourly_stats.sort_values("std").head(5).to_dict(orient="index")
    volatile_hours = hourly_stats.sort_values("std", ascending=False).head(5).to_dict(orient="index")

    # Weekly averages
    df_interp["Day"] = df_interp.index.day_name()
    week_avg = df_interp.groupby("Day")["Glucose Value (mg/dL)"].mean().sort_values()
    best_days = week_avg.head(3).to_dict()
    all_days = week_avg.to_dict()

    # Meal estimation via peak acceleration
    smoothed = glucose.rolling(window=5, center=True).mean().dropna()
    peaks, _ = find_peaks(smoothed, distance=24*3)
    peak_times = smoothed.iloc[peaks].sort_values(ascending=False).head(3).index
    estimated_mealtimes = [(t - pd.Timedelta(minutes=40)).time() for t in peak_times]

    # Spike & dip magnitude
    df_interp["change"] = glucose.diff()
    df_interp["spike_flag"] = df_interp["change"] > 15
    df_interp["dip_flag"] = df_interp["change"] < -15

    spike_mags, dip_mags = [], []
    current_spike, current_dip = [], []
    for _, row in df_interp.iterrows():
        if row["spike_flag"]:
            current_spike.append(row["Glucose Value (mg/dL)"])
        elif current_spike:
            spike_mags.append(max(current_spike) - current_spike[0])
            current_spike = []
        if row["dip_flag"]:
            current_dip.append(row["Glucose Value (mg/dL)"])
        elif current_dip:
            dip_mags.append(current_dip[0] - min(current_dip))
            current_dip = []

    avg_spike_mag = round(np.mean([m for m in spike_mags if m >= 15]), 2) if spike_mags else None
    avg_dip_mag = round(np.mean([m for m in dip_mags if m >= 15]), 2) if dip_mags else None

    # Return to range
    return_durations = []
    in_range = False
    start_time = None
    for time, val in glucose.items():
        if not in_range and (val < 70 or val > 180):
            start_time = time
            in_range = True
        elif in_range and 70 <= val <= 180:
            return_durations.append((time - start_time).total_seconds() / 60.0)
            in_range = False
    avg_return_time = round(np.mean(return_durations), 2) if return_durations else None

    # Time summaries
    total_minutes = len(glucose) * 5
    above_minutes = len(glucose[glucose > 180]) * 5
    below_minutes = len(glucose[glucose < 70]) * 5
    in_range_minutes = total_minutes - above_minutes - below_minutes

    st.subheader("📊 Metrics")
    st.json({
        "Average Glucose (mg/dL)": avg_glucose,
        "GMI (A1c Estimate)": gmi,
        f"Time in Range ({target_low}-{target_high} mg/dL) (%)": tir,
        f"Above Range (> {target_high} mg/dL) (%)": above,
        f"Below Range (< {target_low} mg/dL) (%)": below,
        "Standard Deviation (mg/dL)": std,
        "Variance (mg/dL²)": var,
        "Top 5 Spike Hours (>200 mg/dL)": high_counts,
        "Top 5 Dip Hours (<80 mg/dL)": low_counts,
        "Most Stable Hours": stable_hours,
        "Most Volatile Hours": volatile_hours,
        "Best Days of the Week": best_days,
        "All Weekday Averages": all_days,
        "Estimated Meal Times": estimated_mealtimes,
        "Average Spike Magnitude (>15 mg/dL)": avg_spike_mag,
        "Average Dip Magnitude (>15 mg/dL)": avg_dip_mag,
        "Avg Time to Return to Range (min)": avg_return_time,
        "Total Time In Range (min)": in_range_minutes,
        "Total Time Above Range (min)": above_minutes,
        "Total Time Below Range (min)": below_minutes,
    })
    st.markdown("----")
    # Full timeline plot
    st.subheader("📈 Full Glucose Timeline")
    fig1, ax1 = plt.subplots(figsize=(15, 6))
    df["Glucose Value (mg/dL)"].plot(marker='.', linestyle='', markersize=2, alpha=0.7, ax=ax1)
    ax1.axhline(target_low, color='red', linestyle='--', label=f'Low Threshold ({target_low} mg/dL)')
    ax1.axhline(target_high, color='orange', linestyle='--', label=f'High Threshold ({target_high} mg/dL)')
    ax1.set_title("Glucose Levels Over Entire Timeline")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Glucose (mg/dL)")
    ax1.set_yticks(range(0, int(df['Glucose Value (mg/dL)'].max())+50, 50))
    ax1.legend()
    fig1.tight_layout()
    st.pyplot(fig1)
    st.markdown("----")
    with st.expander("📆 Show Daily Glucose Profiles"):
        plot_daily_glucose_profiles_enhanced(df)

    with st.expander("📅 Show Weekly Glucose Profiles"):
        plot_glucose_weeks_concatenated(df)
    with st.expander("📚 Show Glucose Profiles by Weekday"):
        plot_weekday_glucose_profiles_enhanced(df)
else:
    st.warning("📁 Please upload a CGM CSV file to begin.")
