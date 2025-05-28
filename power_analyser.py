import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from fitparse import FitFile
import numpy as np
from datetime import datetime, timezone, timedelta

# Attempt to import timezonefinder, provide instructions if missing
try:
    from timezonefinder import TimezoneFinder
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="FIT File Power Analyzer")

# --- Helper Functions ---

def parse_fit_file(uploaded_file_bytes):
    """Parses a .fit file and extracts timestamp, power, and first GPS data."""
    if uploaded_file_bytes is None:
        return None, None, None # df, first_lat, first_lon
    
    first_lat, first_lon = None, None
    try:
        fitfile = FitFile(uploaded_file_bytes)
        records = []
        for record in fitfile.get_messages('record'):
            data = {}
            timestamp = None
            power = None
            current_lat, current_lon = None, None

            for field in record:
                if field.name == 'timestamp':
                    timestamp = field.value
                elif field.name == 'power':
                    power = field.value
                elif field.name == 'position_lat' and field.value is not None:
                    current_lat = field.value * (180.0 / 2**31) # Semicircles to degrees
                elif field.name == 'position_long' and field.value is not None:
                    current_lon = field.value * (180.0 / 2**31) # Semicircles to degrees
            
            if timestamp is not None and power is not None:
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                records.append({'timestamp': timestamp, 'power': power})

                if first_lat is None and current_lat is not None and current_lon is not None:
                    first_lat = current_lat
                    first_lon = current_lon
        
        if not records:
            st.warning(f"No power data records found in {uploaded_file_bytes.name}.")
            return None, None, None

        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        df = df.resample('1S').mean() 
        df['power'] = df['power'].interpolate(method='linear')
        df = df.dropna(subset=['power'])
        
        if df.empty:
            st.warning(f"Resampled data is empty for {uploaded_file_bytes.name}. Check file content.")
            return None, first_lat, first_lon # Still return lat/lon if found
            
        return df, first_lat, first_lon
    except Exception as e:
        st.error(f"Error parsing FIT file {getattr(uploaded_file_bytes, 'name', 'unknown')}: {e}")
        return None, None, None


def process_data(df_raw, stretch_factor, shift_seconds, moving_avg_window_seconds):
    if df_raw is None or df_raw.empty:
        return None
    df_processed = df_raw.copy()
    if stretch_factor != 1.0 and not df_processed.empty:
        start_time_abs = df_processed.index[0]
        elapsed_seconds_orig = (df_processed.index - start_time_abs).total_seconds().to_numpy()
        power_values_orig = df_processed['power'].to_numpy()
        elapsed_seconds_stretched = elapsed_seconds_orig * stretch_factor
        if len(elapsed_seconds_stretched) > 1:
            target_stretched_elapsed_seconds = np.arange(0, elapsed_seconds_stretched.max() + 1, 1)
            if len(target_stretched_elapsed_seconds) > 0:
                sort_indices = np.argsort(elapsed_seconds_stretched)
                elapsed_seconds_stretched_sorted = elapsed_seconds_stretched[sort_indices]
                power_values_orig_sorted = power_values_orig[sort_indices]
                unique_stretched_seconds, unique_indices = np.unique(elapsed_seconds_stretched_sorted, return_index=True)
                unique_power_values = power_values_orig_sorted[unique_indices]
                if len(unique_stretched_seconds) > 1:
                    interpolated_power = np.interp(target_stretched_elapsed_seconds, unique_stretched_seconds, unique_power_values)
                    new_indices = [start_time_abs + pd.Timedelta(seconds=s) for s in target_stretched_elapsed_seconds]
                    df_processed = pd.DataFrame({'power': interpolated_power}, index=pd.DatetimeIndex(new_indices))
                else:
                    df_processed = pd.DataFrame(columns=['power']).set_index(pd.DatetimeIndex([]))
            else:
                df_processed = pd.DataFrame(columns=['power']).set_index(pd.DatetimeIndex([]))
        elif len(elapsed_seconds_stretched) == 1:
            new_indices = [start_time_abs + pd.Timedelta(seconds=elapsed_seconds_stretched[0])]
            df_processed = pd.DataFrame({'power': power_values_orig}, index=pd.DatetimeIndex(new_indices))
        else:
             df_processed = pd.DataFrame(columns=['power']).set_index(pd.DatetimeIndex([]))
    if shift_seconds != 0 and not df_processed.empty:
        df_processed.index = df_processed.index + pd.Timedelta(seconds=shift_seconds)
    if moving_avg_window_seconds > 1 and not df_processed.empty:
        df_processed['power'] = df_processed['power'].rolling(window=moving_avg_window_seconds, center=True, min_periods=1).mean()
    if not df_processed.empty:
        df_processed['elapsed_seconds'] = (df_processed.index - df_processed.index[0]).total_seconds()
    else:
        if 'power' not in df_processed.columns: df_processed['power'] = pd.Series(dtype='float64')
        if 'elapsed_seconds' not in df_processed.columns: df_processed['elapsed_seconds'] = pd.Series(dtype='float64')
        if df_processed.index.name != 'timestamp': df_processed.index = pd.DatetimeIndex(df_processed.index, name='timestamp')
    return df_processed

def format_seconds_to_hhmmss(seconds):
    if pd.isna(seconds) or not np.isfinite(seconds): return "00:00:00"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"

# --- Initialize Session State ---
default_states = {
    'df1_raw': None, 'df2_raw': None,
    'df1_processed': None, 'df2_processed': None,
    'file1_name': "File 1", 'file2_name': "File 2",
    'plot_type': "Line", 'x_axis_type': "Elapsed Time",
    'y_axis_min': 0.0, 'y_axis_max': 500.0,
    'y_grid_spacing': 100.0, 'show_y_grid': True,
    'default_y_max_calculated': False,
    'moving_avg_window': 1,
    'f1_color': '#FF0000', 'f1_shift': 0.0, 'f1_stretch': 1.0,
    'f2_color': '#0000FF', 'f2_shift': 0.0, 'f2_stretch': 1.0,
    'activity_timezone_str': None, # For storing timezone like 'Australia/Sydney'
    'show_difference_plot': False,
}
for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- UI Layout ---
st.title("🚴💨 FIT File Power Analyzer")
st.markdown("Compare power data from two .fit files. Use the 'Autoscale' (home icon) in the plot's modebar to reset zoom.")

if not TF_AVAILABLE:
    st.warning("TimezoneFinder library not found. Timezone detection from GPS will not work. "
               "Please install it: `pip install timezonefinder[numba]`")

col_uploader1, col_uploader2 = st.columns(2)
with col_uploader1:
    uploaded_file1 = st.file_uploader("Upload first .fit file", type="fit", key="file1_uploader")
with col_uploader2:
    uploaded_file2 = st.file_uploader("Upload second .fit file", type="fit", key="file2_uploader")

# Data loading and timezone detection
if uploaded_file1 and (st.session_state.df1_raw is None or st.session_state.file1_name != uploaded_file1.name):
    st.session_state.df1_raw, f1_lat, f1_lon = parse_fit_file(uploaded_file1)
    st.session_state.file1_name = uploaded_file1.name
    st.session_state.default_y_max_calculated = False
    if f1_lat is not None and f1_lon is not None and TF_AVAILABLE:
        tf = TimezoneFinder()
        tz_str = tf.timezone_at(lng=f1_lon, lat=f1_lat)
        if tz_str and st.session_state.activity_timezone_str is None: # Set if not already set by other file
             st.session_state.activity_timezone_str = tz_str

if uploaded_file2 and (st.session_state.df2_raw is None or st.session_state.file2_name != uploaded_file2.name):
    st.session_state.df2_raw, f2_lat, f2_lon = parse_fit_file(uploaded_file2)
    st.session_state.file2_name = uploaded_file2.name
    st.session_state.default_y_max_calculated = False
    if f2_lat is not None and f2_lon is not None and TF_AVAILABLE:
        tf = TimezoneFinder()
        tz_str = tf.timezone_at(lng=f2_lon, lat=f2_lat)
        if tz_str and st.session_state.activity_timezone_str is None:
             st.session_state.activity_timezone_str = tz_str

if not st.session_state.default_y_max_calculated and (st.session_state.df1_raw is not None or st.session_state.df2_raw is not None):
    max_p1 = st.session_state.df1_raw['power'].max() if st.session_state.df1_raw is not None and not st.session_state.df1_raw.empty else 0
    max_p2 = st.session_state.df2_raw['power'].max() if st.session_state.df2_raw is not None and not st.session_state.df2_raw.empty else 0
    global_max_power = max(max_p1, max_p2, 100) # Ensure at least 100W for scale
    st.session_state.y_axis_max = float(np.ceil(global_max_power / 100.0) * 100.0)
    st.session_state.y_axis_min = 0.0
    st.session_state.default_y_max_calculated = True

st.session_state.df1_processed = process_data(st.session_state.df1_raw, st.session_state.f1_stretch, st.session_state.f1_shift, st.session_state.moving_avg_window)
st.session_state.df2_processed = process_data(st.session_state.df2_raw, st.session_state.f2_stretch, st.session_state.f2_shift, st.session_state.moving_avg_window)

# --- Plotting Area ---
plot_placeholder = st.empty()
fig = go.Figure()
plot_mode_map = {"Line": "lines", "Scatter": "markers", "Line and Scatter": "lines+markers"}
current_plot_mode = plot_mode_map.get(st.session_state.plot_type, "lines")
scatter_marker_config = dict(size=3) if "Scatter" in st.session_state.plot_type else {}

x_axis_is_elapsed_time = st.session_state.x_axis_type == "Elapsed Time"

# Add traces
hovertemplate_custom = "%{y:.0f}W<extra></extra>"

if st.session_state.df1_processed is not None and not st.session_state.df1_processed.empty:
    x_data1 = st.session_state.df1_processed['elapsed_seconds'] if x_axis_is_elapsed_time else \
              (st.session_state.df1_processed.index.tz_convert(st.session_state.activity_timezone_str)
               if st.session_state.activity_timezone_str and not x_axis_is_elapsed_time
               else st.session_state.df1_processed.index)
    fig.add_trace(go.Scatter(
        x=x_data1, y=st.session_state.df1_processed['power'], mode=current_plot_mode, 
        name=st.session_state.file1_name, line=dict(color=st.session_state.f1_color, width=1),
        marker=scatter_marker_config, hovertemplate=hovertemplate_custom
    ))

if st.session_state.df2_processed is not None and not st.session_state.df2_processed.empty:
    x_data2 = st.session_state.df2_processed['elapsed_seconds'] if x_axis_is_elapsed_time else \
              (st.session_state.df2_processed.index.tz_convert(st.session_state.activity_timezone_str)
               if st.session_state.activity_timezone_str and not x_axis_is_elapsed_time
               else st.session_state.df2_processed.index)
    fig.add_trace(go.Scatter(
        x=x_data2, y=st.session_state.df2_processed['power'], mode=current_plot_mode, 
        name=st.session_state.file2_name, line=dict(color=st.session_state.f2_color, width=1),
        marker=scatter_marker_config, hovertemplate=hovertemplate_custom
    ))

# Difference Plot
if st.session_state.show_difference_plot and \
   st.session_state.df1_processed is not None and not st.session_state.df1_processed.empty and \
   st.session_state.df2_processed is not None and not st.session_state.df2_processed.empty:
    
    merged_df = pd.merge(st.session_state.df1_processed[['power', 'elapsed_seconds']], 
                         st.session_state.df2_processed[['power', 'elapsed_seconds']], 
                         left_index=True, right_index=True, suffixes=('_1', '_2'), how='inner')
    if not merged_df.empty:
        merged_df['power_diff'] = merged_df['power_1'] - merged_df['power_2']
        
        # Determine x-data for difference plot
        if x_axis_is_elapsed_time:
            # Assuming 'elapsed_seconds_1' can represent the common timeline for the merged data
            # This might need adjustment if start times are very different post-processing
            x_data_diff = merged_df['elapsed_seconds_1'] 
        else: # Time of Day
            x_data_diff = merged_df.index
            if st.session_state.activity_timezone_str:
                x_data_diff = x_data_diff.tz_convert(st.session_state.activity_timezone_str)
        
        fig.add_trace(go.Scatter(
            x=x_data_diff, y=merged_df['power_diff'], mode='lines',
            name='Difference (F1-F2)', line=dict(color='green', width=1),
            hovertemplate="%{y:.0f}W (Diff)<extra></extra>"
        ))

# Configure x-axis
current_tz_for_title = st.session_state.activity_timezone_str or "UTC"
x_title = "Elapsed Time (HH:MM:SS)" if x_axis_is_elapsed_time else f"Time of Day ({current_tz_for_title})"

if x_axis_is_elapsed_time:
    min_sec_all, max_sec_all = float('inf'), float('-inf')
    if st.session_state.df1_processed is not None and not st.session_state.df1_processed.empty:
        min_sec_all = min(min_sec_all, st.session_state.df1_processed['elapsed_seconds'].min())
        max_sec_all = max(max_sec_all, st.session_state.df1_processed['elapsed_seconds'].max())
    if st.session_state.df2_processed is not None and not st.session_state.df2_processed.empty:
        min_sec_all = min(min_sec_all, st.session_state.df2_processed['elapsed_seconds'].min())
        max_sec_all = max(max_sec_all, st.session_state.df2_processed['elapsed_seconds'].max())

    if np.isfinite(min_sec_all) and np.isfinite(max_sec_all) and max_sec_all > min_sec_all:
        tickvals = np.linspace(min_sec_all, max_sec_all, num=10)
        ticktext = [format_seconds_to_hhmmss(s) for s in tickvals]
        fig.update_xaxes(title_text=x_title, tickvals=tickvals, ticktext=ticktext)
    else:
        fig.update_xaxes(title_text=x_title, nticks=10)
else:
    fig.update_xaxes(title_text=x_title, nticks=10)

fig.update_layout(
    height=550,
    yaxis_title="Power (Watts)",
    yaxis_range=[st.session_state.y_axis_min, st.session_state.y_axis_max],
    yaxis_dtick=st.session_state.y_grid_spacing if st.session_state.show_y_grid else None,
    yaxis_showgrid=st.session_state.show_y_grid,
    legend_title_text='Files',
    title="Power Comparison",
    margin=dict(t=50, b=50),
    hovermode='x unified', # Vertical line hover
    xaxis_showline=True, yaxis_showline=True, # Chart border
    xaxis_linewidth=1, yaxis_linewidth=1,
    xaxis_linecolor='grey', yaxis_linecolor='grey'
)
plot_placeholder.plotly_chart(fig, use_container_width=True)

# --- Summary Statistics Display ---
st.markdown("---")
st.subheader("📊 Summary of Processed Data (Entire Duration)")
summary_cols = st.columns(2)
avg_power1, duration_str1 = "N/A", "N/A"
if st.session_state.df1_processed is not None and not st.session_state.df1_processed.empty:
    avg_power1 = f"{st.session_state.df1_processed['power'].mean():.1f} W"
    if not st.session_state.df1_processed['elapsed_seconds'].empty:
        total_seconds1 = st.session_state.df1_processed['elapsed_seconds'].iloc[-1] - st.session_state.df1_processed['elapsed_seconds'].iloc[0]
        duration_str1 = format_seconds_to_hhmmss(total_seconds1)
    else: duration_str1 = "00:00:00"
with summary_cols[0]:
    st.markdown(f"**{st.session_state.file1_name}:**")
    st.markdown(f"<span style='color:{st.session_state.f1_color};'>Avg Power: {avg_power1}</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{st.session_state.f1_color};'>Total Duration: {duration_str1}</span>", unsafe_allow_html=True)

avg_power2, duration_str2 = "N/A", "N/A"
if st.session_state.df2_processed is not None and not st.session_state.df2_processed.empty:
    avg_power2 = f"{st.session_state.df2_processed['power'].mean():.1f} W"
    if not st.session_state.df2_processed['elapsed_seconds'].empty:
        total_seconds2 = st.session_state.df2_processed['elapsed_seconds'].iloc[-1] - st.session_state.df2_processed['elapsed_seconds'].iloc[0]
        duration_str2 = format_seconds_to_hhmmss(total_seconds2)
    else: duration_str2 = "00:00:00"
with summary_cols[1]:
    st.markdown(f"**{st.session_state.file2_name}:**")
    st.markdown(f"<span style='color:{st.session_state.f2_color};'>Avg Power: {avg_power2}</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{st.session_state.f2_color};'>Total Duration: {duration_str2}</span>", unsafe_allow_html=True)

# --- Controls Area ---
st.markdown("---")
st.header("⚙️ Analysis Controls")
control_col1, control_col2, control_col3 = st.columns(3)

with control_col1:
    st.subheader("Global Plot Settings")
    
    plot_type_changed = False
    prev_plot_type = st.session_state.plot_type
    st.session_state.plot_type = st.selectbox("Plot Type:", options=["Line", "Scatter", "Line and Scatter"], 
                                              index=["Line", "Scatter", "Line and Scatter"].index(st.session_state.plot_type),
                                              key='sb_plot_type_k')
    if prev_plot_type != st.session_state.plot_type: plot_type_changed = True

    x_axis_type_changed = False
    prev_x_axis_type = st.session_state.x_axis_type
    st.session_state.x_axis_type = st.radio("X-Axis Display:", options=["Elapsed Time", "Time of Day"],
                                             index=["Elapsed Time", "Time of Day"].index(st.session_state.x_axis_type),
                                             key='rb_x_axis_k')
    if prev_x_axis_type != st.session_state.x_axis_type: x_axis_type_changed = True
    
    mov_avg_changed = False
    prev_mov_avg = st.session_state.moving_avg_window
    st.session_state.moving_avg_window = st.number_input("Moving Average Window (seconds, 1 for none):", 
                                                         min_value=1, max_value=600, value=st.session_state.moving_avg_window, 
                                                         step=1, key='ni_mov_avg_k')
    if prev_mov_avg != st.session_state.moving_avg_window: mov_avg_changed = True

    show_diff_changed = False
    prev_show_diff = st.session_state.show_difference_plot
    st.session_state.show_difference_plot = st.toggle("Show Power Difference (File1 - File2)", 
                                                      value=st.session_state.show_difference_plot, key='tgl_diff_plot_k')
    if prev_show_diff != st.session_state.show_difference_plot: show_diff_changed = True
    
    st.markdown("###### Y-Axis Scale & Grid")
    y_min_changed, y_max_changed, y_grid_space_changed, y_grid_show_changed = False, False, False, False
    
    if st.button("Reset Y-Axis Scale", key="btn_reset_y_scale"):
        st.session_state.y_axis_min = 0.0
        st.session_state.default_y_max_calculated = False # This will trigger recalc
        st.rerun() # Rerun immediately to apply reset

    prev_y_min = st.session_state.y_axis_min
    st.session_state.y_axis_min = st.number_input("Y-Axis Min (W):", value=float(st.session_state.y_axis_min), key='ni_y_min_k', format="%.1f")
    if prev_y_min != st.session_state.y_axis_min: y_min_changed = True

    prev_y_max = st.session_state.y_axis_max
    st.session_state.y_axis_max = st.number_input("Y-Axis Max (W):", value=float(st.session_state.y_axis_max), 
                                                  min_value=float(st.session_state.y_axis_min) + 1.0, key='ni_y_max_k', format="%.1f")
    if prev_y_max != st.session_state.y_axis_max: y_max_changed = True

    prev_y_grid = st.session_state.y_grid_spacing
    st.session_state.y_grid_spacing = st.number_input("Y-Axis Grid Spacing (W):", min_value=5.0, 
                                                      value=float(st.session_state.y_grid_spacing), step=5.0, key='ni_y_grid_space_k', format="%.1f")
    if prev_y_grid != st.session_state.y_grid_spacing: y_grid_space_changed = True

    prev_show_grid = st.session_state.show_y_grid
    st.session_state.show_y_grid = st.toggle("Show Y-Axis Gridlines", value=st.session_state.show_y_grid, key='tgl_y_grid_k')
    if prev_show_grid != st.session_state.show_y_grid: y_grid_show_changed = True

    if any([plot_type_changed, x_axis_type_changed, mov_avg_changed, show_diff_changed, 
            y_min_changed, y_max_changed, y_grid_space_changed, y_grid_show_changed]):
        st.rerun()

with control_col2:
    st.subheader(f"File 1 Adjustments: {st.session_state.file1_name}")
    f1_color_changed, f1_shift_changed, f1_stretch_changed = False, False, False
    if st.session_state.df1_raw is not None:
        prev_f1_color = st.session_state.f1_color
        st.session_state.f1_color = st.color_picker("Line Color File 1:", st.session_state.f1_color, key='cp_f1_color_k')
        if prev_f1_color != st.session_state.f1_color: f1_color_changed = True

        prev_f1_shift = st.session_state.f1_shift
        st.session_state.f1_shift = st.number_input("Shift File 1 (seconds, +/-):", value=st.session_state.f1_shift, 
                                                    step=0.1, format="%.1f", key='ni_f1_shift_k')
        if prev_f1_shift != st.session_state.f1_shift: f1_shift_changed = True

        prev_f1_stretch = st.session_state.f1_stretch
        st.session_state.f1_stretch = st.number_input("Stretch File 1 (factor):", value=st.session_state.f1_stretch, 
                                                      min_value=0.1, step=0.01, format="%.2f", key='ni_f1_stretch_k')
        if prev_f1_stretch != st.session_state.f1_stretch: f1_stretch_changed = True
        
        if st.button("Reset File 1 Adjustments", key='btn_reset_f1_k'):
            st.session_state.f1_shift = 0.0
            st.session_state.f1_stretch = 1.0
            st.rerun()
        elif any([f1_color_changed, f1_shift_changed, f1_stretch_changed]):
            st.rerun()
    else:
        st.info("Upload File 1 to see adjustment options.")

with control_col3:
    st.subheader(f"File 2 Adjustments: {st.session_state.file2_name}")
    f2_color_changed, f2_shift_changed, f2_stretch_changed = False, False, False
    if st.session_state.df2_raw is not None:
        prev_f2_color = st.session_state.f2_color
        st.session_state.f2_color = st.color_picker("Line Color File 2:", st.session_state.f2_color, key='cp_f2_color_k')
        if prev_f2_color != st.session_state.f2_color: f2_color_changed = True

        prev_f2_shift = st.session_state.f2_shift
        st.session_state.f2_shift = st.number_input("Shift File 2 (seconds, +/-):", value=st.session_state.f2_shift, 
                                                    step=0.1, format="%.1f", key='ni_f2_shift_k')
        if prev_f2_shift != st.session_state.f2_shift: f2_shift_changed = True

        prev_f2_stretch = st.session_state.f2_stretch
        st.session_state.f2_stretch = st.number_input("Stretch File 2 (factor):", value=st.session_state.f2_stretch, 
                                                      min_value=0.1, step=0.01, format="%.2f", key='ni_f2_stretch_k')
        if prev_f2_stretch != st.session_state.f2_stretch: f2_stretch_changed = True

        if st.button("Reset File 2 Adjustments", key='btn_reset_f2_k'):
            st.session_state.f2_shift = 0.0
            st.session_state.f2_stretch = 1.0
            st.rerun()
        elif any([f2_color_changed, f2_shift_changed, f2_stretch_changed]):
            st.rerun()
    else:
        st.info("Upload File 2 to see adjustment options.")

st.markdown("---")
st.caption("Tip: Hover over the graph to see controls like zoom, pan, and autoscale (reset zoom). The vertical line shows power values in a unified tooltip.")
