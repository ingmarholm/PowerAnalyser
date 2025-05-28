import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from fitparse import FitFile
import numpy as np
from datetime import datetime, timezone

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="FIT File Power Analyzer")

# --- Helper Functions ---

def parse_fit_file(uploaded_file_bytes):
    """Parses a .fit file and extracts timestamp and power data."""
    if uploaded_file_bytes is None:
        return None
    try:
        fitfile = FitFile(uploaded_file_bytes)
        records = []
        for record in fitfile.get_messages('record'):
            data = {}
            timestamp = None
            power = None
            for field in record:
                if field.name == 'timestamp':
                    timestamp = field.value
                elif field.name == 'power':
                    power = field.value
            
            if timestamp is not None and power is not None:
                # Convert to UTC if naive, assuming FIT timestamps are usually UTC
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                records.append({'timestamp': timestamp, 'power': power})

        if not records:
            st.warning(f"No power data records found in {uploaded_file_bytes.name}.")
            return None

        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Resample to 1-second intervals and interpolate
        # This ensures consistent data points for comparison and moving averages
        df = df.resample('1S').mean() 
        df['power'] = df['power'].interpolate(method='linear')
        df = df.dropna(subset=['power']) # Drop rows where power couldn't be interpolated (e.g., at ends)
        
        if df.empty:
            st.warning(f"Resampled data is empty for {uploaded_file_bytes.name}. Check file content.")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error parsing FIT file {getattr(uploaded_file_bytes, 'name', 'unknown')}: {e}")
        return None

def process_data(df_raw, stretch_factor, shift_seconds, moving_avg_window_seconds):
    """Applies transformations (stretch, shift, moving average) to the DataFrame."""
    if df_raw is None or df_raw.empty:
        return None

    df_processed = df_raw.copy()

    # 1. Stretch
    if stretch_factor != 1.0 and not df_processed.empty:
        start_time_abs = df_processed.index[0]
        elapsed_seconds_orig = (df_processed.index - start_time_abs).total_seconds().to_numpy()
        power_values_orig = df_processed['power'].to_numpy()

        elapsed_seconds_stretched = elapsed_seconds_orig * stretch_factor
        
        if len(elapsed_seconds_stretched) > 1 :
            # Create a new regular time grid for interpolation on the stretched timeline
            # Target 1-second steps on the new stretched timeline
            target_stretched_elapsed_seconds = np.arange(0, elapsed_seconds_stretched.max() + 1, 1)
            
            if len(target_stretched_elapsed_seconds) > 0:
                 # Ensure elapsed_seconds_stretched is monotonically increasing for np.interp
                sort_indices = np.argsort(elapsed_seconds_stretched)
                elapsed_seconds_stretched_sorted = elapsed_seconds_stretched[sort_indices]
                power_values_orig_sorted = power_values_orig[sort_indices]

                # Remove duplicates in sorted stretched seconds, keeping the first occurrence
                unique_stretched_seconds, unique_indices = np.unique(elapsed_seconds_stretched_sorted, return_index=True)
                unique_power_values = power_values_orig_sorted[unique_indices]

                if len(unique_stretched_seconds) > 1: # Need at least 2 points to interpolate
                    interpolated_power = np.interp(target_stretched_elapsed_seconds, unique_stretched_seconds, unique_power_values)
                    new_indices = [start_time_abs + pd.Timedelta(seconds=s) for s in target_stretched_elapsed_seconds]
                    df_processed = pd.DataFrame({'power': interpolated_power}, index=pd.DatetimeIndex(new_indices))
                else: # Not enough unique points to interpolate after stretch
                    df_processed = pd.DataFrame({'power': [], 'elapsed_seconds': []}) # Empty df
                    df_processed = df_processed.set_index(pd.DatetimeIndex([]))


            else: # No points in target grid
                df_processed = pd.DataFrame({'power': [], 'elapsed_seconds': []}) 
                df_processed = df_processed.set_index(pd.DatetimeIndex([]))
        elif len(elapsed_seconds_stretched) == 1: # Single point data, just scale time
            new_indices = [start_time_abs + pd.Timedelta(seconds=elapsed_seconds_stretched[0])]
            df_processed = pd.DataFrame({'power': power_values_orig}, index=pd.DatetimeIndex(new_indices))
        else: # Empty after initial load
             df_processed = pd.DataFrame({'power': [], 'elapsed_seconds': []})
             df_processed = df_processed.set_index(pd.DatetimeIndex([]))


    # 2. Shift
    if shift_seconds != 0 and not df_processed.empty:
        df_processed.index = df_processed.index + pd.Timedelta(seconds=shift_seconds)

    # 3. Moving Average
    if moving_avg_window_seconds > 1 and not df_processed.empty:
        df_processed['power'] = df_processed['power'].rolling(window=moving_avg_window_seconds, center=True, min_periods=1).mean()

    # 4. Add Elapsed Time Column (relative to its own start, *after* stretch/shift)
    if not df_processed.empty:
        df_processed['elapsed_seconds'] = (df_processed.index - df_processed.index[0]).total_seconds()
    else:
        # Ensure columns exist even if empty for consistent access later
        if 'power' not in df_processed.columns:
             df_processed['power'] = pd.Series(dtype='float64')
        if 'elapsed_seconds' not in df_processed.columns:
            df_processed['elapsed_seconds'] = pd.Series(dtype='float64')


    return df_processed

# --- Initialize Session State ---
if 'df1_raw' not in st.session_state: st.session_state.df1_raw = None
if 'df2_raw' not in st.session_state: st.session_state.df2_raw = None
if 'df1_processed' not in st.session_state: st.session_state.df1_processed = None
if 'df2_processed' not in st.session_state: st.session_state.df2_processed = None
if 'file1_name' not in st.session_state: st.session_state.file1_name = "File 1"
if 'file2_name' not in st.session_state: st.session_state.file2_name = "File 2"

# Plot settings
if 'plot_type' not in st.session_state: st.session_state.plot_type = "Line"
if 'x_axis_type' not in st.session_state: st.session_state.x_axis_type = "Elapsed Time" # "Elapsed Time" or "Time of Day"
if 'y_axis_min' not in st.session_state: st.session_state.y_axis_min = 0
if 'y_axis_max' not in st.session_state: st.session_state.y_axis_max = 500 # Default, will be updated
if 'y_grid_spacing' not in st.session_state: st.session_state.y_grid_spacing = 100
if 'show_y_grid' not in st.session_state: st.session_state.show_y_grid = True
if 'default_y_max_calculated' not in st.session_state: st.session_state.default_y_max_calculated = False


# Data transformation settings
if 'moving_avg_window' not in st.session_state: st.session_state.moving_avg_window = 1 # 1 means no averaging

# File 1 specific
if 'f1_color' not in st.session_state: st.session_state.f1_color = '#FF0000' # Red
if 'f1_shift' not in st.session_state: st.session_state.f1_shift = 0.0
if 'f1_stretch' not in st.session_state: st.session_state.f1_stretch = 1.0

# File 2 specific
if 'f2_color' not in st.session_state: st.session_state.f2_color = '#0000FF' # Blue
if 'f2_shift' not in st.session_state: st.session_state.f2_shift = 0.0
if 'f2_stretch' not in st.session_state: st.session_state.f2_stretch = 1.0


# --- UI Layout ---
st.title("🚴💨 FIT File Power Analyzer")
st.markdown("Compare power data from two .fit files recorded simultaneously.")

# --- File Uploaders ---
col_uploader1, col_uploader2 = st.columns(2)
with col_uploader1:
    uploaded_file1 = st.file_uploader("Upload first .fit file", type="fit", key="file1")
with col_uploader2:
    uploaded_file2 = st.file_uploader("Upload second .fit file", type="fit", key="file2")

# --- Data Loading and Initial Processing ---
if uploaded_file1 and st.session_state.df1_raw is None: # Process only if new file or not yet processed
    st.session_state.df1_raw = parse_fit_file(uploaded_file1)
    st.session_state.file1_name = uploaded_file1.name
    st.session_state.default_y_max_calculated = False # Recalculate Y max

if uploaded_file2 and st.session_state.df2_raw is None: # Process only if new file or not yet processed
    st.session_state.df2_raw = parse_fit_file(uploaded_file2)
    st.session_state.file2_name = uploaded_file2.name
    st.session_state.default_y_max_calculated = False # Recalculate Y max

# Calculate default Y-axis max if not done yet and data is available
if not st.session_state.default_y_max_calculated and (st.session_state.df1_raw is not None or st.session_state.df2_raw is not None):
    max_p1 = 0
    if st.session_state.df1_raw is not None and not st.session_state.df1_raw.empty:
        max_p1 = st.session_state.df1_raw['power'].max()
    
    max_p2 = 0
    if st.session_state.df2_raw is not None and not st.session_state.df2_raw.empty:
        max_p2 = st.session_state.df2_raw['power'].max()
    
    global_max_power = max(max_p1, max_p2, 100) # ensure at least 100W for scale
    st.session_state.y_axis_max = float(np.ceil(global_max_power / 100.0) * 100.0)
    st.session_state.y_axis_min = 0.0 # Reset min too
    st.session_state.default_y_max_calculated = True


# --- Apply Transformations ---
st.session_state.df1_processed = process_data(
    st.session_state.df1_raw, 
    st.session_state.f1_stretch, 
    st.session_state.f1_shift, 
    st.session_state.moving_avg_window
)
st.session_state.df2_processed = process_data(
    st.session_state.df2_raw, 
    st.session_state.f2_stretch, 
    st.session_state.f2_shift, 
    st.session_state.moving_avg_window
)

# --- Plotting Area ---
plot_placeholder = st.empty() # Use empty to re-draw plot correctly

fig = go.Figure()
plot_mode_map = {"Line": "lines", "Scatter": "markers", "Line and Scatter": "lines+markers"}
current_plot_mode = plot_mode_map.get(st.session_state.plot_type, "lines")

# Determine x_axis data source based on selection
x_axis_is_elapsed_time = st.session_state.x_axis_type == "Elapsed Time"

if st.session_state.df1_processed is not None and not st.session_state.df1_processed.empty:
    x_data1 = st.session_state.df1_processed['elapsed_seconds'] if x_axis_is_elapsed_time else st.session_state.df1_processed.index
    fig.add_trace(go.Scatter(
        x=x_data1, 
        y=st.session_state.df1_processed['power'], 
        mode=current_plot_mode, 
        name=st.session_state.file1_name,
        line=dict(color=st.session_state.f1_color)
    ))

if st.session_state.df2_processed is not None and not st.session_state.df2_processed.empty:
    x_data2 = st.session_state.df2_processed['elapsed_seconds'] if x_axis_is_elapsed_time else st.session_state.df2_processed.index
    fig.add_trace(go.Scatter(
        x=x_data2, 
        y=st.session_state.df2_processed['power'], 
        mode=current_plot_mode, 
        name=st.session_state.file2_name,
        line=dict(color=st.session_state.f2_color)
    ))

x_title = "Elapsed Time (seconds)" if x_axis_is_elapsed_time else "Time of Day"
fig.update_layout(
    height=600,
    xaxis_title=x_title,
    yaxis_title="Power (Watts)",
    yaxis_range=[st.session_state.y_axis_min, st.session_state.y_axis_max],
    yaxis_dtick=st.session_state.y_grid_spacing if st.session_state.show_y_grid else None,
    yaxis_showgrid=st.session_state.show_y_grid,
    legend_title_text='Files',
    title="Power Comparison"
)
# Add drag to zoom for x-axis (Plotly default)
# fig.update_xaxes(dragmode='pan') # or 'zoom' if you want to force it, but default is good

plot_placeholder.plotly_chart(fig, use_container_width=True)


# --- Controls Area (Bottom Third) ---
st.markdown("---")
st.header("⚙️ Analysis Controls")

# Using columns for layout
control_col1, control_col2, control_col3 = st.columns(3)

with control_col1:
    st.subheader("Global Plot Settings")
    st.session_state.plot_type = st.selectbox(
        "Plot Type:", 
        options=["Line", "Scatter", "Line and Scatter"], 
        key='sb_plot_type',
        index=["Line", "Scatter", "Line and Scatter"].index(st.session_state.plot_type)
    )
    st.session_state.x_axis_type = st.radio(
        "X-Axis Display:", 
        options=["Elapsed Time", "Time of Day"], 
        key='rb_x_axis',
        index=["Elapsed Time", "Time of Day"].index(st.session_state.x_axis_type)
    )
    
    st.session_state.moving_avg_window = st.number_input(
        "Moving Average Window (seconds, 1 for none):", 
        min_value=1, max_value=600, value=st.session_state.moving_avg_window, step=1, key='ni_mov_avg'
    )
    
    st.markdown("###### Y-Axis Scale & Grid")
    y_min_val = st.number_input("Y-Axis Min (W):", value=float(st.session_state.y_axis_min), key='ni_y_min')
    y_max_val = st.number_input("Y-Axis Max (W):", value=float(st.session_state.y_axis_max), min_value=y_min_val + 1, key='ni_y_max')
    
    if y_min_val != st.session_state.y_axis_min : st.session_state.y_axis_min = y_min_val
    if y_max_val != st.session_state.y_axis_max : st.session_state.y_axis_max = y_max_val


    st.session_state.y_grid_spacing = st.number_input(
        "Y-Axis Grid Spacing (W):", 
        min_value=5, value=st.session_state.y_grid_spacing, step=5, key='ni_y_grid_space'
    )
    st.session_state.show_y_grid = st.toggle("Show Y-Axis Gridlines", value=st.session_state.show_y_grid, key='tgl_y_grid')


with control_col2:
    st.subheader(f"File 1 Adjustments: {st.session_state.file1_name}")
    if st.session_state.df1_raw is not None:
        st.session_state.f1_color = st.color_picker("Line Color File 1:", st.session_state.f1_color, key='cp_f1_color')
        st.session_state.f1_shift = st.number_input("Shift File 1 (seconds, +/-):", value=st.session_state.f1_shift, step=0.1, format="%.1f", key='ni_f1_shift')
        st.session_state.f1_stretch = st.number_input("Stretch File 1 (factor):", value=st.session_state.f1_stretch, min_value=0.1, step=0.01, format="%.2f", key='ni_f1_stretch')
        if st.button("Reset File 1 Adjustments", key='btn_reset_f1'):
            st.session_state.f1_shift = 0.0
            st.session_state.f1_stretch = 1.0
            st.rerun() # Rerun to apply reset immediately
    else:
        st.info("Upload File 1 to see adjustment options.")

with control_col3:
    st.subheader(f"File 2 Adjustments: {st.session_state.file2_name}")
    if st.session_state.df2_raw is not None:
        st.session_state.f2_color = st.color_picker("Line Color File 2:", st.session_state.f2_color, key='cp_f2_color')
        st.session_state.f2_shift = st.number_input("Shift File 2 (seconds, +/-):", value=st.session_state.f2_shift, step=0.1, format="%.1f", key='ni_f2_shift')
        st.session_state.f2_stretch = st.number_input("Stretch File 2 (factor):", value=st.session_state.f2_stretch, min_value=0.1, step=0.01, format="%.2f", key='ni_f2_stretch')
        if st.button("Reset File 2 Adjustments", key='btn_reset_f2'):
            st.session_state.f2_shift = 0.0
            st.session_state.f2_stretch = 1.0
            st.rerun() # Rerun to apply reset immediately
    else:
        st.info("Upload File 2 to see adjustment options.")

# --- Footer/Info ---
st.markdown("---")
st.markdown("Built with Streamlit. Drag on the graph's X-axis to zoom into a time period. Double-click to reset zoom.")