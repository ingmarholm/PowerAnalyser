import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from fitparse import FitFile, FitParseError
import numpy as np
from datetime import datetime, timezone, timedelta
from streamlit_plotly_events import plotly_events # Added for box select

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="FIT File Power Analyzer")

# --- Helper Functions ---

def parse_fit_file(uploaded_file_bytes):
    """
    Parses a .fit file and extracts only timestamp and power data
    to minimize memory footprint.
    """
    if uploaded_file_bytes is None:
        return None
    
    records = []
    try:
        fitfile = FitFile(uploaded_file_bytes)
        for record in fitfile.get_messages('record'):
            timestamp = None
            power = None
            for field_data in record: # Iterate through FieldData objects
                if field_data.name == 'timestamp':
                    timestamp = field_data.value
                elif field_data.name == 'power':
                    power = field_data.value
            
            if timestamp is not None and power is not None:
                if isinstance(timestamp, datetime): # Ensure it's a datetime object
                    if timestamp.tzinfo is None: # Make timezone aware (assume UTC)
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    records.append({'timestamp': timestamp, 'power': power})
                else: # Skip if timestamp is not a valid datetime (e.g. integer)
                    st.warning(f"Skipping record with invalid timestamp type: {timestamp} in {getattr(uploaded_file_bytes, 'name', 'file')}")


        if not records:
            st.warning(f"No valid power data records found in {getattr(uploaded_file_bytes, 'name', 'file')}.")
            return None

        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Resample to 1-second intervals and interpolate
        df = df.resample('1S').mean() 
        df['power'] = df['power'].interpolate(method='linear')
        df = df.dropna(subset=['power']) # Drop rows where power couldn't be interpolated
        
        if df.empty:
            st.warning(f"Resampled data is empty for {getattr(uploaded_file_bytes, 'name', 'file')}. Check file content.")
            return None
            
        return df
    except FitParseError as e:
        st.error(f"Error parsing FIT file {getattr(uploaded_file_bytes, 'name', 'unknown')}: {e}. The file might be corrupt or not a valid FIT file.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while parsing {getattr(uploaded_file_bytes, 'name', 'unknown')}: {e}")
        return None


def process_data(df_raw, stretch_factor, shift_seconds, moving_avg_window_seconds):
    if df_raw is None or df_raw.empty:
        return None # Return None if no raw data

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
                    df_processed = pd.DataFrame({'power': interpolated_power}, index=pd.DatetimeIndex(new_indices, name='timestamp'))
                else:
                    df_processed = pd.DataFrame(columns=['power']).set_index(pd.DatetimeIndex([], name='timestamp'))
            else:
                df_processed = pd.DataFrame(columns=['power']).set_index(pd.DatetimeIndex([], name='timestamp'))
        elif len(elapsed_seconds_stretched) == 1:
            new_indices = [start_time_abs + pd.Timedelta(seconds=elapsed_seconds_stretched[0])]
            df_processed = pd.DataFrame({'power': power_values_orig}, index=pd.DatetimeIndex(new_indices, name='timestamp'))
        else:
             df_processed = pd.DataFrame(columns=['power']).set_index(pd.DatetimeIndex([], name='timestamp'))
    
    if df_processed is None: df_processed = pd.DataFrame(columns=['power']).set_index(pd.DatetimeIndex([], name='timestamp'))


    if shift_seconds != 0 and not df_processed.empty:
        df_processed.index = df_processed.index + pd.Timedelta(seconds=shift_seconds)

    if moving_avg_window_seconds > 1 and not df_processed.empty:
        df_processed['power'] = df_processed['power'].rolling(window=moving_avg_window_seconds, center=True, min_periods=1).mean()

    if not df_processed.empty:
        df_processed['elapsed_seconds'] = (df_processed.index - df_processed.index[0]).total_seconds()
    else: # Ensure columns exist even if empty for consistent access later
        if 'power' not in df_processed.columns: df_processed['power'] = pd.Series(dtype='float64')
        if 'elapsed_seconds' not in df_processed.columns: df_processed['elapsed_seconds'] = pd.Series(dtype='float64')
        if df_processed.index.name != 'timestamp': df_processed.index = pd.DatetimeIndex(df_processed.index, name='timestamp')


    return df_processed

def format_seconds_to_hhmmss(seconds_val):
    if pd.isna(seconds_val) or not isinstance(seconds_val, (int, float)) or seconds_val < 0:
        return "00:00:00"
    hours = int(seconds_val // 3600)
    minutes = int((seconds_val % 3600) // 60)
    secs = int(seconds_val % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"

def calculate_stats(df1, df2, x_min=None, x_max=None, x_axis_type="Elapsed Time"):
    stats = {
        "avg_power1": "N/A", "duration_str1": "N/A", "total_seconds1": 0,
        "avg_power2": "N/A", "duration_str2": "N/A", "total_seconds2": 0,
        "correlation": "N/A", "window_duration_str": "N/A"
    }
    
    df1_filtered, df2_filtered = df1, df2

    if x_min is not None and x_max is not None:
        stats["window_duration_str"] = format_seconds_to_hhmmss(x_max - x_min)
        if x_axis_type == "Elapsed Time":
            if df1 is not None and not df1.empty:
                df1_filtered = df1[(df1['elapsed_seconds'] >= x_min) & (df1['elapsed_seconds'] <= x_max)]
            if df2 is not None and not df2.empty:
                df2_filtered = df2[(df2['elapsed_seconds'] >= x_min) & (df2['elapsed_seconds'] <= x_max)]
        else: # Time of Day
            # Convert x_min, x_max (likely timestamps from plotly_events) to datetime if they are not
            # This depends on what plotly_events returns for datetime axes; assuming they are compatible
            if isinstance(x_min, (int, float)): x_min = pd.to_datetime(x_min, unit='ms', utc=True)
            if isinstance(x_max, (int, float)): x_max = pd.to_datetime(x_max, unit='ms', utc=True)

            if df1 is not None and not df1.empty:
                df1_filtered = df1[(df1.index >= x_min) & (df1.index <= x_max)]
            if df2 is not None and not df2.empty:
                df2_filtered = df2[(df2.index >= x_min) & (df2.index <= x_max)]
    
    # Calculate for File 1
    if df1_filtered is not None and not df1_filtered.empty:
        stats["avg_power1"] = f"{df1_filtered['power'].mean():.1f} W"
        if not df1_filtered['elapsed_seconds'].empty:
             stats["total_seconds1"] = df1_filtered['elapsed_seconds'].iloc[-1] - df1_filtered['elapsed_seconds'].iloc[0]
        stats["duration_str1"] = format_seconds_to_hhmmss(stats["total_seconds1"])

    # Calculate for File 2
    if df2_filtered is not None and not df2_filtered.empty:
        stats["avg_power2"] = f"{df2_filtered['power'].mean():.1f} W"
        if not df2_filtered['elapsed_seconds'].empty:
            stats["total_seconds2"] = df2_filtered['elapsed_seconds'].iloc[-1] - df2_filtered['elapsed_seconds'].iloc[0]
        stats["duration_str2"] = format_seconds_to_hhmmss(stats["total_seconds2"])

    # Correlation
    if df1_filtered is not None and not df1_filtered.empty and \
       df2_filtered is not None and not df2_filtered.empty:
        # Align data for correlation - crucial step
        aligned_df = pd.merge(df1_filtered['power'], df2_filtered['power'], 
                              left_index=True, right_index=True, 
                              suffixes=('_1', '_2'), how='inner')
        if not aligned_df.empty and len(aligned_df) > 1: # Need more than 1 point for correlation
            correlation = aligned_df['power_1'].corr(aligned_df['power_2'])
            stats["correlation"] = f"{correlation:.3f}"
        elif not aligned_df.empty:
             stats["correlation"] = "N/A (single point)"


    if x_min is None: # If overall, window duration is total duration of the longer file
        overall_duration = max(stats["total_seconds1"], stats["total_seconds2"])
        stats["window_duration_str"] = format_seconds_to_hhmmss(overall_duration)
        
    return stats


# --- Initialize Session State ---
default_states = {
    'df1_raw': None, 'df2_raw': None,
    'df1_processed': None, 'df2_processed': None,
    'file1_name': "File 1", 'file2_name': "File 2",
    'plot_type': "Line", 'x_axis_type': "Elapsed Time",
    'y_axis_min': 0.0, 'y_axis_max': 500.0, 'y_axis_max_default': 500.0,
    'y_grid_spacing': 100.0, 'show_y_grid': True,
    'default_y_max_calculated': False,
    'moving_avg_window': 1,
    'f1_color': '#FF0000', 'f1_shift': 0.0, 'f1_stretch': 1.0,
    'f2_color': '#0000FF', 'f2_shift': 0.0, 'f2_stretch': 1.0,
    'show_difference_plot': False,
    'current_stats': calculate_stats(None, None), # Initial empty stats
    'selected_x_min': None, 'selected_x_max': None,
}
for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- UI Layout ---
st.title("🚴💨 FIT File Power Analyzer")
st.markdown("Compare power data from two .fit files. Use 'Box Select' (in plot modebar) to select a time window for specific stats. Use 'Autoscale' (home icon) to reset zoom.")

col_uploader1, col_uploader2 = st.columns(2)
with col_uploader1:
    uploaded_file1 = st.file_uploader("Upload first .fit file", type="fit", key="file1_uploader")
with col_uploader2:
    uploaded_file2 = st.file_uploader("Upload second .fit file", type="fit", key="file2_uploader")

# --- Data Loading and Initial Processing ---
recalculate_y_axis = False
if uploaded_file1 and (st.session_state.df1_raw is None or st.session_state.file1_name != uploaded_file1.name):
    st.session_state.df1_raw = parse_fit_file(uploaded_file1)
    st.session_state.file1_name = uploaded_file1.name
    recalculate_y_axis = True
    st.session_state.selected_x_min, st.session_state.selected_x_max = None, None # Reset selection

if uploaded_file2 and (st.session_state.df2_raw is None or st.session_state.file2_name != uploaded_file2.name):
    st.session_state.df2_raw = parse_fit_file(uploaded_file2)
    st.session_state.file2_name = uploaded_file2.name
    recalculate_y_axis = True
    st.session_state.selected_x_min, st.session_state.selected_x_max = None, None # Reset selection

if recalculate_y_axis or not st.session_state.default_y_max_calculated:
    max_p1 = 0
    if st.session_state.df1_raw is not None and not st.session_state.df1_raw.empty:
        max_p1 = st.session_state.df1_raw['power'].max()
    max_p2 = 0
    if st.session_state.df2_raw is not None and not st.session_state.df2_raw.empty:
        max_p2 = st.session_state.df2_raw['power'].max()
    
    global_max_power = max(max_p1, max_p2, 100) # ensure at least 100W for scale
    st.session_state.y_axis_max_default = float(np.ceil(global_max_power / 100.0) * 100.0)
    st.session_state.y_axis_max = st.session_state.y_axis_max_default
    st.session_state.y_axis_min = 0.0
    st.session_state.default_y_max_calculated = True


# --- Apply Transformations ---
st.session_state.df1_processed = process_data(st.session_state.df1_raw, st.session_state.f1_stretch, st.session_state.f1_shift, st.session_state.moving_avg_window)
st.session_state.df2_processed = process_data(st.session_state.df2_raw, st.session_state.f2_stretch, st.session_state.f2_shift, st.session_state.moving_avg_window)


# --- Calculate Statistics based on selection or overall ---
st.session_state.current_stats = calculate_stats(
    st.session_state.df1_processed, 
    st.session_state.df2_processed,
    st.session_state.selected_x_min,
    st.session_state.selected_x_max,
    st.session_state.x_axis_type
)


# --- Plotting Area ---
fig = go.Figure()
plot_mode_map = {"Line": "lines", "Scatter": "markers", "Line and Scatter": "lines+markers"}
current_plot_mode = plot_mode_map.get(st.session_state.plot_type, "lines")
scatter_marker_size = 4 if "Scatter" in st.session_state.plot_type else None


x_axis_is_elapsed_time = st.session_state.x_axis_type == "Elapsed Time"

# Add traces
if st.session_state.df1_processed is not None and not st.session_state.df1_processed.empty:
    x_data1 = st.session_state.df1_processed['elapsed_seconds'] if x_axis_is_elapsed_time else st.session_state.df1_processed.index
    fig.add_trace(go.Scatter(
        x=x_data1, y=st.session_state.df1_processed['power'], mode=current_plot_mode, 
        name=st.session_state.file1_name, line=dict(color=st.session_state.f1_color, width=1),
        marker=dict(size=scatter_marker_size),
        hovertemplate="<b>File 1 Power:</b> %{y:.1f}W<extra></extra>"
    ))

if st.session_state.df2_processed is not None and not st.session_state.df2_processed.empty:
    x_data2 = st.session_state.df2_processed['elapsed_seconds'] if x_axis_is_elapsed_time else st.session_state.df2_processed.index
    fig.add_trace(go.Scatter(
        x=x_data2, y=st.session_state.df2_processed['power'], mode=current_plot_mode, 
        name=st.session_state.file2_name, line=dict(color=st.session_state.f2_color, width=1),
        marker=dict(size=scatter_marker_size),
        hovertemplate="<b>File 2 Power:</b> %{y:.1f}W<extra></extra>"
    ))

# Difference Plot
if st.session_state.show_difference_plot and \
   st.session_state.df1_processed is not None and not st.session_state.df1_processed.empty and \
   st.session_state.df2_processed is not None and not st.session_state.df2_processed.empty:
    
    # Align dataframes for difference calculation
    df_diff_aligned = pd.merge(
        st.session_state.df1_processed[['power', 'elapsed_seconds' if x_axis_is_elapsed_time else None].dropna()], 
        st.session_state.df2_processed['power'],
        left_index=True, 
        right_index=True,
        how='inner', # Only include timestamps present in both
        suffixes=('_1', '_2')
    )
    if not df_diff_aligned.empty:
        df_diff_aligned['power_difference'] = df_diff_aligned['power_1'] - df_diff_aligned['power_2']
        x_data_diff = df_diff_aligned['elapsed_seconds_1'] if x_axis_is_elapsed_time else df_diff_aligned.index
        fig.add_trace(go.Scatter(
            x=x_data_diff, y=df_diff_aligned['power_difference'], mode='lines',
            name="Difference (F1-F2)", line=dict(color='green', width=1, dash='dash'),
            hovertemplate="<b>Difference:</b> %{y:.1f}W<extra></extra>"
        ))


# Configure x-axis
x_title = "Elapsed Time (HH:MM:SS)" if x_axis_is_elapsed_time else "Time of Day (UTC)"
if x_axis_is_elapsed_time:
    min_sec_all, max_sec_all = float('inf'), float('-inf')
    if st.session_state.df1_processed is not None and not st.session_state.df1_processed.empty:
        min_sec_all = min(min_sec_all, st.session_state.df1_processed['elapsed_seconds'].min())
        max_sec_all = max(max_sec_all, st.session_state.df1_processed['elapsed_seconds'].max())
    if st.session_state.df2_processed is not None and not st.session_state.df2_processed.empty:
        min_sec_all = min(min_sec_all, st.session_state.df2_processed['elapsed_seconds'].min())
        max_sec_all = max(max_sec_all, st.session_state.df2_processed['elapsed_seconds'].max())

    if pd.notna(min_sec_all) and pd.notna(max_sec_all) and max_sec_all > min_sec_all :
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
    margin=dict(t=50, b=50, l=50, r=50), # Added some margin
    xaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True), # Chart outline
    yaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True), # Chart outline
    hovermode='x unified' # Unified hover with vertical line
)

# Use plotly_events to capture selections
selected_points = plotly_events(fig, select_event=True, key="plot_selector_events")

if selected_points and 'xrange' in selected_points[0]:
    new_xmin, new_xmax = selected_points[0]['xrange']
    # Check if selection actually changed to avoid redundant reruns
    if new_xmin != st.session_state.selected_x_min or new_xmax != st.session_state.selected_x_max:
        st.session_state.selected_x_min = new_xmin
        st.session_state.selected_x_max = new_xmax
        st.rerun() # Rerun to update stats based on new selection

# --- Display Plot ---
st.plotly_chart(fig, use_container_width=True)


# --- Summary Statistics Display ---
st.markdown("---")
stats_header_text = "📊 Summary Statistics"
if st.session_state.selected_x_min is not None:
    stats_header_text += f" (Selected Window: {st.session_state.current_stats['window_duration_str']})"
else:
    stats_header_text += f" (Overall Duration: {st.session_state.current_stats['window_duration_str']})"
st.subheader(stats_header_text)


summary_cols = st.columns([2,2,1]) # Three columns for stats
with summary_cols[0]:
    st.markdown(f"**{st.session_state.file1_name}:**")
    st.markdown(f"<span style='color:{st.session_state.f1_color};'>Avg Power: {st.session_state.current_stats['avg_power1']}</span>", unsafe_allow_html=True)
    if st.session_state.selected_x_min is None: # Show total duration only for overall stats
        st.markdown(f"<span style='color:{st.session_state.f1_color};'>Total Duration: {st.session_state.current_stats['duration_str1']}</span>", unsafe_allow_html=True)

with summary_cols[1]:
    st.markdown(f"**{st.session_state.file2_name}:**")
    st.markdown(f"<span style='color:{st.session_state.f2_color};'>Avg Power: {st.session_state.current_stats['avg_power2']}</span>", unsafe_allow_html=True)
    if st.session_state.selected_x_min is None: # Show total duration only for overall stats
        st.markdown(f"<span style='color:{st.session_state.f2_color};'>Total Duration: {st.session_state.current_stats['duration_str2']}</span>", unsafe_allow_html=True)

with summary_cols[2]:
    st.markdown("**Comparison:**")
    st.markdown(f"Correlation: {st.session_state.current_stats['correlation']}")
    if st.session_state.selected_x_min is not None:
        if st.button("Clear Selection & Show Overall Stats", key="clear_selection_btn"):
            st.session_state.selected_x_min = None
            st.session_state.selected_x_max = None
            st.rerun()


# --- Controls Area ---
st.markdown("---")
st.header("⚙️ Analysis Controls")

control_col1, control_col2, control_col3 = st.columns(3)

with control_col1:
    st.subheader("Global Plot Settings")
    
    # Selectbox for Plot Type
    prev_plot_type = st.session_state.plot_type
    st.session_state.plot_type = st.selectbox("Plot Type:", options=["Line", "Scatter", "Line and Scatter"], 
                                              index=["Line", "Scatter", "Line and Scatter"].index(st.session_state.plot_type),
                                              key='sb_plot_type_k')
    if prev_plot_type != st.session_state.plot_type: st.rerun()

    # Radio for X-Axis Type
    prev_x_axis_type = st.session_state.x_axis_type
    st.session_state.x_axis_type = st.radio("X-Axis Display:", options=["Elapsed Time", "Time of Day"],
                                             index=["Elapsed Time", "Time of Day"].index(st.session_state.x_axis_type),
                                             key='rb_x_axis_k')
    if prev_x_axis_type != st.session_state.x_axis_type: 
        st.session_state.selected_x_min, st.session_state.selected_x_max = None, None # Reset selection on axis type change
        st.rerun()
    
    # Number input for Moving Average
    prev_mov_avg = st.session_state.moving_avg_window
    st.session_state.moving_avg_window = st.number_input("Moving Average Window (seconds, 1 for none):", 
                                                         min_value=1, max_value=600, value=st.session_state.moving_avg_window, 
                                                         step=1, key='ni_mov_avg_k')
    if prev_mov_avg != st.session_state.moving_avg_window: st.rerun()

    # Toggle Difference Plot
    prev_show_diff = st.session_state.show_difference_plot
    st.session_state.show_difference_plot = st.toggle("Show Power Difference Plot (F1-F2)", value=st.session_state.show_difference_plot, key='tgl_diff_plot_k')
    if prev_show_diff != st.session_state.show_difference_plot: st.rerun()
    
    st.markdown("###### Y-Axis Scale & Grid")
    # Y-Axis Min
    prev_y_min = st.session_state.y_axis_min
    st.session_state.y_axis_min = st.number_input("Y-Axis Min (W):", value=float(st.session_state.y_axis_min), key='ni_y_min_k', format="%.1f", step=10.0)
    if prev_y_min != st.session_state.y_axis_min: st.rerun()

    # Y-Axis Max
    prev_y_max = st.session_state.y_axis_max
    st.session_state.y_axis_max = st.number_input("Y-Axis Max (W):", value=float(st.session_state.y_axis_max), 
                                                  min_value=float(st.session_state.y_axis_min) + 1.0, key='ni_y_max_k', format="%.1f", step=10.0)
    if prev_y_max != st.session_state.y_axis_max: st.rerun()

    if st.button("Reset Y-Axis Scale", key="reset_y_scale_btn"):
        st.session_state.y_axis_min = 0.0
        st.session_state.y_axis_max = st.session_state.y_axis_max_default # Use stored default
        st.rerun()

    # Y-Grid Spacing
    prev_y_grid = st.session_state.y_grid_spacing
    st.session_state.y_grid_spacing = st.number_input("Y-Axis Grid Spacing (W):", min_value=5.0, 
                                                      value=float(st.session_state.y_grid_spacing), step=5.0, key='ni_y_grid_space_k', format="%.1f")
    if prev_y_grid != st.session_state.y_grid_spacing: st.rerun()

    # Toggle Y-Grid
    prev_show_grid = st.session_state.show_y_grid
    st.session_state.show_y_grid = st.toggle("Show Y-Axis Gridlines", value=st.session_state.show_y_grid, key='tgl_y_grid_k')
    if prev_show_grid != st.session_state.show_y_grid: st.rerun()


with control_col2:
    st.subheader(f"File 1 Adjustments: {st.session_state.file1_name}")
    if st.session_state.df1_raw is not None:
        prev_f1_color = st.session_state.f1_color
        st.session_state.f1_color = st.color_picker("Line Color File 1:", st.session_state.f1_color, key='cp_f1_color_k')
        if prev_f1_color != st.session_state.f1_color: st.rerun()

        prev_f1_shift = st.session_state.f1_shift
        st.session_state.f1_shift = st.number_input("Shift File 1 (seconds, +/-):", value=st.session_state.f1_shift, 
                                                    step=0.1, format="%.1f", key='ni_f1_shift_k')
        if prev_f1_shift != st.session_state.f1_shift: st.rerun()

        prev_f1_stretch = st.session_state.f1_stretch
        st.session_state.f1_stretch = st.number_input("Stretch File 1 (factor):", value=st.session_state.f1_stretch, 
                                                      min_value=0.01, step=0.01, format="%.2f", key='ni_f1_stretch_k') # min_value changed
        if prev_f1_stretch != st.session_state.f1_stretch: st.rerun()
        
        if st.button("Reset File 1 Adjustments", key='btn_reset_f1_k'):
            st.session_state.f1_shift = 0.0
            st.session_state.f1_stretch = 1.0
            st.rerun()
    else:
        st.info("Upload File 1 to see adjustment options.")

with control_col3:
    st.subheader(f"File 2 Adjustments: {st.session_state.file2_name}")
    if st.session_state.df2_raw is not None:
        prev_f2_color = st.session_state.f2_color
        st.session_state.f2_color = st.color_picker("Line Color File 2:", st.session_state.f2_color, key='cp_f2_color_k')
        if prev_f2_color != st.session_state.f2_color: st.rerun()

        prev_f2_shift = st.session_state.f2_shift
        st.session_state.f2_shift = st.number_input("Shift File 2 (seconds, +/-):", value=st.session_state.f2_shift, 
                                                    step=0.1, format="%.1f", key='ni_f2_shift_k')
        if prev_f2_shift != st.session_state.f2_shift: st.rerun()

        prev_f2_stretch = st.session_state.f2_stretch
        st.session_state.f2_stretch = st.number_input("Stretch File 2 (factor):", value=st.session_state.f2_stretch, 
                                                      min_value=0.01, step=0.01, format="%.2f", key='ni_f2_stretch_k') # min_value changed
        if prev_f2_stretch != st.session_state.f2_stretch: st.rerun()

        if st.button("Reset File 2 Adjustments", key='btn_reset_f2_k'):
            st.session_state.f2_shift = 0.0
            st.session_state.f2_stretch = 1.0
            st.rerun()
    else:
        st.info("Upload File 2 to see adjustment options.")

st.markdown("---")
st.caption("Tip: Use 'Box Select' or 'Lasso Select' from the plot modebar to select a time window for detailed stats. Hover over the graph to see other controls like zoom, pan, and autoscale (reset zoom). Timestamps are processed and displayed in UTC.")
