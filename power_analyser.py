import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from fitparse import FitFile, FitParseError
import numpy as np
from datetime import timezone, timedelta
from streamlit_plotly_events import plotly_events # Needs: pip install streamlit-plotly-events

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="FIT File Power Analyzer")

# --- Helper Functions ---

def parse_fit_file(uploaded_file_bytes, timezone_offset_hours):
    """Parses a .fit file, extracts timestamp and power, applies timezone offset."""
    if uploaded_file_bytes is None:
        return None
    try:
        # Ensure uploaded_file_bytes is bytes-like
        if hasattr(uploaded_file_bytes, 'getvalue'): # Streamlit UploadedFile
            file_bytes_content = uploaded_file_bytes.getvalue()
        elif isinstance(uploaded_file_bytes, bytes):
            file_bytes_content = uploaded_file_bytes
        else:
            st.error("Invalid file object passed to parse_fit_file.")
            return None

        fitfile = FitFile(file_bytes_content)
        records = []
        for record in fitfile.get_messages('record'):
            timestamp = None
            power = None
            for field in record:
                if field.name == 'timestamp':
                    timestamp = field.value
                elif field.name == 'power':
                    power = field.value
            
            if timestamp is not None and power is not None:
                # Ensure timestamp is a pandas Timestamp for consistent handling
                if not isinstance(timestamp, pd.Timestamp):
                     timestamp = pd.to_datetime(timestamp)

                # Standardize to UTC first
                if timestamp.tzinfo is None: # If naive, assume UTC as per fitparse behavior
                    timestamp = timestamp.tz_localize('UTC')
                else: # If timezone aware, convert to UTC to standardize
                    timestamp = timestamp.astimezone(timezone.utc)
                
                # Apply user-defined timezone offset
                if timezone_offset_hours != 0:
                    timestamp = timestamp + timedelta(hours=timezone_offset_hours)
                
                records.append({'timestamp': timestamp, 'power': power})

        if not records:
            st.warning(f"No power data records found in {getattr(uploaded_file_bytes, 'name', 'file')}.")
            return None

        df = pd.DataFrame(records)
        df = df.set_index('timestamp') # timestamp is now the index
        
        # Performance: Reduce data size early by keeping only the 'power' column
        df = df[['power']] 
        
        # Resample to 1-second intervals and interpolate
        df = df.resample('1S').mean() 
        df['power'] = df['power'].interpolate(method='linear')
        df = df.dropna(subset=['power']) # Drop rows where power couldn't be interpolated
        
        if df.empty:
            st.warning(f"Resampled data is empty for {getattr(uploaded_file_bytes, 'name', 'file')}. Check file content.")
            return None
        return df
    except FitParseError as e:
        st.error(f"Error parsing FIT file (FitParseError) {getattr(uploaded_file_bytes, 'name', 'unknown')}: {e}. The file might be corrupt or not a valid FIT file.")
        return None
    except Exception as e:
        st.error(f"General error parsing FIT file {getattr(uploaded_file_bytes, 'name', 'unknown')}: {e}")
        return None

def process_data(df_raw, stretch_factor, shift_seconds, moving_avg_window_seconds):
    if df_raw is None or df_raw.empty:
        return None

    df_processed = df_raw.copy()

    # 1. Stretch
    if stretch_factor != 1.0 and not df_processed.empty:
        start_time_abs = df_processed.index[0]
        # Calculate elapsed seconds from the true start of this series
        elapsed_seconds_orig = (df_processed.index - start_time_abs).total_seconds().to_numpy()
        power_values_orig = df_processed['power'].to_numpy()

        elapsed_seconds_stretched = elapsed_seconds_orig * stretch_factor
        
        if len(elapsed_seconds_stretched) > 1 :
            # Create a new regular time grid for interpolation on the stretched timeline
            # Target 1-second steps on the new stretched timeline
            # Ensure min_val <= max_val for arange
            min_stretched_sec = elapsed_seconds_stretched.min()
            max_stretched_sec = elapsed_seconds_stretched.max()
            if min_stretched_sec > max_stretched_sec: # Should not happen if data is ordered
                min_stretched_sec, max_stretched_sec = max_stretched_sec, min_stretched_sec

            target_stretched_elapsed_seconds = np.arange(min_stretched_sec, max_stretched_sec + 1, 1)
            
            if len(target_stretched_elapsed_seconds) > 0:
                sort_indices = np.argsort(elapsed_seconds_stretched)
                elapsed_seconds_stretched_sorted = elapsed_seconds_stretched[sort_indices]
                power_values_orig_sorted = power_values_orig[sort_indices]

                unique_stretched_seconds, unique_indices = np.unique(elapsed_seconds_stretched_sorted, return_index=True)
                unique_power_values = power_values_orig_sorted[unique_indices]

                if len(unique_stretched_seconds) > 1: # Need at least 2 points to interpolate
                    interpolated_power = np.interp(target_stretched_elapsed_seconds, unique_stretched_seconds, unique_power_values)
                    # Create new timestamps relative to the original start_time_abs
                    new_indices = [start_time_abs + pd.Timedelta(seconds=s) for s in target_stretched_elapsed_seconds]
                    df_processed = pd.DataFrame({'power': interpolated_power}, index=pd.DatetimeIndex(new_indices, name='timestamp'))
                else: 
                    df_processed = pd.DataFrame(columns=['power']).set_index(pd.DatetimeIndex([], name='timestamp'))
            else: 
                df_processed = pd.DataFrame(columns=['power']).set_index(pd.DatetimeIndex([], name='timestamp'))
        elif len(elapsed_seconds_stretched) == 1: # Single point data
            new_indices = [start_time_abs + pd.Timedelta(seconds=elapsed_seconds_stretched[0])]
            df_processed = pd.DataFrame({'power': power_values_orig}, index=pd.DatetimeIndex(new_indices, name='timestamp'))
        else: # Empty after initial load
             df_processed = pd.DataFrame(columns=['power']).set_index(pd.DatetimeIndex([], name='timestamp'))

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
        if 'power' not in df_processed.columns: df_processed['power'] = pd.Series(dtype='float64')
        if 'elapsed_seconds' not in df_processed.columns: df_processed['elapsed_seconds'] = pd.Series(dtype='float64')
        if df_processed.index.name != 'timestamp': df_processed.index = pd.DatetimeIndex(df_processed.index, name='timestamp') # Ensure index has a name
    return df_processed

def format_seconds_to_hhmmss(seconds):
    if pd.isna(seconds) or not isinstance(seconds, (int, float)) or seconds < 0:
        return "00:00:00" # Or handle as error/None
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"

# --- Initialize Session State ---
default_states = {
    'df1_raw': None, 'df2_raw': None,
    'df1_processed': None, 'df2_processed': None,
    'df1_visible': None, 'df2_visible': None, 
    'file1_name': "File 1", 'file2_name': "File 2",
    'plot_type': "Line", 'x_axis_type': "Elapsed Time",
    'y_axis_min': 0.0, 'y_axis_max': 500.0, 'y_axis_default_max': 500.0,
    'y_grid_spacing': 100.0, 'show_y_grid': True,
    'default_y_max_calculated': False,
    'moving_avg_window': 1,
    'f1_color': '#FF0000', 'f1_shift': 0.0, 'f1_stretch': 1.0,
    'f2_color': '#0000FF', 'f2_shift': 0.0, 'f2_stretch': 1.0,
    'show_difference_plot': False,
    'timezone_offset': 0.0, 
    'current_xaxis_range': None, 
    'hover_power1': None, 'hover_power2': None, 'hover_x_value_display': None, # For annotations
    'last_hover_x_raw': None, # To detect actual hover change
}
for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- UI Layout ---
st.title("🚴💨 FIT File Power Analyzer")

# --- File Upload & Timezone (Sidebar) ---
st.sidebar.header("📁 File Upload & Settings")
uploaded_file1_obj = st.sidebar.file_uploader("Upload first .fit file", type="fit", key="file1_uploader_key")
uploaded_file2_obj = st.sidebar.file_uploader("Upload second .fit file", type="fit", key="file2_uploader_key")

st.session_state.timezone_offset = st.sidebar.number_input(
    "Timezone Offset from UTC (hours)", 
    value=st.session_state.timezone_offset, 
    step=0.5, format="%.1f",
    help="Adjust if 'Time of Day' on the x-axis is incorrect. E.g., for AEST (UTC+10), enter 10."
)

# --- Data Loading and Initial Processing ---
# Re-parse if timezone offset changes or new file
if uploaded_file1_obj and (st.session_state.df1_raw is None or st.session_state.file1_name != uploaded_file1_obj.name or 'tz_offset_f1' not in st.session_state or st.session_state.tz_offset_f1 != st.session_state.timezone_offset):
    st.session_state.df1_raw = parse_fit_file(uploaded_file1_obj, st.session_state.timezone_offset)
    st.session_state.file1_name = uploaded_file1_obj.name
    st.session_state.tz_offset_f1 = st.session_state.timezone_offset
    st.session_state.default_y_max_calculated = False
    st.session_state.current_xaxis_range = None 

if uploaded_file2_obj and (st.session_state.df2_raw is None or st.session_state.file2_name != uploaded_file2_obj.name or 'tz_offset_f2' not in st.session_state or st.session_state.tz_offset_f2 != st.session_state.timezone_offset):
    st.session_state.df2_raw = parse_fit_file(uploaded_file2_obj, st.session_state.timezone_offset)
    st.session_state.file2_name = uploaded_file2_obj.name
    st.session_state.tz_offset_f2 = st.session_state.timezone_offset
    st.session_state.default_y_max_calculated = False
    st.session_state.current_xaxis_range = None

if not st.session_state.default_y_max_calculated and (st.session_state.df1_raw is not None or st.session_state.df2_raw is not None):
    max_p1 = st.session_state.df1_raw['power'].max() if st.session_state.df1_raw is not None and not st.session_state.df1_raw.empty else 0
    max_p2 = st.session_state.df2_raw['power'].max() if st.session_state.df2_raw is not None and not st.session_state.df2_raw.empty else 0
    global_max_power = max(max_p1, max_p2, 100) 
    st.session_state.y_axis_default_max = float(np.ceil(global_max_power / 100.0) * 100.0)
    st.session_state.y_axis_max = st.session_state.y_axis_default_max
    st.session_state.y_axis_min = 0.0
    st.session_state.default_y_max_calculated = True

st.session_state.df1_processed = process_data(st.session_state.df1_raw, st.session_state.f1_stretch, st.session_state.f1_shift, st.session_state.moving_avg_window)
st.session_state.df2_processed = process_data(st.session_state.df2_raw, st.session_state.f2_stretch, st.session_state.f2_shift, st.session_state.moving_avg_window)

# --- Plotting Area ---
fig = go.Figure()
plot_mode_map = {"Line": "lines", "Scatter": "markers", "Line and Scatter": "lines+markers"}
current_plot_mode = plot_mode_map.get(st.session_state.plot_type, "lines")
scatter_marker_size = 4 
x_axis_is_elapsed_time = st.session_state.x_axis_type == "Elapsed Time"

if st.session_state.df1_processed is not None and not st.session_state.df1_processed.empty:
    x_data1 = st.session_state.df1_processed['elapsed_seconds'] if x_axis_is_elapsed_time else st.session_state.df1_processed.index
    fig.add_trace(go.Scatter(
        x=x_data1, y=st.session_state.df1_processed['power'], mode=current_plot_mode, 
        name=st.session_state.file1_name, line=dict(color=st.session_state.f1_color, width=1),
        marker=dict(size=scatter_marker_size), hoverinfo='none' 
    ))

if st.session_state.df2_processed is not None and not st.session_state.df2_processed.empty:
    x_data2 = st.session_state.df2_processed['elapsed_seconds'] if x_axis_is_elapsed_time else st.session_state.df2_processed.index
    fig.add_trace(go.Scatter(
        x=x_data2, y=st.session_state.df2_processed['power'], mode=current_plot_mode, 
        name=st.session_state.file2_name, line=dict(color=st.session_state.f2_color, width=1),
        marker=dict(size=scatter_marker_size), hoverinfo='none'
    ))

if st.session_state.show_difference_plot and \
   st.session_state.df1_processed is not None and not st.session_state.df1_processed.empty and \
   st.session_state.df2_processed is not None and not st.session_state.df2_processed.empty:
    df1_aligned, df2_aligned = st.session_state.df1_processed.align(st.session_state.df2_processed, join='inner', axis=0)
    if not df1_aligned.empty and not df2_aligned.empty:
        power_diff = df1_aligned['power'] - df2_aligned['power']
        x_data_diff = df1_aligned['elapsed_seconds'] if x_axis_is_elapsed_time else df1_aligned.index
        fig.add_trace(go.Scatter(
            x=x_data_diff, y=power_diff, mode='lines', name='Power Diff (F1-F2)',
            line=dict(color='rgba(0,128,0,0.7)', width=1, dash='dash'), hoverinfo='none' # Semi-transparent green
        ))

x_title = "Elapsed Time (HH:MM:SS)" if x_axis_is_elapsed_time else "Time of Day"
xaxis_config = dict(title_text=x_title, nticks=10, showspikes=True, spikemode='across', spikesnap='cursor', spikethickness=1, spikedash='solid', spikecolor='grey')

if x_axis_is_elapsed_time:
    min_sec_all, max_sec_all = float('inf'), float('-inf')
    for df_proc in [st.session_state.df1_processed, st.session_state.df2_processed]:
        if df_proc is not None and not df_proc.empty and 'elapsed_seconds' in df_proc.columns:
            min_sec_all = min(min_sec_all, df_proc['elapsed_seconds'].min())
            max_sec_all = max(max_sec_all, df_proc['elapsed_seconds'].max())
    if max_sec_all > min_sec_all and np.isfinite(min_sec_all) and np.isfinite(max_sec_all):
        tickvals = np.linspace(min_sec_all, max_sec_all, num=10)
        ticktext = [format_seconds_to_hhmmss(s) for s in tickvals]
        xaxis_config.update(tickvals=tickvals, ticktext=ticktext)
fig.update_xaxes(**xaxis_config)

if st.session_state.current_xaxis_range:
    fig.update_xaxes(range=st.session_state.current_xaxis_range)

# --- Dynamic Annotations for Hover ---
current_annotations = []
if st.session_state.hover_x_value_display is not None: # Check if there's hover data to display
    annotation_props = dict(
        xref="paper", x=1.01, 
        yref="y", 
        showarrow=False, align="left",
        bgcolor="rgba(255,255,255,0.8)", borderpad=3, font=dict(size=11)
    )
    if st.session_state.hover_power1 is not None:
        current_annotations.append(dict(y=st.session_state.hover_power1, 
                                        text=f"{st.session_state.hover_power1:.0f}W", 
                                        font_color=st.session_state.f1_color, **annotation_props))
    if st.session_state.hover_power2 is not None:
        current_annotations.append(dict(y=st.session_state.hover_power2, 
                                        text=f"{st.session_state.hover_power2:.0f}W", 
                                        font_color=st.session_state.f2_color, **annotation_props))

fig.update_layout(
    height=550,
    yaxis_title="Power (Watts)",
    yaxis_range=[st.session_state.y_axis_min, st.session_state.y_axis_max],
    yaxis_dtick=st.session_state.y_grid_spacing if st.session_state.show_y_grid else None,
    yaxis_showgrid=st.session_state.show_y_grid,
    legend_title_text='Files', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    title="Power Comparison",
    margin=dict(t=50, b=50, r=120), # Increased right margin for annotations
    xaxis_showline=True, yaxis_showline=True, xaxis_linewidth=1, yaxis_linewidth=1, xaxis_linecolor='darkgrey', yaxis_linecolor='darkgrey',
    hovermode='x unified',
    annotations=current_annotations
)

plot_event_data = plotly_events(fig, key="main_plot_events", click_event=False, hover_event=True, select_event=False, relayout_event=True, override_height=fig.layout.height)

# --- Handle Plot Events for Zoomed Stats & Hover ---
st.session_state.df1_visible = st.session_state.df1_processed # Default to all processed data
st.session_state.df2_visible = st.session_state.df2_processed

if plot_event_data:
    relayout_data = next((event for event in plot_event_data if "xaxis.range[0]" in event or "xaxis.autorange" in event), None)
    if relayout_data:
        should_rerun_for_zoom = False
        if "xaxis.autorange" in relayout_data and relayout_data["xaxis.autorange"]:
            if st.session_state.current_xaxis_range is not None: # If it was previously zoomed
                st.session_state.current_xaxis_range = None
                should_rerun_for_zoom = True
        elif "xaxis.range[0]" in relayout_data:
            new_range = [relayout_data["xaxis.range[0]"], relayout_data["xaxis.range[1]"]]
            if st.session_state.current_xaxis_range != new_range:
                 st.session_state.current_xaxis_range = new_range
                 should_rerun_for_zoom = True
        
        if should_rerun_for_zoom:
            st.rerun() # Rerun to apply zoom filtering and update stats

    hover_data_points = next((event.get("points") for event in plot_event_data if "points" in event and event["points"]), None)
    if hover_data_points:
        raw_x_val = hover_data_points[0]['x'] # From 'x unified', x is common
        
        if st.session_state.last_hover_x_raw != raw_x_val: # Only update if x value actually changed
            st.session_state.last_hover_x_raw = raw_x_val
            st.session_state.hover_power1, st.session_state.hover_power2 = None, None

            if x_axis_is_elapsed_time:
                st.session_state.hover_x_value_display = format_seconds_to_hhmmss(raw_x_val)
                if st.session_state.df1_processed is not None and not st.session_state.df1_processed.empty:
                    st.session_state.hover_power1 = np.interp(raw_x_val, st.session_state.df1_processed['elapsed_seconds'], st.session_state.df1_processed['power'])
                if st.session_state.df2_processed is not None and not st.session_state.df2_processed.empty:
                    st.session_state.hover_power2 = np.interp(raw_x_val, st.session_state.df2_processed['elapsed_seconds'], st.session_state.df2_processed['power'])
            else: # Time of Day
                try:
                    hover_x_dt = pd.to_datetime(raw_x_val)
                    st.session_state.hover_x_value_display = hover_x_dt.strftime('%H:%M:%S') # Show time part for hover
                    
                    # Interpolation for datetime index (more robust: reindex and interpolate)
                    for i, df_proc in enumerate([st.session_state.df1_processed, st.session_state.df2_processed]):
                        if df_proc is not None and not df_proc.empty:
                            # Create a temporary Series with the hover_x_dt to align
                            temp_index = df_proc.index.union([hover_x_dt])
                            reindexed_power = df_proc['power'].reindex(temp_index).interpolate(method='time')
                            power_val = reindexed_power.get(hover_x_dt)
                            if i == 0: st.session_state.hover_power1 = power_val
                            else: st.session_state.hover_power2 = power_val
                except Exception: # Catch parsing or interpolation errors
                    st.session_state.hover_x_value_display = "N/A"

            st.rerun() # Rerun to update annotations with new hover values
    # Clear annotations if mouse leaves plot (no hover data)
    elif not hover_data_points and st.session_state.last_hover_x_raw is not None:
        st.session_state.hover_power1, st.session_state.hover_power2, st.session_state.hover_x_value_display = None, None, None
        st.session_state.last_hover_x_raw = None
        st.rerun()

# Filter data based on current_xaxis_range for visible stats
if st.session_state.current_xaxis_range:
    x_min, x_max = st.session_state.current_xaxis_range
    for i, df_proc in enumerate([st.session_state.df1_processed, st.session_state.df2_processed]):
        df_visible_setter = st.session_state.__setitem__
        df_visible_key = f'df{i+1}_visible'
        if df_proc is not None and not df_proc.empty:
            if x_axis_is_elapsed_time:
                df_visible_setter(df_visible_key, df_proc[(df_proc['elapsed_seconds'] >= x_min) & (df_proc['elapsed_seconds'] <= x_max)])
            else:
                try:
                    # Ensure x_min, x_max are timezone-aware if df_proc.index is
                    tz_info = df_proc.index.tz
                    dt_x_min = pd.to_datetime(x_min)
                    dt_x_max = pd.to_datetime(x_max)
                    if tz_info:
                        dt_x_min = dt_x_min.tz_localize('UTC').tz_convert(tz_info) if dt_x_min.tzinfo is None else dt_x_min.astimezone(tz_info)
                        dt_x_max = dt_x_max.tz_localize('UTC').tz_convert(tz_info) if dt_x_max.tzinfo is None else dt_x_max.astimezone(tz_info)
                    
                    df_visible_setter(df_visible_key, df_proc[(df_proc.index >= dt_x_min) & (df_proc.index <= dt_x_max)])
                except Exception:
                    df_visible_setter(df_visible_key, df_proc) # Fallback
        else:
            df_visible_setter(df_visible_key, None)
else: # No zoom range means full data is visible
    st.session_state.df1_visible = st.session_state.df1_processed
    st.session_state.df2_visible = st.session_state.df2_processed


# --- Summary Statistics Display (for visible range) ---
st.markdown("---")
st.subheader("📊 Summary of Visible Data in Graph")
summary_cols = st.columns(3) 
visible_duration_seconds = None

# Determine visible duration (use df1_visible as primary, fallback to df2_visible)
df_for_duration = st.session_state.df1_visible if st.session_state.df1_visible is not None and not st.session_state.df1_visible.empty else st.session_state.df2_visible

if df_for_duration is not None and not df_for_duration.empty:
    if x_axis_is_elapsed_time and 'elapsed_seconds' in df_for_duration.columns:
        min_vis_sec = df_for_duration['elapsed_seconds'].min()
        max_vis_sec = df_for_duration['elapsed_seconds'].max()
        if pd.notna(min_vis_sec) and pd.notna(max_vis_sec):
            visible_duration_seconds = max_vis_sec - min_vis_sec
    elif not x_axis_is_elapsed_time: # Time of Day
        min_vis_time = df_for_duration.index.min()
        max_vis_time = df_for_duration.index.max()
        if pd.notna(min_vis_time) and pd.notna(max_vis_time):
            visible_duration_seconds = (max_vis_time - min_vis_time).total_seconds()

summary_cols[0].metric("Visible Duration", format_seconds_to_hhmmss(visible_duration_seconds) if visible_duration_seconds is not None else "N/A")

avg_power1_vis_str = "N/A"
if st.session_state.df1_visible is not None and not st.session_state.df1_visible.empty:
    avg_power1_vis_str = f"{st.session_state.df1_visible['power'].mean():.1f} W"
summary_cols[0].markdown(f"**{st.session_state.file1_name}:** <span style='color:{st.session_state.f1_color};'>Avg Power: {avg_power1_vis_str}</span>", unsafe_allow_html=True)

avg_power2_vis_str = "N/A"
if st.session_state.df2_visible is not None and not st.session_state.df2_visible.empty:
    avg_power2_vis_str = f"{st.session_state.df2_visible['power'].mean():.1f} W"
summary_cols[1].markdown(f"**{st.session_state.file2_name}:** <span style='color:{st.session_state.f2_color};'>Avg Power: {avg_power2_vis_str}</span>", unsafe_allow_html=True)

correlation_str = "N/A"
if st.session_state.df1_visible is not None and not st.session_state.df1_visible.empty and \
   st.session_state.df2_visible is not None and not st.session_state.df2_visible.empty:
    df1_vis_aligned, df2_vis_aligned = st.session_state.df1_visible.align(st.session_state.df2_visible, join='inner', axis=0)
    if not df1_vis_aligned.empty and len(df1_vis_aligned['power']) > 1 and len(df2_vis_aligned['power']) > 1:
        correlation = df1_vis_aligned['power'].corr(df2_vis_aligned['power'])
        if pd.notna(correlation): correlation_str = f"{correlation:.3f}"
summary_cols[2].metric("Power Correlation (Visible)", correlation_str)

# --- Controls Area (Sidebar) ---
st.sidebar.header("⚙️ Analysis Controls")
st.sidebar.subheader("Global Plot Settings")

def create_sidebar_control(widget_type, label, options=None, index=0, min_val=None, max_val=None, step=None, format_str=None, key_suffix="", help_text=None):
    ss_key = f"{key_suffix}_ss" # session state key
    widget_key = f"{key_suffix}_widget"
    
    current_val = st.session_state[ss_key]
    
    if widget_type == "selectbox": new_val = st.sidebar.selectbox(label, options=options, index=index, key=widget_key, help=help_text)
    elif widget_type == "radio": new_val = st.sidebar.radio(label, options=options, index=index, key=widget_key, help=help_text)
    elif widget_type == "number_input": new_val = st.sidebar.number_input(label, min_value=min_val, max_value=max_val, value=current_val, step=step, format=format_str, key=widget_key, help=help_text)
    elif widget_type == "toggle": new_val = st.sidebar.toggle(label, value=current_val, key=widget_key, help=help_text)
    elif widget_type == "color_picker": new_val = st.sidebar.color_picker(label, value=current_val, key=widget_key, help=help_text)
    else: raise ValueError("Invalid widget type")

    if new_val != current_val:
        st.session_state[ss_key] = new_val
        if ss_key == 'x_axis_type_ss': # Reset zoom if x-axis type changes
            st.session_state.current_xaxis_range = None
        st.rerun()

create_sidebar_control("selectbox", "Plot Type:", options=["Line", "Scatter", "Line and Scatter"], 
                       index=["Line", "Scatter", "Line and Scatter"].index(st.session_state.plot_type), 
                       key_suffix="plot_type", ss_key_override='plot_type') # Use direct session state key

create_sidebar_control("radio", "X-Axis Display:", options=["Elapsed Time", "Time of Day"],
                       index=["Elapsed Time", "Time of Day"].index(st.session_state.x_axis_type),
                       key_suffix="x_axis_type", ss_key_override='x_axis_type')

create_sidebar_control("number_input", "Moving Average Window (s):", min_val=1, max_val=600, 
                       step=1, key_suffix="moving_avg_window", ss_key_override='moving_avg_window')

create_sidebar_control("toggle", "Show Power Difference Plot (F1-F2)", 
                       key_suffix="show_difference_plot", ss_key_override='show_difference_plot')

st.sidebar.markdown("###### Y-Axis Scale & Grid")
if st.sidebar.button("Reset Y-Axis Scale", key='btn_reset_y_k'):
    st.session_state.y_axis_min = 0.0
    st.session_state.y_axis_max = st.session_state.y_axis_default_max 
    st.rerun()

create_sidebar_control("number_input", "Y-Axis Min (W):", format_str="%.1f", 
                       key_suffix="y_axis_min", ss_key_override='y_axis_min')
create_sidebar_control("number_input", "Y-Axis Max (W):", min_val=float(st.session_state.y_axis_min) + 1.0, 
                       format_str="%.1f", key_suffix="y_axis_max", ss_key_override='y_axis_max')
create_sidebar_control("number_input", "Y-Axis Grid Spacing (W):", min_val=5.0, step=5.0, 
                       format_str="%.1f", key_suffix="y_grid_spacing", ss_key_override='y_grid_spacing')
create_sidebar_control("toggle", "Show Y-Axis Gridlines", 
                       key_suffix="show_y_grid", ss_key_override='show_y_grid')

for i in [1, 2]:
    st.sidebar.subheader(f"File {i} Adjustments: {st.session_state[f'file{i}_name']}")
    if st.session_state[f'df{i}_raw'] is not None:
        create_sidebar_control("color_picker", f"Line Color File {i}:", 
                               key_suffix=f"f{i}_color", ss_key_override=f'f{i}_color')
        create_sidebar_control("number_input", f"Shift File {i} (s):", step=0.1, format_str="%.1f", 
                               key_suffix=f"f{i}_shift", ss_key_override=f'f{i}_shift')
        create_sidebar_control("number_input", f"Stretch File {i} (factor):", min_val=0.1, step=0.01, format_str="%.2f", 
                               key_suffix=f"f{i}_stretch", ss_key_override=f'f{i}_stretch')
        if st.sidebar.button(f"Reset File {i} Adjustments", key=f'btn_reset_f{i}_k'):
            st.session_state[f'f{i}_shift'] = 0.0
            st.session_state[f'f{i}_stretch'] = 1.0
            st.rerun()
    else:
        st.sidebar.info(f"Upload File {i} for its adjustment options.")

st.markdown("---")
st.caption("Tip: Zoom by dragging on the graph. Use 'Autoscale' (home icon) in plot modebar to reset zoom.")
