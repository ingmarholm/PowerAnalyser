import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from fitparse import FitFile, FitParseError
import numpy as np
from datetime import datetime, timezone, timedelta
from timezonefinder import TimezoneFinder
import pytz # For timezone localization

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="FIT File Power Analyzer")

# --- Helper Functions ---
@st.cache_data # Cache the result of parsing to speed up reruns if file bytes are the same
def parse_fit_file_data(uploaded_file_bytes, file_name_for_error):
    if uploaded_file_bytes is None:
        return None, None
    
    records = []
    first_gps_data = None 

    try:
        fitfile = FitFile(uploaded_file_bytes)
        
        for record_msg in fitfile.get_messages(['record', 'lap', 'session']):
            if first_gps_data is None and record_msg.name == 'record':
                lat, lon, ts_gps = None, None, None
                for field_data in record_msg:
                    if field_data.name == 'position_lat' and field_data.value is not None:
                        lat = field_data.value * (180.0 / 2**31) 
                    elif field_data.name == 'position_long' and field_data.value is not None:
                        lon = field_data.value * (180.0 / 2**31) 
                    elif field_data.name == 'timestamp' and field_data.value is not None:
                        ts_gps = field_data.value
                
                if lat is not None and lon is not None and ts_gps is not None:
                    if ts_gps.tzinfo is None: 
                         ts_gps = ts_gps.replace(tzinfo=timezone.utc)
                    first_gps_data = {'latitude': lat, 'longitude': lon, 'timestamp': ts_gps}

            if record_msg.name == 'record':
                timestamp, power = None, None
                for field_data in record_msg:
                    if field_data.name == 'timestamp':
                        timestamp = field_data.value
                    elif field_data.name == 'power':
                        power = field_data.value
                
                if timestamp is not None and power is not None:
                    if timestamp.tzinfo is None: 
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    records.append({'timestamp': timestamp, 'power': power})

        if not records:
            st.warning(f"No power data records found in {file_name_for_error}.")
            return None, first_gps_data 

        df = pd.DataFrame(records)
        if df.empty: # Should be caught by "not records" but as a safeguard
            st.warning(f"No data to process in {file_name_for_error}.")
            return None, first_gps_data

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        df = df.resample('1S').mean() 
        df['power'] = df['power'].interpolate(method='linear')
        df = df.dropna(subset=['power']) 
        
        if df.empty:
            st.warning(f"Resampled power data is empty for {file_name_for_error}. Check file content.")
            return None, first_gps_data
            
        return df, first_gps_data
    
    except FitParseError as e:
        st.error(f"Error parsing FIT file {file_name_for_error} (FitParseError): {e}")
        return None, None
    except Exception as e:
        st.error(f"General error processing FIT file {file_name_for_error}: {e}")
        return None, None


def get_local_timezone_from_gps(gps_data):
    if gps_data:
        tf = TimezoneFinder()
        tz_name = tf.timezone_at(lng=gps_data['longitude'], lat=gps_data['latitude'])
        if tz_name:
            st.info(f"Timezone detected from GPS: {tz_name}")
            return pytz.timezone(tz_name)
        else:
            st.warning("Could not determine timezone from GPS coordinates. Defaulting to UTC for 'Time of Day'.")
    else:
        st.info("No GPS data found in FIT file to determine local timezone. Defaulting to UTC for 'Time of Day'.")
    return pytz.utc


def process_data(df_raw, stretch_factor, shift_seconds, moving_avg_window_seconds, local_tz):
    if df_raw is None or df_raw.empty:
        return None

    df_processed = df_raw.copy() 

    if local_tz != pytz.utc and not df_processed.empty and df_processed.index.tz is not None :
        try:
            df_processed.index = df_processed.index.tz_convert(local_tz)
        except Exception as e: # Handle cases like naive datetime index
            st.warning(f"Could not convert timestamps to local timezone {local_tz}: {e}. Ensuring UTC or keeping as is if naive.")
            if df_processed.index.tz is not None: # If it was tz-aware but failed conversion
                df_processed.index = df_processed.index.tz_convert(pytz.utc)
            # If it was naive, we might leave it naive or assume UTC based on parsing
            # Parsing stage ensures UTC, so this path for naive should be rare.


    if stretch_factor != 1.0 and not df_processed.empty:
        original_tz = df_processed.index.tz
        # Make naive for calculation if it's timezone aware
        df_processed_index_naive = df_processed.index.tz_localize(None) if original_tz else df_processed.index
        
        start_time_abs_naive = df_processed_index_naive[0]
        elapsed_seconds_orig = (df_processed_index_naive - start_time_abs_naive).total_seconds().to_numpy()
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
                    new_indices_naive = [start_time_abs_naive + pd.Timedelta(seconds=s) for s in target_stretched_elapsed_seconds]
                    df_processed = pd.DataFrame({'power': interpolated_power}, index=pd.DatetimeIndex(new_indices_naive))
                else:
                    df_processed = pd.DataFrame(columns=['power']).set_index(pd.DatetimeIndex([])) 
            else:
                df_processed = pd.DataFrame(columns=['power']).set_index(pd.DatetimeIndex([])) 
        elif len(elapsed_seconds_stretched) == 1:
            new_indices_naive = [start_time_abs_naive + pd.Timedelta(seconds=elapsed_seconds_stretched[0])]
            df_processed = pd.DataFrame({'power': power_values_orig}, index=pd.DatetimeIndex(new_indices_naive))
        else:
             df_processed = pd.DataFrame(columns=['power']).set_index(pd.DatetimeIndex([])) 
        
        if original_tz is not None and not df_processed.empty:
            df_processed.index = df_processed.index.tz_localize(original_tz)
        elif not df_processed.empty and df_processed.index.tz is None: # If it became naive and original was also naive or None
             df_processed.index = df_processed.index.tz_localize(pytz.utc) # Default to UTC if somehow lost and was naive

    if shift_seconds != 0 and not df_processed.empty:
        df_processed.index = df_processed.index + pd.Timedelta(seconds=shift_seconds)

    if moving_avg_window_seconds > 1 and not df_processed.empty:
        df_processed['power'] = df_processed['power'].rolling(window=moving_avg_window_seconds, center=True, min_periods=1).mean()

    if not df_processed.empty:
        # Ensure index is DatetimeIndex before calculating elapsed_seconds
        if not isinstance(df_processed.index, pd.DatetimeIndex):
            st.warning("Processed data index is not DatetimeIndex. Cannot calculate elapsed seconds accurately.")
            df_processed['elapsed_seconds'] = pd.Series(dtype='float64') # Empty or NaN series
        else:
            # Ensure index is not naive for subtraction, use UTC if it is
            current_index_for_elapsed = df_processed.index
            if current_index_for_elapsed.tz is None:
                current_index_for_elapsed = current_index_for_elapsed.tz_localize(pytz.utc)
            df_processed['elapsed_seconds'] = (current_index_for_elapsed - current_index_for_elapsed[0]).total_seconds()
    else:
        if 'power' not in df_processed.columns: df_processed['power'] = pd.Series(dtype='float64')
        if 'elapsed_seconds' not in df_processed.columns: df_processed['elapsed_seconds'] = pd.Series(dtype='float64')
        if not isinstance(df_processed.index, pd.DatetimeIndex): 
            df_processed = df_processed.set_index(pd.DatetimeIndex([]))
        if df_processed.index.name != 'timestamp': df_processed.index.name = 'timestamp'


    return df_processed


def format_seconds_to_hhmmss(seconds):
    if pd.isna(seconds) or not isinstance(seconds, (int, float, np.number)): return ""
    return str(timedelta(seconds=int(round(seconds))))


# --- Initialize Session State ---
default_states = {
    'df1_raw': None, 'df2_raw': None, 'gps1_info': None, 'gps2_info': None, 'determined_local_tz': pytz.utc,
    'df1_processed': None, 'df2_processed': None,
    'file1_name': "File 1", 'file2_name': "File 2",
    'plot_type': "Line", 'x_axis_type': "Elapsed Time",
    'y_axis_min': 0.0, 'y_axis_max': 500.0, 'y_axis_min_default': 0.0, 'y_axis_max_default': 500.0,
    'y_grid_spacing': 100.0, 'show_y_grid': True,
    'default_y_max_calculated': False,
    'moving_avg_window': 1,
    'f1_color': '#FF0000', 'f1_shift': 0.0, 'f1_stretch': 1.0,
    'f2_color': '#0000FF', 'f2_shift': 0.0, 'f2_stretch': 1.0,
    'show_difference_plot': False, 'diff_color': '#008000', 
}
for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- UI Layout ---
st.title("🚴💨 FIT File Power Analyzer")
st.markdown("Compare power data from two .fit files. Use the 'Autoscale' (home icon) in the plot's modebar to reset zoom.")

col_uploader1, col_uploader2 = st.columns(2)
with col_uploader1:
    uploaded_file1_bytes = st.file_uploader("Upload first .fit file", type="fit", key="file1_uploader")
with col_uploader2:
    uploaded_file2_bytes = st.file_uploader("Upload second .fit file", type="fit", key="file2_uploader")

if uploaded_file1_bytes and (st.session_state.df1_raw is None or st.session_state.file1_name != uploaded_file1_bytes.name):
    st.session_state.df1_raw, st.session_state.gps1_info = parse_fit_file_data(uploaded_file1_bytes, uploaded_file1_bytes.name)
    st.session_state.file1_name = uploaded_file1_bytes.name
    st.session_state.default_y_max_calculated = False

if uploaded_file2_bytes and (st.session_state.df2_raw is None or st.session_state.file2_name != uploaded_file2_bytes.name):
    st.session_state.df2_raw, st.session_state.gps2_info = parse_fit_file_data(uploaded_file2_bytes, uploaded_file2_bytes.name) # Corrected bracket
    st.session_state.file2_name = uploaded_file2_bytes.name
    st.session_state.default_y_max_calculated = False

if (st.session_state.gps1_info or st.session_state.gps2_info) and st.session_state.determined_local_tz == pytz.utc:
    active_gps_info = st.session_state.gps1_info if st.session_state.gps1_info else st.session_state.gps2_info
    if active_gps_info:
        st.session_state.determined_local_tz = get_local_timezone_from_gps(active_gps_info)

if not st.session_state.default_y_max_calculated and (st.session_state.df1_raw is not None or st.session_state.df2_raw is not None):
    max_p1 = 0
    if st.session_state.df1_raw is not None and not st.session_state.df1_raw.empty and 'power' in st.session_state.df1_raw.columns:
        max_p1 = st.session_state.df1_raw['power'].max()
    max_p2 = 0
    if st.session_state.df2_raw is not None and not st.session_state.df2_raw.empty and 'power' in st.session_state.df2_raw.columns:
        max_p2 = st.session_state.df2_raw['power'].max()
    
    global_max_power = max(max_p1, max_p2, 100) 
    st.session_state.y_axis_max_default = float(np.ceil(global_max_power / 100.0) * 100.0)
    st.session_state.y_axis_min_default = 0.0
    st.session_state.y_axis_max = st.session_state.y_axis_max_default
    st.session_state.y_axis_min = st.session_state.y_axis_min_default
    st.session_state.default_y_max_calculated = True

st.session_state.df1_processed = process_data(st.session_state.df1_raw, st.session_state.f1_stretch, st.session_state.f1_shift, st.session_state.moving_avg_window, st.session_state.determined_local_tz)
st.session_state.df2_processed = process_data(st.session_state.df2_raw, st.session_state.f2_stretch, st.session_state.f2_shift, st.session_state.moving_avg_window, st.session_state.determined_local_tz)

# --- Plotting Area ---
plot_placeholder = st.empty()
fig = go.Figure()
plot_mode_map = {"Line": "lines", "Scatter": "markers", "Line and Scatter": "lines+markers"}
current_plot_mode = plot_mode_map.get(st.session_state.plot_type, "lines")
marker_size = 3 if "Scatter" in st.session_state.plot_type else 6 

x_axis_is_elapsed_time = st.session_state.x_axis_type == "Elapsed Time"

# Add traces
if st.session_state.df1_processed is not None and not st.session_state.df1_processed.empty and 'power' in st.session_state.df1_processed.columns:
    x_data1 = st.session_state.df1_processed['elapsed_seconds'] if x_axis_is_elapsed_time else st.session_state.df1_processed.index
    if not x_data1.empty: # Ensure x_data is not empty
        fig.add_trace(go.Scatter(
            x=x_data1, y=st.session_state.df1_processed['power'], mode=current_plot_mode, 
            name=st.session_state.file1_name, line=dict(color=st.session_state.f1_color, width=1),
            marker=dict(size=marker_size)
        ))

if st.session_state.df2_processed is not None and not st.session_state.df2_processed.empty and 'power' in st.session_state.df2_processed.columns:
    x_data2 = st.session_state.df2_processed['elapsed_seconds'] if x_axis_is_elapsed_time else st.session_state.df2_processed.index
    if not x_data2.empty: # Ensure x_data is not empty
        fig.add_trace(go.Scatter(
            x=x_data2, y=st.session_state.df2_processed['power'], mode=current_plot_mode, 
            name=st.session_state.file2_name, line=dict(color=st.session_state.f2_color, width=1),
            marker=dict(size=marker_size)
        ))

if st.session_state.show_difference_plot and \
   st.session_state.df1_processed is not None and not st.session_state.df1_processed.empty and \
   st.session_state.df2_processed is not None and not st.session_state.df2_processed.empty:
    
    merged_df = None
    if x_axis_is_elapsed_time and 'elapsed_seconds' in st.session_state.df1_processed and 'elapsed_seconds' in st.session_state.df2_processed:
        merged_df = pd.merge(
            st.session_state.df1_processed[['elapsed_seconds', 'power']].rename(columns={'power': 'power1'}),
            st.session_state.df2_processed[['elapsed_seconds', 'power']].rename(columns={'power': 'power2'}),
            on='elapsed_seconds', how='inner'
        )
    elif not x_axis_is_elapsed_time: 
        df1_tod = st.session_state.df1_processed.copy()
        df2_tod = st.session_state.df2_processed.copy()
        common_tz = st.session_state.determined_local_tz
        if df1_tod.index.tz != common_tz: df1_tod.index = df1_tod.index.tz_convert(common_tz)
        if df2_tod.index.tz != common_tz: df2_tod.index = df2_tod.index.tz_convert(common_tz)
        merged_df = pd.merge(
            df1_tod[['power']].rename(columns={'power': 'power1'}),
            df2_tod[['power']].rename(columns={'power': 'power2'}),
            left_index=True, right_index=True, how='inner'
        )
    
    if merged_df is not None and not merged_df.empty:
        merged_df['power_diff'] = merged_df['power1'] - merged_df['power2']
        x_data_diff = merged_df['elapsed_seconds'] if x_axis_is_elapsed_time else merged_df.index
        y_data_diff = merged_df['power_diff']

        if not y_data_diff.empty:
            fig.add_trace(go.Scatter(
                x=x_data_diff, y=y_data_diff, mode='lines',
                name='Difference (F1-F2)', line=dict(color=st.session_state.diff_color, width=1, dash='dash'),
                yaxis="y2" 
            ))

x_title = "Elapsed Time (HH:MM:SS)" if x_axis_is_elapsed_time else f"Time of Day ({st.session_state.determined_local_tz})"
if x_axis_is_elapsed_time:
    min_sec_all, max_sec_all = None, None
    all_elapsed_seconds_series = []
    if st.session_state.df1_processed is not None and not st.session_state.df1_processed.empty and 'elapsed_seconds' in st.session_state.df1_processed:
        all_elapsed_seconds_series.append(st.session_state.df1_processed['elapsed_seconds'])
    if st.session_state.df2_processed is not None and not st.session_state.df2_processed.empty and 'elapsed_seconds' in st.session_state.df2_processed:
        all_elapsed_seconds_series.append(st.session_state.df2_processed['elapsed_seconds'])
    
    if all_elapsed_seconds_series:
        combined_elapsed_seconds = pd.concat(all_elapsed_seconds_series).dropna()
        if not combined_elapsed_seconds.empty:
            min_sec_all = combined_elapsed_seconds.min()
            max_sec_all = combined_elapsed_seconds.max()

    if min_sec_all is not None and max_sec_all is not None:
        if max_sec_all == min_sec_all : # Single point in time
             tickvals = [min_sec_all]
             ticktext = [format_seconds_to_hhmmss(min_sec_all)]
        else: # Range of time
            tickvals = np.linspace(min_sec_all, max_sec_all, num=10)
            ticktext = [format_seconds_to_hhmmss(s) for s in tickvals]
        fig.update_xaxes(title_text=x_title, tickvals=tickvals, ticktext=ticktext,
                         showspikes=True, spikemode='across', spikesnap='cursor', 
                         spikedash='solid', spikethickness=1)
    else: # Fallback if no valid elapsed seconds data
        fig.update_xaxes(title_text=x_title, nticks=10,
                         showspikes=True, spikemode='across', spikesnap='cursor',
                         spikedash='solid', spikethickness=1)
else: # Time of Day
    fig.update_xaxes(title_text=x_title, nticks=10,
                     showspikes=True, spikemode='across', spikesnap='cursor',
                     spikedash='solid', spikethickness=1)

fig.update_layout(
    height=550,
    yaxis_title="Power (Watts)",
    yaxis=dict(
        range=[st.session_state.y_axis_min, st.session_state.y_axis_max],
        dtick=st.session_state.y_grid_spacing if st.session_state.show_y_grid else None,
        showgrid=st.session_state.show_y_grid,
    ),
    legend_title_text='Files',
    title="Power Comparison",
    margin=dict(t=50, b=50, l=50, r=100), 
    plot_bgcolor='rgba(0,0,0,0)', 
    paper_bgcolor='rgba(0,0,0,0)', 
    hovermode='x unified', 
    hoverlabel=dict(bgcolor="rgba(255,255,255,0.1)", font_size=1, bordercolor="rgba(255,255,255,0)")
)
fig.update_yaxes(showspikes=False) 
fig.update_xaxes(showline=True, linewidth=1, linecolor='grey', mirror=True)
fig.update_yaxes(showline=True, linewidth=1, linecolor='grey', mirror=True)

if st.session_state.show_difference_plot:
    max_abs_diff = 0
    if 'merged_df' in locals() and merged_df is not None and not merged_df.empty and 'power_diff' in merged_df: 
        if not merged_df['power_diff'].empty:
             max_abs_diff = merged_df['power_diff'].abs().max()
    diff_axis_range = [-max_abs_diff * 1.1, max_abs_diff * 1.1] if max_abs_diff > 0 else [-50, 50]
    fig.update_layout(
        yaxis2=dict(
            title="Power Difference (W)", overlaying="y", side="right",
            showgrid=False, range=diff_axis_range, 
        )
    )
plot_placeholder.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("📊 Summary of Processed Data (Entire Duration)")
summary_cols = st.columns(2)
avg_power1, duration_str1 = "N/A", "N/A"
if st.session_state.df1_processed is not None and not st.session_state.df1_processed.empty and 'power' in st.session_state.df1_processed and 'elapsed_seconds' in st.session_state.df1_processed and not st.session_state.df1_processed['elapsed_seconds'].empty:
    avg_power1 = f"{st.session_state.df1_processed['power'].mean():.1f} W"
    total_seconds1 = st.session_state.df1_processed['elapsed_seconds'].iloc[-1] - st.session_state.df1_processed['elapsed_seconds'].iloc[0]
    duration_str1 = format_seconds_to_hhmmss(total_seconds1)
with summary_cols[0]:
    st.markdown(f"**{st.session_state.file1_name}:**")
    st.markdown(f"<span style='color:{st.session_state.f1_color};'>Avg Power: {avg_power1}</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{st.session_state.f1_color};'>Total Duration: {duration_str1}</span>", unsafe_allow_html=True)

avg_power2, duration_str2 = "N/A", "N/A"
if st.session_state.df2_processed is not None and not st.session_state.df2_processed.empty and 'power' in st.session_state.df2_processed and 'elapsed_seconds' in st.session_state.df2_processed and not st.session_state.df2_processed['elapsed_seconds'].empty:
    avg_power2 = f"{st.session_state.df2_processed['power'].mean():.1f} W"
    total_seconds2 = st.session_state.df2_processed['elapsed_seconds'].iloc[-1] - st.session_state.df2_processed['elapsed_seconds'].iloc[0]
    duration_str2 = format_seconds_to_hhmmss(total_seconds2)
with summary_cols[1]:
    st.markdown(f"**{st.session_state.file2_name}:**")
    st.markdown(f"<span style='color:{st.session_state.f2_color};'>Avg Power: {avg_power2}</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{st.session_state.f2_color};'>Total Duration: {duration_str2}</span>", unsafe_allow_html=True)

st.markdown("---")
st.header("⚙️ Analysis Controls")
control_col1, control_col2, control_col3 = st.columns(3)

with control_col1:
    st.subheader("Global Plot Settings")
    st.session_state.plot_type = st.selectbox("Plot Type:", options=["Line", "Scatter", "Line and Scatter"], 
                                              index=["Line", "Scatter", "Line and Scatter"].index(st.session_state.plot_type),
                                              key='sb_plot_type_k')
    st.session_state.x_axis_type = st.radio("X-Axis Display:", options=["Elapsed Time", "Time of Day"],
                                             index=["Elapsed Time", "Time of Day"].index(st.session_state.x_axis_type),
                                             key='rb_x_axis_k')
    st.session_state.moving_avg_window = st.number_input("Moving Average Window (seconds, 1 for none):", 
                                                         min_value=1, max_value=600, value=st.session_state.moving_avg_window, 
                                                         step=1, key='ni_mov_avg_k')
    st.session_state.show_difference_plot = st.checkbox( 
        "Show Power Difference Plot (File1 - File2)", 
        value=st.session_state.show_difference_plot, 
        key='cb_diff_plot_k' 
    )
    if st.session_state.show_difference_plot: 
        st.session_state.diff_color = st.color_picker(
            "Difference Line Color:", 
            st.session_state.diff_color, 
            key='cp_diff_color_k'
        )
    st.markdown("###### Y-Axis Scale & Grid")
    if st.button("Reset Y-Axis Scale to Default", key='btn_reset_y_scale_k'):
        st.session_state.y_axis_min = st.session_state.y_axis_min_default
        st.session_state.y_axis_max = st.session_state.y_axis_max_default
    st.session_state.y_axis_min = st.number_input("Y-Axis Min (W):", value=float(st.session_state.y_axis_min), key='ni_y_min_k', format="%.1f")
    st.session_state.y_axis_max = st.number_input("Y-Axis Max (W):", value=float(st.session_state.y_axis_max), 
                                                  min_value=float(st.session_state.y_axis_min) + 1.0, key='ni_y_max_k', format="%.1f")
    st.session_state.y_grid_spacing = st.number_input("Y-Axis Grid Spacing (W):", min_value=5.0, 
                                                      value=float(st.session_state.y_grid_spacing), step=5.0, key='ni_y_grid_space_k', format="%.1f")
    st.session_state.show_y_grid = st.checkbox( 
        "Show Y-Axis Gridlines", 
        value=st.session_state.show_y_grid, 
        key='cb_y_grid_k' 
    )
with control_col2:
    st.subheader(f"File 1 Adjustments: {st.session_state.file1_name}")
    if st.session_state.df1_raw is not None:
        st.session_state.f1_color = st.color_picker("Line Color File 1:", st.session_state.f1_color, key='cp_f1_color_k')
        st.session_state.f1_shift = st.number_input("Shift File 1 (seconds, +/-):", value=st.session_state.f1_shift, 
                                                    step=0.1, format="%.1f", key='ni_f1_shift_k')
        st.session_state.f1_stretch = st.number_input("Stretch File 1 (factor):", value=st.session_state.f1_stretch, 
                                                      min_value=0.1, step=0.01, format="%.2f", key='ni_f1_stretch_k')
        if st.button("Reset File 1 Adjustments", key='btn_reset_f1_k'):
            st.session_state.f1_shift = 0.0
            st.session_state.f1_stretch = 1.0
    else:
        st.info("Upload File 1 to see adjustment options.")
with control_col3:
    st.subheader(f"File 2 Adjustments: {st.session_state.file2_name}")
    if st.session_state.df2_raw is not None:
        st.session_state.f2_color = st.color_picker("Line Color File 2:", st.session_state.f2_color, key='cp_f2_color_k')
        st.session_state.f2_shift = st.number_input("Shift File 2 (seconds, +/-):", value=st.session_state.f2_shift, 
                                                    step=0.1, format="%.1f", key='ni_f2_shift_k')
        st.session_state.f2_stretch = st.number_input("Stretch File 2 (factor):", value=st.session_state.f2_stretch, 
                                                      min_value=0.1, step=0.01, format="%.2f", key='ni_f2_stretch_k')
        if st.button("Reset File 2 Adjustments", key='btn_reset_f2_k'):
            st.session_state.f2_shift = 0.0
            st.session_state.f2_stretch = 1.0
    else:
        st.info("Upload File 2 to see adjustment options.")

st.markdown("---")
st.caption("Tip: Hover over the graph to see controls like zoom, pan, and autoscale (reset zoom). Spikeline shows values in hoverbox.")
