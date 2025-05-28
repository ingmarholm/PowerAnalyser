import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from fitparse import FitFile, FitParseError
import numpy as np
from datetime import timezone, timedelta
from streamlit_plotly_events import plotly_events # Needs: pip install streamlit-plotly-events

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="FIT File Power Analyzer")

# --- IMPORTANT ENVIRONMENT NOTE ---
st.error("""
    **CRITICAL ENVIRONMENT INCOMPATIBILITY DETECTED:**
    Your deployment environment is using Python 3.13+ with an older Streamlit version (e.g., 1.22.0). 
    This combination causes a `ModuleNotFoundError: No module named 'imghdr'` because `imghdr` was 
    removed from Python 3.13, but older Streamlit versions require it.

    **THIS SCRIPT LIKELY WILL NOT RUN CORRECTLY UNTIL THIS IS FIXED IN YOUR ENVIRONMENT.**

    **Solutions (must be done in your deployment environment):**
    1.  **Upgrade Streamlit:** Update Streamlit to version 1.30.0 or newer.
    2.  **Downgrade Python:** Use Python 3.11 or 3.12.

    The code below attempts to provide the requested features but cannot circumvent this fundamental environment issue.
""")
st.warning("""
    **Zoomed Statistics Workaround:**
    Due to limitations in the installed `streamlit-plotly-events==0.0.6` (it does not support `relayout_event` for general zoom), 
    to calculate statistics for a specific window:
    1. Use the **'Box Select'** tool from the graph's modebar (hover over the graph to see it).
    2. Drag a rectangle over the desired region on the graph.
    3. Statistics for this selected region will then be displayed.
    Zooming by other means (scroll, double-click) will not update these specific windowed statistics.
    Use the "Clear Selection & Show Overall Stats" button to revert.
""")


# --- Helper Functions ---
def parse_fit_file(uploaded_file_bytes, timezone_offset_hours):
    if uploaded_file_bytes is None: return None
    try:
        file_bytes_content = uploaded_file_bytes.getvalue() if hasattr(uploaded_file_bytes, 'getvalue') else uploaded_file_bytes
        fitfile = FitFile(file_bytes_content)
        records = []
        for record in fitfile.get_messages('record'):
            timestamp, power = None, None
            for field in record:
                if field.name == 'timestamp': timestamp = field.value
                elif field.name == 'power': power = field.value
            if timestamp is not None and power is not None:
                timestamp = pd.to_datetime(timestamp) if not isinstance(timestamp, pd.Timestamp) else timestamp
                timestamp = timestamp.tz_localize('UTC') if timestamp.tzinfo is None else timestamp.astimezone(timezone.utc)
                if timezone_offset_hours != 0: timestamp += timedelta(hours=timezone_offset_hours)
                records.append({'timestamp': timestamp, 'power': power})
        if not records:
            st.warning(f"No power data in {getattr(uploaded_file_bytes, 'name', 'file')}.")
            return None
        df = pd.DataFrame(records).set_index('timestamp')[['power']]
        df = df.resample('1S').mean().interpolate(method='linear').dropna(subset=['power'])
        return df if not df.empty else None
    except FitParseError as e: st.error(f"FitParseError for {getattr(uploaded_file_bytes, 'name', 'unknown')}: {e}. Corrupt/invalid FIT."); return None
    except Exception as e: st.error(f"Error parsing {getattr(uploaded_file_bytes, 'name', 'unknown')}: {e}"); return None

def process_data(df_raw, stretch_factor, shift_seconds, moving_avg_window_seconds):
    if df_raw is None or df_raw.empty: return None
    df_processed = df_raw.copy()
    if stretch_factor != 1.0 and not df_processed.empty:
        start_time_abs = df_processed.index[0]
        elapsed_seconds_orig = (df_processed.index - start_time_abs).total_seconds().to_numpy()
        power_values_orig = df_processed['power'].to_numpy()
        elapsed_seconds_stretched = elapsed_seconds_orig * stretch_factor
        if len(elapsed_seconds_stretched) > 1:
            min_s, max_s = elapsed_seconds_stretched.min(), elapsed_seconds_stretched.max()
            target_secs = np.arange(min_s, max_s + 1, 1) if min_s <= max_s else np.array([])
            if len(target_secs) > 0:
                unique_secs, idx = np.unique(np.sort(elapsed_seconds_stretched), return_index=True)
                unique_power = power_values_orig[np.argsort(elapsed_seconds_stretched)][idx]
                if len(unique_secs) > 1:
                    interp_power = np.interp(target_secs, unique_secs, unique_power)
                    new_idx = [start_time_abs + pd.Timedelta(seconds=s) for s in target_secs]
                    df_processed = pd.DataFrame({'power': interp_power}, index=pd.DatetimeIndex(new_idx, name='timestamp'))
                else: df_processed = pd.DataFrame(columns=['power']).set_index(pd.DatetimeIndex([], name='timestamp'))
            else: df_processed = pd.DataFrame(columns=['power']).set_index(pd.DatetimeIndex([], name='timestamp'))
        elif len(elapsed_seconds_stretched) == 1:
            df_processed = pd.DataFrame({'power': power_values_orig}, index=pd.DatetimeIndex([start_time_abs + pd.Timedelta(seconds=elapsed_seconds_stretched[0])], name='timestamp'))
        else: df_processed = pd.DataFrame(columns=['power']).set_index(pd.DatetimeIndex([], name='timestamp'))
    if shift_seconds != 0 and not df_processed.empty: df_processed.index += pd.Timedelta(seconds=shift_seconds)
    if moving_avg_window_seconds > 1 and not df_processed.empty:
        df_processed['power'] = df_processed['power'].rolling(window=moving_avg_window_seconds, center=True, min_periods=1).mean()
    if not df_processed.empty: df_processed['elapsed_seconds'] = (df_processed.index - df_processed.index[0]).total_seconds()
    else:
        df_processed['power'] = pd.Series(dtype='float64')
        df_processed['elapsed_seconds'] = pd.Series(dtype='float64')
        if df_processed.index.name != 'timestamp': df_processed.index = pd.DatetimeIndex(df_processed.index, name='timestamp')
    return df_processed

def format_seconds_to_hhmmss(seconds):
    if pd.isna(seconds) or not isinstance(seconds, (int, float)) or seconds < 0: return "00:00:00"
    return f"{int(seconds // 3600):02}:{int((seconds % 3600) // 60):02}:{int(seconds % 60):02}"

# --- Initialize Session State ---
default_states = {
    'df1_raw': None, 'df2_raw': None, 'df1_processed': None, 'df2_processed': None,
    'df1_selected_window': None, 'df2_selected_window': None, # For stats from selected range
    'file1_name': "File 1", 'file2_name': "File 2", 'plot_type': "Line", 'x_axis_type': "Elapsed Time",
    'y_axis_min': 0.0, 'y_axis_max': 500.0, 'y_axis_default_max': 500.0, 'y_grid_spacing': 100.0, 'show_y_grid': True,
    'default_y_max_calculated': False, 'moving_avg_window': 1,
    'f1_color': '#FF0000', 'f1_shift': 0.0, 'f1_stretch': 1.0,
    'f2_color': '#0000FF', 'f2_shift': 0.0, 'f2_stretch': 1.0,
    'show_difference_plot': False, 'timezone_offset': 0.0, 
    'current_selection_range': None, # Stores {'x': [min, max]} from select_event
    'hover_power1': None, 'hover_power2': None, 'hover_x_value_display': None, 'last_hover_x_raw': None,
}
for key, value in default_states.items():
    if key not in st.session_state: st.session_state[key] = value

# --- UI ---
st.title("🚴💨 FIT File Power Analyzer")
st.sidebar.header("📁 File Upload & Settings")
uploaded_file1_obj = st.sidebar.file_uploader("Upload first .fit file", type="fit", key="file1_ul_k")
uploaded_file2_obj = st.sidebar.file_uploader("Upload second .fit file", type="fit", key="file2_ul_k")
st.session_state.timezone_offset = st.sidebar.number_input("Timezone Offset from UTC (hours)", value=st.session_state.timezone_offset, step=0.5, format="%.1f", help="E.g., AEST (UTC+10) is 10.")

# Data Loading & Processing (triggered by file upload or timezone change)
for i, uploaded_obj in enumerate([uploaded_file1_obj, uploaded_file2_obj]):
    df_raw_key, file_name_key, tz_offset_key = f'df{i+1}_raw', f'file{i+1}_name', f'tz_offset_f{i+1}'
    if uploaded_obj and (st.session_state[df_raw_key] is None or st.session_state[file_name_key] != uploaded_obj.name or st.session_state.get(tz_offset_key) != st.session_state.timezone_offset):
        st.session_state[df_raw_key] = parse_fit_file(uploaded_obj, st.session_state.timezone_offset)
        st.session_state[file_name_key] = uploaded_obj.name
        st.session_state[tz_offset_key] = st.session_state.timezone_offset
        st.session_state.default_y_max_calculated = False
        st.session_state.current_selection_range = None # Reset selection on new file

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

# --- Plotting ---
fig = go.Figure()
plot_mode_map = {"Line": "lines", "Scatter": "markers", "Line and Scatter": "lines+markers"}
current_plot_mode, scatter_marker_size = plot_mode_map.get(st.session_state.plot_type, "lines"), 4
x_axis_is_elapsed_time = st.session_state.x_axis_type == "Elapsed Time"

for i, df_proc in enumerate([st.session_state.df1_processed, st.session_state.df2_processed]):
    if df_proc is not None and not df_proc.empty:
        x_data = df_proc['elapsed_seconds'] if x_axis_is_elapsed_time else df_proc.index
        color = st.session_state.f1_color if i == 0 else st.session_state.f2_color
        name = st.session_state.file1_name if i == 0 else st.session_state.file2_name
        fig.add_trace(go.Scatter(x=x_data, y=df_proc['power'], mode=current_plot_mode, name=name, 
                                 line=dict(color=color, width=1), marker=dict(size=scatter_marker_size), hoverinfo='none'))

if st.session_state.show_difference_plot and st.session_state.df1_processed is not None and st.session_state.df2_processed is not None:
    df1_aligned, df2_aligned = st.session_state.df1_processed.align(st.session_state.df2_processed, join='inner', axis=0)
    if not df1_aligned.empty:
        power_diff = df1_aligned['power'] - df2_aligned['power']
        x_data_diff = df1_aligned['elapsed_seconds'] if x_axis_is_elapsed_time else df1_aligned.index
        fig.add_trace(go.Scatter(x=x_data_diff, y=power_diff, mode='lines', name='Power Diff (F1-F2)',
                                 line=dict(color='rgba(0,128,0,0.7)', width=1, dash='dash'), hoverinfo='none'))

x_title = "Elapsed Time (HH:MM:SS)" if x_axis_is_elapsed_time else "Time of Day"
xaxis_config = dict(title_text=x_title, nticks=10, showspikes=True, spikemode='across', spikesnap='cursor', spikethickness=1, spikedash='solid', spikecolor='grey')
if x_axis_is_elapsed_time:
    min_s, max_s = float('inf'), float('-inf')
    for df_p in [st.session_state.df1_processed, st.session_state.df2_processed]:
        if df_p is not None and not df_p.empty and 'elapsed_seconds' in df_p:
            min_s, max_s = min(min_s, df_p['elapsed_seconds'].min()), max(max_s, df_p['elapsed_seconds'].max())
    if np.isfinite(min_s) and np.isfinite(max_s) and max_s > min_s:
        xaxis_config.update(tickvals=np.linspace(min_s, max_s, 10), ticktext=[format_seconds_to_hhmmss(s) for s in np.linspace(min_s, max_s, 10)])
fig.update_xaxes(**xaxis_config)

current_annotations = []
if st.session_state.hover_x_value_display is not None:
    props = dict(xref="paper", x=1.01, yref="y", showarrow=False, align="left", bgcolor="rgba(255,255,255,0.8)", borderpad=3, font=dict(size=11))
    if st.session_state.hover_power1 is not None: current_annotations.append(dict(y=st.session_state.hover_power1, text=f"{st.session_state.hover_power1:.0f}W", font_color=st.session_state.f1_color, **props))
    if st.session_state.hover_power2 is not None: current_annotations.append(dict(y=st.session_state.hover_power2, text=f"{st.session_state.hover_power2:.0f}W", font_color=st.session_state.f2_color, **props))

fig.update_layout(height=550, yaxis_title="Power (Watts)", yaxis_range=[st.session_state.y_axis_min, st.session_state.y_axis_max],
                  yaxis_dtick=st.session_state.y_grid_spacing if st.session_state.show_y_grid else None, yaxis_showgrid=st.session_state.show_y_grid,
                  legend_title_text='Files', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), title="Power Comparison",
                  margin=dict(t=50, b=50, r=120), xaxis_showline=True, yaxis_showline=True, xaxis_linewidth=1, yaxis_linewidth=1,
                  xaxis_linecolor='darkgrey', yaxis_linecolor='darkgrey', hovermode='x unified', annotations=current_annotations,
                  dragmode='select') # Default dragmode to 'select' to encourage its use for stats

plot_event_data = plotly_events(fig, key="main_plot_events_key", click_event=False, hover_event=True, select_event=True, override_height=fig.layout.height)

# --- Event Handling for Hover and SELECTION (for stats) ---
if plot_event_data:
    selection_data = next((event for event in plot_event_data if event.get("event_type") == "select" and "range" in event and "x" in event["range"]), None)
    if selection_data:
        new_selection_range = selection_data["range"]["x"]
        if st.session_state.current_selection_range != new_selection_range: # Avoid rerun if range is same
            st.session_state.current_selection_range = new_selection_range
            st.rerun() # Rerun to update stats based on new selection

    hover_data_points = next((event.get("points") for event in plot_event_data if event.get("event_type") == "hover" and event.get("points")), None)
    if hover_data_points:
        raw_x_val = hover_data_points[0]['x']
        if st.session_state.last_hover_x_raw != raw_x_val:
            st.session_state.last_hover_x_raw = raw_x_val
            st.session_state.hover_power1, st.session_state.hover_power2 = None, None
            if x_axis_is_elapsed_time:
                st.session_state.hover_x_value_display = format_seconds_to_hhmmss(raw_x_val)
                for i, df_p in enumerate([st.session_state.df1_processed, st.session_state.df2_processed]):
                    if df_p is not None and not df_p.empty:
                        power_val = np.interp(raw_x_val, df_p['elapsed_seconds'], df_p['power'])
                        if i == 0: st.session_state.hover_power1 = power_val
                        else: st.session_state.hover_power2 = power_val
            else: # Time of Day
                try:
                    hover_x_dt = pd.to_datetime(raw_x_val)
                    st.session_state.hover_x_value_display = hover_x_dt.strftime('%H:%M:%S')
                    for i, df_p in enumerate([st.session_state.df1_processed, st.session_state.df2_processed]):
                        if df_p is not None and not df_p.empty:
                            temp_idx = df_p.index.union([hover_x_dt])
                            power_val = df_p['power'].reindex(temp_idx).interpolate(method='time').get(hover_x_dt)
                            if i == 0: st.session_state.hover_power1 = power_val
                            else: st.session_state.hover_power2 = power_val
                except Exception: st.session_state.hover_x_value_display = "N/A"
            st.rerun()
    elif not hover_data_points and st.session_state.last_hover_x_raw is not None: # Clear hover if mouse leaves
        st.session_state.hover_power1, st.session_state.hover_power2, st.session_state.hover_x_value_display, st.session_state.last_hover_x_raw = None, None, None, None
        st.rerun()

# Filter data for SELECTED window stats
st.session_state.df1_selected_window, st.session_state.df2_selected_window = st.session_state.df1_processed, st.session_state.df2_processed # Default to all
if st.session_state.current_selection_range:
    x_min, x_max = st.session_state.current_selection_range
    for i, df_proc in enumerate([st.session_state.df1_processed, st.session_state.df2_processed]):
        df_sel_key = f'df{i+1}_selected_window'
        if df_proc is not None and not df_proc.empty:
            if x_axis_is_elapsed_time:
                st.session_state[df_sel_key] = df_proc[(df_proc['elapsed_seconds'] >= x_min) & (df_proc['elapsed_seconds'] <= x_max)]
            else: # Time of Day
                try:
                    tz_info = df_proc.index.tz
                    dt_x_min = pd.to_datetime(x_min).tz_localize('UTC').tz_convert(tz_info) if tz_info and pd.to_datetime(x_min).tzinfo is None else pd.to_datetime(x_min)
                    dt_x_max = pd.to_datetime(x_max).tz_localize('UTC').tz_convert(tz_info) if tz_info and pd.to_datetime(x_max).tzinfo is None else pd.to_datetime(x_max)
                    st.session_state[df_sel_key] = df_proc[(df_proc.index >= dt_x_min) & (df_proc.index <= dt_x_max)]
                except Exception: st.session_state[df_sel_key] = df_proc # Fallback
        else: st.session_state[df_sel_key] = None

# --- Statistics Display ---
st.markdown("---")
st.subheader("📊 Statistics for Selected Window (or Overall if no selection)")
if st.sidebar.button("Clear Selection & Show Overall Stats", key="clear_sel_stats_btn"):
    st.session_state.current_selection_range = None
    st.rerun()

stat_cols = st.columns(3)
selected_duration_str = "N/A"
df_for_sel_dur = st.session_state.df1_selected_window if st.session_state.df1_selected_window is not None and not st.session_state.df1_selected_window.empty else st.session_state.df2_selected_window
if df_for_sel_dur is not None and not df_for_sel_dur.empty:
    if x_axis_is_elapsed_time and 'elapsed_seconds' in df_for_sel_dur:
        sel_dur_secs = df_for_sel_dur['elapsed_seconds'].max() - df_for_sel_dur['elapsed_seconds'].min()
        selected_duration_str = format_seconds_to_hhmmss(sel_dur_secs)
    elif not x_axis_is_elapsed_time:
        sel_dur_secs = (df_for_sel_dur.index.max() - df_for_sel_dur.index.min()).total_seconds()
        selected_duration_str = format_seconds_to_hhmmss(sel_dur_secs)
stat_cols[0].metric("Selected Duration", selected_duration_str)

avg_p1_sel, avg_p2_sel = "N/A", "N/A"
if st.session_state.df1_selected_window is not None and not st.session_state.df1_selected_window.empty: avg_p1_sel = f"{st.session_state.df1_selected_window['power'].mean():.1f} W"
if st.session_state.df2_selected_window is not None and not st.session_state.df2_selected_window.empty: avg_p2_sel = f"{st.session_state.df2_selected_window['power'].mean():.1f} W"
stat_cols[1].markdown(f"**{st.session_state.file1_name} Avg:** <span style='color:{st.session_state.f1_color};'>{avg_p1_sel}</span>", unsafe_allow_html=True)
stat_cols[1].markdown(f"**{st.session_state.file2_name} Avg:** <span style='color:{st.session_state.f2_color};'>{avg_p2_sel}</span>", unsafe_allow_html=True)

corr_sel_str = "N/A"
if st.session_state.df1_selected_window is not None and not st.session_state.df1_selected_window.empty and \
   st.session_state.df2_selected_window is not None and not st.session_state.df2_selected_window.empty:
    df1_sel_aligned, df2_sel_aligned = st.session_state.df1_selected_window.align(st.session_state.df2_selected_window, join='inner', axis=0)
    if not df1_sel_aligned.empty and len(df1_sel_aligned['power']) > 1: # Pearson needs >1 point
        corr = df1_sel_aligned['power'].corr(df2_sel_aligned['power'])
        if pd.notna(corr): corr_sel_str = f"{corr:.3f}"
stat_cols[2].metric("Power Correlation (Selected)", corr_sel_str)

# --- Controls Area (Sidebar) ---
st.sidebar.header("⚙️ Analysis Controls")
st.sidebar.subheader("Global Plot Settings")
def create_sb_control(w_type, lbl, ss_key, opts=None, idx=0, min_v=None, max_v=None, stp=None, fmt=None, help_txt=None):
    curr_v, w_key = st.session_state[ss_key], f"{ss_key}_wk"
    if w_type=="selectbox": new_v = st.sidebar.selectbox(lbl, options=opts, index=idx, key=w_key, help=help_txt)
    elif w_type=="radio": new_v = st.sidebar.radio(lbl, options=opts, index=idx, key=w_key, help=help_txt)
    elif w_type=="number_input": new_v = st.sidebar.number_input(lbl, min_value=min_v, max_value=max_v, value=curr_v, step=stp, format=fmt, key=w_key, help=help_txt)
    elif w_type=="toggle": new_v = st.sidebar.toggle(lbl, value=curr_v, key=w_key, help=help_txt)
    elif w_type=="color_picker": new_v = st.sidebar.color_picker(lbl, value=curr_v, key=w_key, help=help_txt)
    else: raise ValueError(f"Bad widget: {w_type}")
    if new_v != curr_v: st.session_state[ss_key] = new_v; st.rerun()

create_sb_control("selectbox", "Plot Type:", 'plot_type', opts=["Line", "Scatter", "Line and Scatter"], idx=["Line", "Scatter", "Line and Scatter"].index(st.session_state.plot_type))
create_sb_control("radio", "X-Axis Display:", 'x_axis_type', opts=["Elapsed Time", "Time of Day"], idx=["Elapsed Time", "Time of Day"].index(st.session_state.x_axis_type))
create_sb_control("number_input", "Moving Avg Window (s):", 'moving_avg_window', min_v=1, max_v=600, stp=1)
create_sb_control("toggle", "Show Power Diff Plot (F1-F2)", 'show_difference_plot')

st.sidebar.markdown("###### Y-Axis Scale & Grid")
if st.sidebar.button("Reset Y-Axis Scale", key='btn_reset_y_k'):
    st.session_state.y_axis_min, st.session_state.y_axis_max = 0.0, st.session_state.y_axis_default_max; st.rerun()
create_sb_control("number_input", "Y-Axis Min (W):", 'y_axis_min', fmt="%.1f")
create_sb_control("number_input", "Y-Axis Max (W):", 'y_axis_max', min_v=float(st.session_state.y_axis_min) + 1.0, fmt="%.1f")
create_sb_control("number_input", "Y-Grid Spacing (W):", 'y_grid_spacing', min_v=5.0, stp=5.0, fmt="%.1f")
create_sb_control("toggle", "Show Y-Gridlines", 'show_y_grid')

for i in [1, 2]:
    st.sidebar.subheader(f"File {i} Adjustments: {st.session_state[f'file{i}_name']}")
    if st.session_state[f'df{i}_raw'] is not None:
        create_sb_control("color_picker", f"Line Color File {i}:", f'f{i}_color')
        create_sb_control("number_input", f"Shift File {i} (s):", f'f{i}_shift', stp=0.1, fmt="%.1f")
        create_sb_control("number_input", f"Stretch File {i} (factor):", f'f{i}_stretch', min_v=0.1, stp=0.01, fmt="%.2f")
        if st.sidebar.button(f"Reset File {i} Adjustments", key=f'btn_reset_f{i}_k'):
            st.session_state[f'f{i}_shift'], st.session_state[f'f{i}_stretch'] = 0.0, 1.0; st.rerun()
    else: st.sidebar.info(f"Upload File {i} for options.")
st.markdown("---")
st.caption("Tip: Use 'Box Select' on graph for windowed stats. 'Autoscale' (home icon) in modebar resets visual zoom.")
