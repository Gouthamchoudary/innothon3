"""
EED SmartGrid Analytics - Professional Dashboard
=================================================
Enterprise-grade interactive energy analytics dashboard.
NIT Warangal, EED Building.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="EED SmartGrid Analytics",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg'><text y='32' font-size='32'>E</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DESIGN SYSTEM
# ============================================================================
# Professional color palette - no neon, no garish
ACCENT_PRIMARY = "#3b82f6"      # Clean blue
ACCENT_SECONDARY = "#8b5cf6"    # Purple
ACCENT_SUCCESS = "#10b981"      # Emerald
ACCENT_WARNING = "#f59e0b"      # Amber
ACCENT_DANGER = "#ef4444"       # Red
ACCENT_SOLAR = "#eab308"        # Gold
BG_CARD = "rgba(17, 24, 39, 0.7)"
BG_SURFACE = "rgba(31, 41, 55, 0.5)"
TEXT_PRIMARY = "#f9fafb"
TEXT_SECONDARY = "rgba(156, 163, 175, 1)"
BORDER = "rgba(75, 85, 99, 0.4)"

CHART_COLORS = [
    "#3b82f6", "#8b5cf6", "#10b981", "#f59e0b",
    "#ef4444", "#06b6d4", "#ec4899", "#84cc16",
    "#f97316", "#6366f1"
]

# ============================================================================
# PREMIUM CSS  (bespoke elements only — global theme is in .streamlit/config.toml)
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 40%, #0f172a 100%);
    }
    .stApp > header { background: transparent; }

    /* Section labels */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 4px;
    }

    /* Voltage health badge */
    .health-ok  { color: #10b981; font-weight: 700; }
    .health-bad { color: #ef4444; font-weight: 700; }

    /* Explanation boxes */
    .explain-box {
        background: rgba(30,41,59,0.6);
        border-left: 3px solid #3b82f6;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.82rem;
        color: #94a3b8;
        margin-bottom: 12px;
        line-height: 1.6;
    }
    .explain-box b { color: #f1f5f9; }

    /* DataFrames */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    hr { margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# CHART LAYOUT DEFAULTS
# ============================================================================
def base_layout(title=None, height=400, xaxis_title="", yaxis_title=""):
    """Standard professional chart layout."""
    layout = dict(
        template="plotly_dark",
        height=height,
        margin=dict(l=48, r=24, t=40 if title else 16, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        title=dict(text=title, font=dict(size=14, color="#f1f5f9"), x=0, xanchor="left") if title else None,
        xaxis=dict(
            title=xaxis_title, gridcolor="rgba(51,65,85,0.3)",
            zeroline=False, showline=True, linecolor="rgba(51,65,85,0.5)"
        ),
        yaxis=dict(
            title=yaxis_title, gridcolor="rgba(51,65,85,0.3)",
            zeroline=False, showline=True, linecolor="rgba(51,65,85,0.5)"
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            font=dict(size=10, color="#94a3b8"),
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        hoverlabel=dict(
            bgcolor="#1e293b", bordercolor="rgba(51,65,85,0.5)",
            font=dict(family="Inter", size=11, color="#f1f5f9")
        ),
    )
    return layout


# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base_dir)
    from data_pipeline import load_all_data
    return load_all_data()


@st.cache_data(ttl=600, show_spinner=False)
def load_model_results():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results = {}
    for key, fname in [('comparison', 'model_comparison.csv'), ('predictions', 'predictions.csv'),
                        ('feature_importances', 'feature_importances.csv'), ('anomalies', 'anomalies.csv')]:
        try:
            kw = {'index_col': 0} if key == 'feature_importances' else {}
            results[key] = pd.read_csv(os.path.join(base_dir, fname), **kw)
        except:
            results[key] = None
    return results


@st.cache_data(ttl=600, show_spinner=False)
def run_anomaly_detection(df_merged):
    """Run Isolation Forest anomaly detection inline."""
    features = ['active_power_kw', 'current_avg', 'voltage_ll_avg', 'power_factor', 'reactive_power_kvar']
    avail = [f for f in features if f in df_merged.columns]
    df = df_merged.dropna(subset=avail).copy()
    X = df[avail].values

    iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=100, n_jobs=-1)
    df['anomaly_label'] = iso.fit_predict(X)
    df['anomaly_score'] = iso.decision_function(X)

    anomalies = df[df['anomaly_label'] == -1].copy()
    anomalies['severity'] = pd.cut(
        anomalies['anomaly_score'],
        bins=[-np.inf, -0.3, -0.15, 0],
        labels=['Critical', 'Warning', 'Minor']
    )
    return df, anomalies


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 24px 0 16px;">
        <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #3b82f6, #8b5cf6);
                    border-radius: 12px; display: inline-flex; align-items: center; justify-content: center;
                    font-size: 1.5rem; font-weight: 800; color: white; margin-bottom: 12px;">E</div>
        <h2 style="font-size: 1.1rem; margin: 0; color: #f1f5f9; font-weight: 700;">EED SmartGrid</h2>
        <p style="font-size: 0.75rem; color: #64748b; margin: 4px 0 0;">Energy Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-label" style="padding:0 4px">Institution</p>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size: 0.85rem; color: #e2e8f0; margin: 0 0 2px; font-weight: 500;">NIT Warangal — EED Building</p>
    <p style="font-size: 0.75rem; color: #94a3b8; margin:0;">Dec 2025 — Mar 2026 &nbsp;|&nbsp; 15-min intervals</p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-label" style="padding:0 4px">Date Filter</p>', unsafe_allow_html=True)
    _date_range = st.date_input(
        "Select date range",
        value=[],
        label_visibility="collapsed",
        key="sidebar_date_range"
    )
    SIDEBAR_DATE_START = pd.Timestamp(_date_range[0]) if len(_date_range) >= 1 else None
    SIDEBAR_DATE_END   = pd.Timestamp(_date_range[1]) if len(_date_range) >= 2 else None

    st.markdown("---")
    st.markdown('<p class="section-label" style="padding:0 4px">Feeder Filter</p>', unsafe_allow_html=True)
    _ALL_FEEDERS = [
        "EED_Incomer_1", "EED_Incomer_2", "Research_Wing_Incomer_1",
        "Research_Wing_Incomer_2", "EED_Load_Feeder", "Civil_Load_Feeder",
        "Research_Wing_Solar", "EED_Solar"
    ]
    SIDEBAR_FEEDERS = st.multiselect(
        "Feeders", _ALL_FEEDERS, default=_ALL_FEEDERS,
        label_visibility="collapsed", key="sidebar_feeders"
    )
    if not SIDEBAR_FEEDERS:
        SIDEBAR_FEEDERS = _ALL_FEEDERS  # fallback: all

    st.markdown("---")
    st.markdown("""
    <div style="padding: 0 4px;">
        <p style="font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
                  color: #64748b; margin-bottom: 6px;">Data Sources</p>
        <p style="font-size: 0.75rem; color: #94a3b8; line-height: 1.7; margin:0;">
            Grid Incomer Readings<br>Solar Generation Data<br>
            8 Feeder-Level Meters<br>32 Daily Reports
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; font-size: 0.65rem; color: #475569; padding: 4px 0;">
        Streamlit &middot; Plotly &middot; scikit-learn<br>XGBoost &middot; TensorFlow
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN
# ============================================================================
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="font-size: 2rem; font-weight: 800; letter-spacing: -0.03em;
               background: linear-gradient(90deg, #3b82f6, #8b5cf6);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               margin-bottom: 4px;">EED SmartGrid Analytics</h1>
    <p style="font-size: 0.85rem; color: #64748b; font-weight: 400;">
        ML-Powered Energy Intelligence Platform &middot; NIT Warangal</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading energy data..."):
    try:
        data = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Run `python main.py` first to generate processed data.")
        st.stop()

ml_data = load_model_results()

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Consumption Analytics",
    "ML Predictions",
    "Solar Analytics",
    "Anomaly Detection",
    "Forecasting"
])


# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================
with tab1:
    # Dashboard introduction
    st.markdown("""
    <div class="explain-box">
    <b>About this Dashboard:</b> This platform analyses 15-minute interval electrical measurements from the
    EED Building at National Institute of Technology, Warangal (Dec 2025 – Mar 2026). Data spans the grid incomer,
    8 feeder-level meters, and a rooftop solar generation system — enabling real-time load profiling,
    anomaly detection, and energy forecasting.
    </div>
    """, unsafe_allow_html=True)

    df_merged = data['merged'].copy()
    df_daily  = data['daily'].copy()

    # Apply sidebar date filter
    if SIDEBAR_DATE_START:
        df_merged = df_merged[df_merged['timestamp'] >= SIDEBAR_DATE_START]
        df_daily  = df_daily[df_daily['date'] >= SIDEBAR_DATE_START]
    if SIDEBAR_DATE_END:
        df_merged = df_merged[df_merged['timestamp'] <= SIDEBAR_DATE_END + pd.Timedelta(days=1)]
        df_daily  = df_daily[df_daily['date'] <= SIDEBAR_DATE_END]

    # Apply feeder filter to daily data
    if SIDEBAR_FEEDERS:
        df_daily = df_daily[df_daily['feeder'].isin(SIDEBAR_FEEDERS)]

    if df_merged.empty:
        st.warning("No data for the selected date range. Please widen the filter.")
        st.stop()

    # --- KPIs ---
    total_energy = df_merged['active_energy_kwh'].max() - df_merged['active_energy_kwh'].min()
    avg_power    = df_merged['active_power_kw'].mean()
    peak_power   = df_merged['active_power_kw'].max()
    avg_pf       = df_merged['power_factor'].mean()
    num_days     = max(1, (df_merged['timestamp'].max() - df_merged['timestamp'].min()).days)
    par          = peak_power / avg_power if avg_power > 0 else 0
    pf_compliance = (df_merged['power_factor'] >= 0.95).mean() * 100

    # Net grid import (grid - solar generation)
    if 'solar_active_power_kw' in df_merged.columns:
        solar_gen_total = df_merged.loc[df_merged['solar_active_power_kw'] > 0, 'solar_active_power_kw'].sum() * 0.25  # kWh
        net_import = total_energy - solar_gen_total
    else:
        solar_gen_total = 0
        net_import = total_energy

    st.markdown('<p class="section-label">Key Performance Indicators</p>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Total Energy",     f"{total_energy:,.0f} kWh",
              help="Cumulative active energy consumed (max - min of energy meter reading) over the selected period.")
    c2.metric("Avg Power",        f"{avg_power:.2f} kW",
              help="Mean active power demand across all 15-minute intervals in the selected period.")
    c3.metric("Peak Demand",      f"{peak_power:.2f} kW",
              help="Single highest active power reading recorded in the selected period.")
    c4.metric("Power Factor",     f"{avg_pf:.3f}",
              help="Average power factor. Target is > 0.95 per IS/IEC standards. Low PF increases reactive losses.")
    c5.metric("Data Points",      f"{len(df_merged):,}",
              help="Total number of 15-minute interval records in the selected period.")
    c6.metric("Peak-to-Avg Ratio", f"{par:.2f}x",
              help="Peak-to-Average Ratio (PAR): how much the peak demand exceeds average. Lower is better for grid stability.")
    c7.metric("PF Compliance",    f"{pf_compliance:.1f}%",
              help="Percentage of 15-minute intervals where the power factor met or exceeded the 0.95 target.")
    c8.metric("Net Grid Import",  f"{net_import:,.0f} kWh",
              help="Total energy drawn from the grid after subtracting solar generation contribution.")

    st.markdown("---")

    # --- Voltage Health Card ---
    v_avg = df_merged['voltage_ll_avg'].mean()
    v_ref = 415.0  # Indian LT standard L-L voltage
    v_dev_pct = abs(v_avg - v_ref) / v_ref * 100
    v_status = "NORMAL" if v_dev_pct <= 6.0 else "ALERT"
    v_cls    = "health-ok" if v_status == "NORMAL" else "health-bad"
    st.markdown(f"""
    <div class="explain-box">
    <b>Voltage Health:</b> Average line-to-line voltage is <b>{v_avg:.1f} V</b>
    ({'+' if v_avg >= v_ref else ''}{v_avg - v_ref:.1f} V from nominal 415 V).
    Deviation: <b>{v_dev_pct:.2f}%</b> &nbsp; &rarr; &nbsp;
    <span class="{v_cls}">{v_status}</span>
    (IS/IEC tolerance: ±6% of nominal 415 V).
    </div>
    """, unsafe_allow_html=True)

    # Power consumption time series with range slider
    fig_power = go.Figure()
    fig_power.add_trace(go.Scatter(
        x=df_merged['timestamp'], y=df_merged['active_power_kw'],
        mode='lines', name='Active Power',
        line=dict(color=ACCENT_PRIMARY, width=1.2),
        fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.08)',
    ))
    fig_power.update_layout(**base_layout(title="Active Power Demand (kW)", height=380, yaxis_title="kW"))
    fig_power.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), type="date")
    st.plotly_chart(fig_power, use_container_width=True)
    st.caption("15-minute interval active power drawn from the EED Building grid incomer. Use the range slider below the chart to zoom into specific periods.")

    col_l, col_r = st.columns(2)

    with col_l:
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(
            x=df_merged['timestamp'], y=df_merged['voltage_ll_avg'],
            mode='lines', name='Voltage L-L Avg',
            line=dict(color=ACCENT_DANGER, width=1),
        ))
        # Tolerance bands
        fig_v.add_hline(y=415*1.06, line_dash="dot", line_color="rgba(245,158,11,0.5)",
                        annotation_text="+6%", annotation_position="top right")
        fig_v.add_hline(y=415*0.94, line_dash="dot", line_color="rgba(245,158,11,0.5)",
                        annotation_text="-6%", annotation_position="bottom right")
        fig_v.update_layout(**base_layout(title="Voltage Profile (V)", height=320, yaxis_title="Volts"))
        st.plotly_chart(fig_v, use_container_width=True)
        st.caption("Line-to-line average voltage (V). Dashed bands show ±6% tolerance from 415 V nominal (IS 12360).")

    with col_r:
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(
            x=df_merged['timestamp'], y=df_merged['current_avg'],
            mode='lines', name='Current Avg',
            line=dict(color=ACCENT_SUCCESS, width=1),
        ))
        fig_c.update_layout(**base_layout(title="Current Profile (A)", height=320, yaxis_title="Amps"))
        st.plotly_chart(fig_c, use_container_width=True)
        st.caption("Average three-phase current drawn at the incomer. Spikes indicate sudden load increases or equipment startup.")

    # Daily feeder stacked bar
    if len(df_daily) > 0:
        fig_feeder = px.bar(
            df_daily, x='date', y='daily_kwh', color='feeder',
            color_discrete_sequence=CHART_COLORS, barmode='stack',
        )
        fig_feeder.update_layout(**base_layout(title="Daily Feeder Consumption (kWh)", height=420, yaxis_title="kWh"))
        fig_feeder.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5,
                        font=dict(size=9)))
        st.plotly_chart(fig_feeder, use_container_width=True)
        st.caption("Daily energy consumption (kWh) broken down by feeder. Each color represents one of the 8 sub-feeders. Use the Feeder Filter in the sidebar to isolate specific circuits.")


# ============================================================================
# TAB 2: CONSUMPTION ANALYTICS
# ============================================================================
with tab2:
    st.markdown("""
    <div class="explain-box">
    <b>Consumption Analytics</b> reveals patterns in energy demand — by hour, day of week, and a
    combined heatmap. The <b>Power Factor Distribution</b> and <b>Power Triangle</b> expose reactive
    power quality, critical for minimising utility penalties under Indian Electricity Act norms.
    </div>
    """, unsafe_allow_html=True)

    df_m = data['merged'].dropna(subset=['active_power_kw']).copy()

    col1, col2 = st.columns(2)

    with col1:
        hourly = df_m.groupby('hour')['active_power_kw'].agg(['mean', 'std', 'max']).reset_index()
        fig_h = go.Figure()
        fig_h.add_trace(go.Bar(
            x=hourly['hour'], y=hourly['mean'], name='Average',
            marker=dict(
                color=hourly['mean'],
                colorscale=[[0, '#1e3a5f'], [0.5, '#3b82f6'], [1, '#93c5fd']],
                cornerradius=4
            ),
            error_y=dict(type='data', array=hourly['std'].values, visible=True,
                         color='rgba(148,163,184,0.4)', thickness=1)
        ))
        fig_h.add_trace(go.Scatter(
            x=hourly['hour'], y=hourly['max'], name='Peak',
            mode='markers+lines', line=dict(color=ACCENT_DANGER, width=1.5, dash='dot'),
            marker=dict(size=4, color=ACCENT_DANGER),
        ))
        fig_h.update_layout(**base_layout(title="Hourly Load Profile", height=400,
                                          xaxis_title="Hour of Day", yaxis_title="Power (kW)"))
        st.plotly_chart(fig_h, use_container_width=True)
        st.caption("Average (bar) and peak (dashed line) active power demand per hour. Error bars show 1σ spread across all days. Reveals consistent peak windows for demand-side management.")

    with col2:
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dow = df_m.groupby('day_of_week')['active_power_kw'].agg(['mean', 'std']).reset_index()
        dow['day_name'] = [dow_names[i] for i in dow['day_of_week']]
        colors = [ACCENT_PRIMARY if i < 5 else ACCENT_SECONDARY for i in dow['day_of_week']]
        fig_dow = go.Figure()
        fig_dow.add_trace(go.Bar(
            x=dow['day_name'], y=dow['mean'], name='Average',
            marker=dict(color=colors, cornerradius=4),
            error_y=dict(type='data', array=dow['std'].values, visible=True,
                         color='rgba(148,163,184,0.4)', thickness=1)
        ))
        fig_dow.update_layout(**base_layout(title="Day-of-Week Load Pattern", height=400,
                                            xaxis_title="", yaxis_title="Avg Power (kW)"))
        st.plotly_chart(fig_dow, use_container_width=True)
        st.caption("Average load by day of week. Blue = weekday, purple = weekend. Sharp drop on weekends confirms usage is primarily academic/lab-driven.")

    # Heatmap — RdYlGn_r: Red = high load, Green = low load
    heatmap_data = df_m.groupby(['day_of_week', 'hour'])['active_power_kw'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='active_power_kw')
    heatmap_pivot.index = [dow_names[i] for i in heatmap_pivot.index]

    fig_heat = px.imshow(
        heatmap_pivot, color_continuous_scale='RdYlGn_r', aspect='auto',
        labels=dict(x="Hour of Day", y="Day of Week", color="kW"),
    )
    fig_heat.update_layout(**base_layout(title="Load Heatmap (Hour x Day)  —  Red = High, Green = Low", height=300))
    fig_heat.update_layout(coloraxis_colorbar=dict(
        title="kW", thickness=12, len=0.8, tickfont=dict(size=9),
    ))
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("Heatmap of average kW demand for every hour-day combination. Red cells = high consumption periods; green = low/off periods. Useful for scheduling energy-intensive operations in low-demand windows.")

    col1, col2 = st.columns(2)
    with col1:
        fig_pf = go.Figure()
        fig_pf.add_trace(go.Histogram(
            x=df_m['power_factor'], nbinsx=60, name='Power Factor',
            marker=dict(color=ACCENT_SECONDARY, line=dict(color='rgba(139,92,246,0.8)', width=0.5)),
        ))
        fig_pf.add_vline(x=0.95, line_dash="dash", line_color=ACCENT_SUCCESS,
                         annotation_text="Target 0.95", annotation_position="top right")
        fig_pf.update_layout(**base_layout(title="Power Factor Distribution", height=350,
                                           xaxis_title="Power Factor", yaxis_title="Frequency"))
        st.plotly_chart(fig_pf, use_container_width=True)
        st.caption("Distribution of 15-min power factor readings. Values to the left of 0.95 (green line) attract reactive energy penalty charges under APERC tariff norms.")

    with col2:
        sample = df_m.sample(min(2000, len(df_m)), random_state=42)
        fig_tri = px.scatter(
            sample, x='active_power_kw', y='reactive_power_kvar',
            color='power_factor', color_continuous_scale='RdYlGn', opacity=0.5,
        )
        fig_tri.update_layout(**base_layout(title="Power Triangle", height=350,
                                             xaxis_title="Active Power (kW)", yaxis_title="Reactive Power (kVAR)"))
        fig_tri.update_layout(coloraxis_colorbar=dict(title="PF", thickness=12, len=0.8, tickfont=dict(size=9)))
        st.plotly_chart(fig_tri, use_container_width=True)
        st.caption("Each dot is one 15-min interval. The angle from the x-axis represents the power factor (colour-coded). Dots near the x-axis (green) indicate high PF; vertical scatter (red) signals heavy reactive load.")


# ============================================================================
# TAB 3: ML PREDICTIONS
# ============================================================================
with tab3:
    st.markdown("""
    <div class="explain-box">
    <b>ML Predictions</b> — Six regression models were trained on <b>36 engineered features</b> (cyclical time
    encodings, 15-min lag/rolling statistics, power quality metrics, and solar contribution ratios)
    using a <b>75%/25% chronological train/test split</b> to prevent look-ahead bias.
    <br><br>
    <b>How to read the metrics:</b>
    &nbsp; <b>R²</b> (R-Squared): closer to 1.0 is better — proportion of variance in energy consumption explained by the model.
    &nbsp; <b>RMSE</b> (Root Mean Squared Error): lower is better — average prediction error in kWh.
    &nbsp; <b>MAE</b> (Mean Absolute Error): lower is better — median-robust error in kWh.
    </div>
    """, unsafe_allow_html=True)

    if ml_data['comparison'] is not None:
        df_comp = ml_data['comparison']

        # Header cards
        st.markdown('<p class="section-label">MODEL RANKINGS</p>', unsafe_allow_html=True)
        cols = st.columns(min(len(df_comp), 5))
        rank_labels = ['#1', '#2', '#3', '#4', '#5', '#6', '#7']
        for i, (_, row) in enumerate(df_comp.iterrows()):
            if i < len(cols):
                with cols[i]:
                    st.metric(
                        f"{rank_labels[i]} {row['model']}",
                        f"R² = {row['R2_Score']:.4f}",
                        help=f"RMSE: {row['RMSE']:.4f} | MAE: {row.get('MAE', float('nan')):.4f}"
                    )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            fig_r2 = go.Figure()
            sorted_comp = df_comp.sort_values('R2_Score')
            fig_r2.add_trace(go.Bar(
                x=sorted_comp['R2_Score'], y=sorted_comp['model'], orientation='h',
                marker=dict(
                    color=sorted_comp['R2_Score'],
                    colorscale=[[0, '#1e3a5f'], [1, '#3b82f6']],
                    cornerradius=4
                ),
                text=[f"{v:.4f}" for v in sorted_comp['R2_Score']],
                textposition='outside', textfont=dict(size=10, color='#94a3b8'),
            ))
            fig_r2.update_layout(**base_layout(title="R-Squared Score", height=380, xaxis_title="R\u00b2"))
            fig_r2.update_layout(yaxis=dict(title=""))
            st.plotly_chart(fig_r2, use_container_width=True)

        with col2:
            fig_rmse = go.Figure()
            sorted_rmse = df_comp.sort_values('RMSE', ascending=False)
            fig_rmse.add_trace(go.Bar(
                x=sorted_rmse['RMSE'], y=sorted_rmse['model'], orientation='h',
                marker=dict(
                    color=sorted_rmse['RMSE'],
                    colorscale=[[0, '#10b981'], [1, '#ef4444']],
                    cornerradius=4
                ),
                text=[f"{v:.4f}" for v in sorted_rmse['RMSE']],
                textposition='outside', textfont=dict(size=10, color='#94a3b8'),
            ))
            fig_rmse.update_layout(**base_layout(title="Root Mean Squared Error", height=380, xaxis_title="RMSE"))
            fig_rmse.update_layout(yaxis=dict(title=""))
            st.plotly_chart(fig_rmse, use_container_width=True)

        # Radar chart
        metrics_list = ['R2_Score', 'MAE', 'RMSE']
        avail_m = [m for m in metrics_list if m in df_comp.columns]
        if len(avail_m) >= 2:
            metric_labels = {'R2_Score': 'R\u00b2 Score', 'MAE': 'MAE (inv.)', 'RMSE': 'RMSE (inv.)'}
            fig_radar = go.Figure()
            for idx, (_, row) in enumerate(df_comp.head(5).iterrows()):
                vals = []
                for m in avail_m:
                    if m == 'R2_Score':
                        vals.append(max(0, row[m]))
                    else:
                        mx = df_comp[m].max()
                        vals.append(max(0, 1 - row[m] / mx) if mx > 0 else 0)
                vals.append(vals[0])
                theta = [metric_labels.get(m, m) for m in avail_m] + [metric_labels.get(avail_m[0], avail_m[0])]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=theta, fill='toself',
                    name=row['model'], opacity=0.5,
                    line=dict(color=CHART_COLORS[idx % len(CHART_COLORS)]),
                ))
            fig_radar.update_layout(
                **base_layout(title="Multi-Metric Comparison", height=420),
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(51,65,85,0.3)',
                                    tickfont=dict(size=9)),
                    angularaxis=dict(gridcolor='rgba(51,65,85,0.3)')
                ),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Full table
        st.markdown('<p class="section-label">DETAILED COMPARISON</p>', unsafe_allow_html=True)
        st.dataframe(
            df_comp.style.format({
                'R2_Score': '{:.4f}', 'RMSE': '{:.4f}', 'MAE': '{:.4f}',
                'MAPE_%': '{:.2f}', 'time_sec': '{:.2f}'
            }).highlight_max(subset=['R2_Score'], color='rgba(16,185,129,0.2)')
             .highlight_min(subset=['RMSE', 'MAE'], color='rgba(16,185,129,0.2)'),
            use_container_width=True, hide_index=True
        )

    else:
        st.warning("Model results not found. Run `python ml_models.py` first.")

    # Actual vs Predicted
    if ml_data['predictions'] is not None:
        st.markdown("---")
        st.markdown('<p class="section-label">ACTUAL VS PREDICTED</p>', unsafe_allow_html=True)
        pred_df = ml_data['predictions']
        model_cols = [c for c in pred_df.columns if c != 'actual']
        if model_cols:
            selected_model = st.selectbox("Model", model_cols, label_visibility="collapsed")
            pred_clean = pred_df[['actual', selected_model]].dropna()

            fig_avp = go.Figure()
            fig_avp.add_trace(go.Scatter(
                x=pred_clean.index, y=pred_clean['actual'],
                mode='lines', name='Actual',
                line=dict(color=ACCENT_PRIMARY, width=1.5),
            ))
            fig_avp.add_trace(go.Scatter(
                x=pred_clean.index, y=pred_clean[selected_model],
                mode='lines', name=f'Predicted',
                line=dict(color=ACCENT_DANGER, width=1.2, dash='dot'),
            ))
            fig_avp.update_layout(**base_layout(
                title=f"Prediction Overlay - {selected_model}", height=380,
                xaxis_title="Sample Index", yaxis_title="Energy (kWh)"
            ))
            st.plotly_chart(fig_avp, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                residuals = pred_clean['actual'] - pred_clean[selected_model]
                fig_res = go.Figure()
                fig_res.add_trace(go.Histogram(
                    x=residuals, nbinsx=50, name='Residuals',
                    marker=dict(color=ACCENT_SECONDARY, line=dict(width=0.5, color='rgba(139,92,246,0.8)'))
                ))
                fig_res.add_vline(x=0, line_dash="dash", line_color=ACCENT_SUCCESS)
                fig_res.update_layout(**base_layout(title="Residual Distribution", height=320,
                                                     xaxis_title="Residual", yaxis_title="Count"))
                st.plotly_chart(fig_res, use_container_width=True)

            with col2:
                fig_sc = px.scatter(
                    pred_clean, x='actual', y=selected_model, opacity=0.4,
                    color_discrete_sequence=[ACCENT_PRIMARY],
                )
                mn = min(pred_clean['actual'].min(), pred_clean[selected_model].min())
                mx = max(pred_clean['actual'].max(), pred_clean[selected_model].max())
                fig_sc.add_trace(go.Scatter(
                    x=[mn, mx], y=[mn, mx], mode='lines', name='Ideal',
                    line=dict(color=ACCENT_SUCCESS, dash='dash', width=2),
                ))
                fig_sc.update_layout(**base_layout(title="Predicted vs Actual", height=320,
                                                     xaxis_title="Actual", yaxis_title="Predicted"))
                st.plotly_chart(fig_sc, use_container_width=True)

    # Feature Importance
    if ml_data['feature_importances'] is not None:
        st.markdown("---")
        st.markdown('<p class="section-label">FEATURE IMPORTANCE</p>', unsafe_allow_html=True)
        fi_df = ml_data['feature_importances']
        fi_cols = fi_df.columns.tolist()
        if fi_cols:
            fi_model = st.selectbox("Model for feature importance", fi_cols, label_visibility="collapsed")
            fi_sorted = fi_df[fi_model].sort_values(ascending=True).tail(15)

            fig_fi = go.Figure()
            fig_fi.add_trace(go.Bar(
                x=fi_sorted.values, y=fi_sorted.index, orientation='h',
                marker=dict(
                    color=fi_sorted.values,
                    colorscale=[[0, '#1e3a5f'], [0.5, '#3b82f6'], [1, '#93c5fd']],
                    cornerradius=4,
                ),
                text=[f"{v:.3f}" for v in fi_sorted.values],
                textposition='outside', textfont=dict(size=9, color='#94a3b8'),
            ))
            fig_fi.update_layout(**base_layout(title=f"Top Features - {fi_model}", height=480,
                                                xaxis_title="Importance"))
            fig_fi.update_layout(yaxis=dict(title="", tickfont=dict(size=10)))
            st.plotly_chart(fig_fi, use_container_width=True)


# ============================================================================
# TAB 4: SOLAR ANALYTICS
# ============================================================================
with tab4:
    st.markdown("""
    <div class="explain-box">
    <b>Solar Analytics</b> — Measurements from the <b>EED Research Wing Solar</b> inverter meter at 15-min intervals.
    Generation is counted only when the inverter output is <b>positive</b> (actively producing). Negative readings
    (night-time inverter standby draw) are excluded to avoid inflating generation totals.
    </div>
    """, unsafe_allow_html=True)
    df_m = data['merged'].copy()

    if 'solar_active_power_kw' in df_m.columns:
        solar_data = df_m.dropna(subset=['solar_active_power_kw']).copy()

        # Correct aggregation: generation only when output > 0
        solar_gen_mask = solar_data['solar_active_power_kw'] > 0
        solar_total    = solar_data.loc[solar_gen_mask, 'solar_active_power_kw'].sum() * 0.25  # kWh
        solar_peak     = solar_data.loc[solar_gen_mask, 'solar_active_power_kw'].max() if solar_gen_mask.any() else 0
        active_solar   = solar_data[solar_gen_mask]
        solar_avg      = active_solar['solar_active_power_kw'].mean() if len(active_solar) > 0 else 0
        night_draw_kwh = abs(solar_data.loc[solar_data['solar_active_power_kw'] < 0, 'solar_active_power_kw'].sum()) * 0.25

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Solar Generation",    f"{solar_total:,.1f} kWh",
                  help="Total generation = sum of positive solar readings × 0.25 h (15-min intervals). Night readings excluded.")
        c2.metric("Peak Solar Output",   f"{solar_peak:.2f} kW",
                  help="Highest recorded positive solar active power output in the dataset.")
        c3.metric("Avg Daylight Output", f"{solar_avg:.2f} kW",
                  help="Mean solar output calculated only from intervals where inverter was actively generating (>0 kW).")
        c4.metric("Night Draw",          f"{night_draw_kwh:,.2f} kWh",
                  help="Cumulative inverter standby consumption when output was negative (night hours). Tracked for net energy balance.")

        st.markdown("---")

        # Solar-only time series (gaps at night)
        solar_gen_series = df_m['solar_active_power_kw'].copy()
        solar_gen_series[solar_gen_series <= 0] = None
        fig_solar = go.Figure()
        fig_solar.add_trace(go.Scatter(
            x=df_m['timestamp'], y=solar_gen_series,
            mode='lines', name='Solar Generation',
            line=dict(color=ACCENT_SOLAR, width=1.5),
            fill='tozeroy', fillcolor='rgba(234, 179, 8, 0.12)',
        ))
        fig_solar.update_layout(**base_layout(
            title="Solar Generation Over Time (kW) — Gaps = Night / No Generation",
            height=420, yaxis_title="kW"))
        fig_solar.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), type="date")
        st.plotly_chart(fig_solar, use_container_width=True)
        st.caption("Solar active power output (kW) from the EED Research Wing inverter. Night hours appear as gaps because negative values are excluded. Use the range slider to zoom into individual days.")

        col1, col2 = st.columns(2)
        with col1:
            hourly_solar = (
                solar_data[solar_gen_mask]
                .groupby('hour')['solar_active_power_kw']
                .mean().reset_index()
            )
            fig_sh = go.Figure()
            fig_sh.add_trace(go.Bar(
                x=hourly_solar['hour'], y=hourly_solar['solar_active_power_kw'],
                marker=dict(
                    color=hourly_solar['solar_active_power_kw'],
                    colorscale=[[0, '#78350f'], [0.5, '#eab308'], [1, '#fef08a']],
                    cornerradius=4,
                ),
            ))
            fig_sh.update_layout(**base_layout(
                title="Avg Solar Output by Hour (Generation Hours Only)",
                height=350, xaxis_title="Hour", yaxis_title="Avg kW"))
            st.plotly_chart(fig_sh, use_container_width=True)
            st.caption("Mean solar output per hour computed only from intervals where the inverter was actively generating. Peak output typically between 10:00—14:00.")

        with col2:
            _dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            dow_solar = (
                solar_data[solar_gen_mask]
                .groupby('day_of_week')['solar_active_power_kw']
                .agg(['mean', 'max']).reset_index()
            )
            dow_solar['day_name'] = [_dow_names[i] for i in dow_solar['day_of_week']]
            fig_sdow = go.Figure()
            fig_sdow.add_trace(go.Bar(
                x=dow_solar['day_name'], y=dow_solar['mean'], name='Avg Generation',
                marker=dict(color=ACCENT_SOLAR, cornerradius=4),
            ))
            fig_sdow.add_trace(go.Scatter(
                x=dow_solar['day_name'], y=dow_solar['max'], name='Peak',
                mode='markers+lines', marker=dict(size=5, color=ACCENT_WARNING),
                line=dict(color=ACCENT_WARNING, dash='dot', width=1.5),
            ))
            fig_sdow.update_layout(**base_layout(
                title="Solar Output by Day of Week", height=350, yaxis_title="Avg kW"))
            st.plotly_chart(fig_sdow, use_container_width=True)
            st.caption("Average (bar) and peak (line) solar generation per day of week. Unlike load, variability here reflects weather/cloud cover rather than day-type patterns.")
    else:
        st.info("Solar data not available in the merged dataset.")


# ============================================================================
# TAB 5: ANOMALY DETECTION
# ============================================================================
with tab5:
    st.markdown("""
    <div class="explain-box">
    <b>Detection Algorithm: Isolation Forest</b><br>
    Isolation Forest is an unsupervised ML algorithm that detects anomalies by <b>random feature partitioning</b>.
    Normal points require more splits to isolate; anomalies — being rare and extreme — are isolated faster,
    yielding a lower <b>anomaly score</b> (more negative = more anomalous).<br><br>
    <b>Features used for detection:</b>
    Active Power (kW), Average Current (A), Line-to-Line Voltage (V), Power Factor, Reactive Power (kVAR).<br><br>
    <b>Contamination parameter:</b> 5% — the model expects approximately 1 in 20 readings to be anomalous.<br><br>
    <b>Severity thresholds (anomaly score):</b><br>
    &nbsp; <span style="color:#ef4444;">&#9679; Critical</span>: score &lt; −0.30 &mdash; strongly isolated; extreme deviation in multiple features.<br>
    &nbsp; <span style="color:#f59e0b;">&#9679; Warning</span>: −0.30 &le; score &lt; −0.15 &mdash; notable but moderate deviation.<br>
    &nbsp; <span style="color:#64748b;">&#9679; Minor</span>: −0.15 &le; score &lt; 0 &mdash; marginally outside the normal envelope.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-label">ISOLATION FOREST ANOMALY DETECTION</p>', unsafe_allow_html=True)

    # Run anomaly detection inline
    df_m = data['merged'].copy()
    with st.spinner("Analyzing patterns..."):
        df_clean, anomalies = run_anomaly_detection(df_m)

    n_total = len(df_clean)
    n_anomalies = len(anomalies)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",     f"{n_total:,}",
              help="Total 15-minute interval records passed through the Isolation Forest.")
    c2.metric("Anomalies Detected",f"{n_anomalies:,}",
              help="Points flagged as anomalous (approximately 5% of total per contamination setting).")
    c3.metric("Anomaly Rate",      f"{n_anomalies/n_total*100:.1f}%",
              help="Fraction of total records classified as anomalous.")
    if 'severity' in anomalies.columns:
        critical = len(anomalies[anomalies['severity'] == 'Critical'])
        c4.metric("Critical Events",   f"{critical}",
                  help="Readings with anomaly score < -0.30, indicating extreme multi-feature deviation.")
    else:
        c4.metric("Detection Method", "Isolation Forest")

    st.markdown("---")

    # Timeline overlay
    if 'timestamp' in anomalies.columns and 'active_power_kw' in anomalies.columns:
        fig_anom = go.Figure()
        fig_anom.add_trace(go.Scatter(
            x=df_clean['timestamp'], y=df_clean['active_power_kw'],
            mode='lines', name='Normal Operation',
            line=dict(color=ACCENT_PRIMARY, width=0.8), opacity=0.6,
        ))
        fig_anom.add_trace(go.Scatter(
            x=anomalies['timestamp'], y=anomalies['active_power_kw'],
            mode='markers', name='Anomaly',
            marker=dict(color=ACCENT_DANGER, size=5, symbol='diamond',
                        line=dict(width=0.5, color='rgba(239,68,68,0.5)')),
        ))
        fig_anom.update_layout(**base_layout(title="Anomaly Detection Timeline", height=420,
                                              yaxis_title="Active Power (kW)"))
        fig_anom.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), type="date")
        st.plotly_chart(fig_anom, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        if 'anomaly_score' in anomalies.columns:
            fig_score = go.Figure()
            fig_score.add_trace(go.Histogram(
                x=anomalies['anomaly_score'], nbinsx=40,
                marker=dict(color=ACCENT_DANGER, line=dict(width=0.5, color='rgba(239,68,68,0.8)')),
            ))
            fig_score.update_layout(**base_layout(title="Anomaly Score Distribution", height=340,
                                                   xaxis_title="Score (lower = more anomalous)", yaxis_title="Count"))
            st.plotly_chart(fig_score, use_container_width=True)

    with col2:
        if 'severity' in anomalies.columns:
            sev_counts = anomalies['severity'].value_counts().reset_index()
            sev_counts.columns = ['severity', 'count']
            sev_colors = {'Critical': ACCENT_DANGER, 'Warning': ACCENT_WARNING, 'Minor': '#64748b'}
            fig_sev = go.Figure()
            fig_sev.add_trace(go.Bar(
                x=sev_counts['severity'], y=sev_counts['count'],
                marker=dict(color=[sev_colors.get(s, ACCENT_PRIMARY) for s in sev_counts['severity']],
                            cornerradius=4),
                text=sev_counts['count'], textposition='outside',
                textfont=dict(size=11, color='#94a3b8'),
            ))
            fig_sev.update_layout(**base_layout(title="Severity Breakdown", height=340,
                                                 xaxis_title="", yaxis_title="Count"))
            st.plotly_chart(fig_sev, use_container_width=True)

    # Anomaly details table
    st.markdown('<p class="section-label">ANOMALY DETAILS</p>', unsafe_allow_html=True)
    display_cols = ['timestamp', 'active_power_kw', 'current_avg', 'voltage_ll_avg',
                    'power_factor', 'anomaly_score', 'severity']
    avail_cols = [c for c in display_cols if c in anomalies.columns]
    st.dataframe(anomalies[avail_cols].head(100), use_container_width=True, hide_index=True)


# ============================================================================
# TAB 6: FORECASTING
# ============================================================================
with tab6:
    st.markdown("""
    <div class="explain-box">
    <b>Forecasting Methodology</b> \u2014 All forecasts on this page are <b>statistical / pattern-based</b>,
    derived from historical 15-minute interval data. For each signal, the historical mean and standard deviation
    are computed per hour of day. The shaded bands represent <b>68% (\u00b11\u03c3)</b> and <b>95% (\u00b12\u03c3)</b> confidence intervals
    assuming normally distributed readings. No future data is used; this is purely an extrapolation of observed patterns.
    <br><br>
    <b>Gap handling:</b> Raw timestamps are preserved so that gaps in measurement appear as true discontinuities
    in the trend charts \u2014 rather than the line "jumping" across missing periods.
    </div>
    """, unsafe_allow_html=True)

    # Use full merged df (no dropna) - gaps preserved in timeline
    df_full = data['merged'].copy()
    df_m    = df_full.dropna(subset=['energy_consumption'])  # only for aggregations

    if len(df_m) > 100:

        # ---- KPI row ----
        st.markdown('<p class="section-label">ENERGY CONSUMPTION FORECAST</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        hourly_stats = df_m.groupby('hour')['energy_consumption'].agg(['mean', 'std', 'median']).reset_index()
        hourly_stats.columns = ['hour', 'expected', 'uncertainty', 'median']
        daily_expected = hourly_stats['expected'].sum() * 4
        peak_hour = int(hourly_stats.loc[hourly_stats['expected'].idxmax(), 'hour'])
        min_hour  = int(hourly_stats.loc[hourly_stats['expected'].idxmin(), 'hour'])
        c1.metric("Expected Daily Consumption", f"{daily_expected:.1f} kWh",
                  help="Sum of hourly mean consumption \u00d7 4 (15-min intervals per hour).")
        c2.metric("Predicted Peak Hour",     f"{peak_hour:02d}:00",
                  help="Hour of day with the highest average energy consumption.")
        c3.metric("Predicted Off-Peak Hour", f"{min_hour:02d}:00",
                  help="Hour of day with the lowest average energy consumption.")

        st.markdown("---")

        # ---- Recent 7-day trend (with correct gap handling) ----
        recent = df_full.tail(96 * 7).copy()  # full df - NaN values become gaps in Plotly
        recent['ma_24h'] = recent['energy_consumption'].rolling(96, min_periods=1).mean()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=recent['timestamp'], y=recent['energy_consumption'],
            mode='lines', name='Actual Consumption',
            line=dict(color=ACCENT_PRIMARY, width=1),
            fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.06)',
        ))
        fig_trend.add_trace(go.Scatter(
            x=recent['timestamp'], y=recent['ma_24h'],
            mode='lines', name='24h Moving Average',
            line=dict(color=ACCENT_WARNING, width=2.5),
        ))
        fig_trend.update_layout(**base_layout(title="Recent 7-Day Consumption Trend", height=400,
                                               yaxis_title="Energy (kWh per 15-min)"))
        fig_trend.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), type="date")
        st.plotly_chart(fig_trend, use_container_width=True)
        st.caption("Last 7 days of energy consumption (kWh per 15-min interval) with a 24-hour rolling average overlay. NaN intervals appear as genuine gaps, preserving temporal accuracy.")

        # ---- Energy forecast + Weekday vs Weekend ----
        col1, col2 = st.columns(2)
        with col1:
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=list(hourly_stats['hour']) + list(hourly_stats['hour'][::-1]),
                y=list(hourly_stats['expected'] + 2 * hourly_stats['uncertainty']) +
                  list((hourly_stats['expected'] - 2 * hourly_stats['uncertainty']).clip(lower=0)[::-1]),
                fill='toself', fillcolor='rgba(59,130,246,0.08)',
                line=dict(width=0), name='95% Confidence', showlegend=True,
            ))
            fig_fc.add_trace(go.Scatter(
                x=list(hourly_stats['hour']) + list(hourly_stats['hour'][::-1]),
                y=list(hourly_stats['expected'] + hourly_stats['uncertainty']) +
                  list((hourly_stats['expected'] - hourly_stats['uncertainty']).clip(lower=0)[::-1]),
                fill='toself', fillcolor='rgba(59,130,246,0.15)',
                line=dict(width=0), name='68% Confidence', showlegend=True,
            ))
            fig_fc.add_trace(go.Scatter(
                x=hourly_stats['hour'], y=hourly_stats['expected'],
                mode='lines+markers', name='Expected',
                line=dict(color=ACCENT_PRIMARY, width=3),
                marker=dict(size=6, color=ACCENT_PRIMARY),
            ))
            fig_fc.update_layout(**base_layout(title="24-Hour Energy Forecast (Pattern-Based)", height=400,
                                                xaxis_title="Hour of Day", yaxis_title="Energy (kWh)"))
            st.plotly_chart(fig_fc, use_container_width=True)
            st.caption("Expected energy consumption by hour \u00b1 1\u03c3 / 2\u03c3 bands, derived from all historical readings. Shaded bands show forecast uncertainty; wider bands indicate more variable hours.")

        with col2:
            wkday = df_m[df_m['is_weekend'] == 0].groupby('hour')['energy_consumption'].mean().reset_index()
            wkend = df_m[df_m['is_weekend'] == 1].groupby('hour')['energy_consumption'].mean().reset_index()
            fig_ww = go.Figure()
            fig_ww.add_trace(go.Scatter(
                x=wkday['hour'], y=wkday['energy_consumption'],
                mode='lines+markers', name='Weekday',
                line=dict(color=ACCENT_PRIMARY, width=2.5), marker=dict(size=4),
            ))
            fig_ww.add_trace(go.Scatter(
                x=wkend['hour'], y=wkend['energy_consumption'],
                mode='lines+markers', name='Weekend',
                line=dict(color=ACCENT_SECONDARY, width=2.5), marker=dict(size=4),
            ))
            fig_ww.update_layout(**base_layout(title="Weekday vs Weekend Energy Profile", height=400,
                                                xaxis_title="Hour of Day", yaxis_title="Energy (kWh)"))
            st.plotly_chart(fig_ww, use_container_width=True)
            st.caption("Comparison of weekday vs weekend hourly patterns. The delta between the two curves highlights the impact of academic/lab activity on building energy demand.")

        st.markdown("---")

        # ---- Active Power Forecast ----
        st.markdown('<p class="section-label">ACTIVE POWER DEMAND FORECAST</p>', unsafe_allow_html=True)
        st.caption("Methodology: historical mean active power (kW) per hour of day, with \u00b11\u03c3 / \u00b12\u03c3 confidence bands. Useful for capacity planning and identifying consistent peak-demand windows.")
        pw_stats = df_m.groupby('hour')['active_power_kw'].agg(['mean', 'std']).reset_index()
        pw_stats.columns = ['hour', 'mean', 'std']
        fig_pw = go.Figure()
        fig_pw.add_trace(go.Scatter(
            x=list(pw_stats['hour']) + list(pw_stats['hour'][::-1]),
            y=list(pw_stats['mean'] + 2*pw_stats['std']) +
              list((pw_stats['mean'] - 2*pw_stats['std']).clip(lower=0)[::-1]),
            fill='toself', fillcolor='rgba(59,130,246,0.08)',
            line=dict(width=0), name='95% CI',
        ))
        fig_pw.add_trace(go.Scatter(
            x=list(pw_stats['hour']) + list(pw_stats['hour'][::-1]),
            y=list(pw_stats['mean'] + pw_stats['std']) +
              list((pw_stats['mean'] - pw_stats['std']).clip(lower=0)[::-1]),
            fill='toself', fillcolor='rgba(59,130,246,0.15)',
            line=dict(width=0), name='68% CI',
        ))
        fig_pw.add_trace(go.Scatter(
            x=pw_stats['hour'], y=pw_stats['mean'],
            mode='lines+markers', name='Expected',
            line=dict(color=ACCENT_PRIMARY, width=3),
            marker=dict(size=6, color=ACCENT_PRIMARY),
        ))
        fig_pw.update_layout(**base_layout(title="24-Hour Active Power Demand Forecast (kW)", height=380,
                                            xaxis_title="Hour of Day", yaxis_title="Active Power (kW)"))
        st.plotly_chart(fig_pw, use_container_width=True)

        st.markdown("---")

        # ---- Voltage Forecast ----
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-label">VOLTAGE PROFILE FORECAST</p>', unsafe_allow_html=True)
            st.caption("Historical mean voltage per hour \u00b1 1\u03c3/2\u03c3. Hours where the lower CI approaches 415\u00d70.94 = 390 V flag potential voltage-dip risk periods.")
            vt_stats = df_m.groupby('hour')['voltage_ll_avg'].agg(['mean', 'std']).reset_index()
            vt_stats.columns = ['hour', 'mean', 'std']
            fig_vt = go.Figure()
            fig_vt.add_trace(go.Scatter(
                x=list(vt_stats['hour']) + list(vt_stats['hour'][::-1]),
                y=list(vt_stats['mean'] + 2*vt_stats['std']) +
                  list((vt_stats['mean'] - 2*vt_stats['std'])[::-1]),
                fill='toself', fillcolor='rgba(239,68,68,0.07)',
                line=dict(width=0), name='95% CI',
            ))
            fig_vt.add_trace(go.Scatter(
                x=list(vt_stats['hour']) + list(vt_stats['hour'][::-1]),
                y=list(vt_stats['mean'] + vt_stats['std']) +
                  list((vt_stats['mean'] - vt_stats['std'])[::-1]),
                fill='toself', fillcolor='rgba(239,68,68,0.14)',
                line=dict(width=0), name='68% CI',
            ))
            fig_vt.add_trace(go.Scatter(
                x=vt_stats['hour'], y=vt_stats['mean'],
                mode='lines+markers', name='Expected',
                line=dict(color=ACCENT_DANGER, width=2.5),
                marker=dict(size=5, color=ACCENT_DANGER),
            ))
            fig_vt.add_hline(y=415*0.94, line_dash="dot", line_color="rgba(245,158,11,0.6)",
                              annotation_text="-6% limit (390 V)", annotation_position="bottom right")
            fig_vt.update_layout(**base_layout(title="Hourly Voltage Forecast (V)", height=380,
                                                xaxis_title="Hour of Day", yaxis_title="Voltage (V)"))
            st.plotly_chart(fig_vt, use_container_width=True)

        with col2:
            st.markdown('<p class="section-label">SOLAR GENERATION FORECAST</p>', unsafe_allow_html=True)
            st.caption("Derived from positive solar readings only (generation hours). The forecast shows expected solar yield per hour. Zero output outside 6AM\u201418:00 window confirms daylight-only generation.")
            if 'solar_active_power_kw' in df_m.columns:
                sol_gen = df_m[df_m['solar_active_power_kw'] > 0]
                sol_stats = sol_gen.groupby('hour')['solar_active_power_kw'].agg(['mean', 'std']).reset_index()
                sol_stats.columns = ['hour', 'mean', 'std']
                sol_stats['mean'].fillna(0, inplace=True)
                sol_stats['std'].fillna(0, inplace=True)
                fig_sol = go.Figure()
                fig_sol.add_trace(go.Scatter(
                    x=list(sol_stats['hour']) + list(sol_stats['hour'][::-1]),
                    y=list(sol_stats['mean'] + 2*sol_stats['std']) +
                      list((sol_stats['mean'] - 2*sol_stats['std']).clip(lower=0)[::-1]),
                    fill='toself', fillcolor='rgba(234,179,8,0.08)',
                    line=dict(width=0), name='95% CI',
                ))
                fig_sol.add_trace(go.Scatter(
                    x=list(sol_stats['hour']) + list(sol_stats['hour'][::-1]),
                    y=list(sol_stats['mean'] + sol_stats['std']) +
                      list((sol_stats['mean'] - sol_stats['std']).clip(lower=0)[::-1]),
                    fill='toself', fillcolor='rgba(234,179,8,0.15)',
                    line=dict(width=0), name='68% CI',
                ))
                fig_sol.add_trace(go.Scatter(
                    x=sol_stats['hour'], y=sol_stats['mean'],
                    mode='lines+markers', name='Expected',
                    line=dict(color=ACCENT_SOLAR, width=2.5),
                    marker=dict(size=5, color=ACCENT_SOLAR),
                ))
                fig_sol.update_layout(**base_layout(title="Hourly Solar Generation Forecast (kW)", height=380,
                                                     xaxis_title="Hour of Day", yaxis_title="Solar Power (kW)"))
                st.plotly_chart(fig_sol, use_container_width=True)
            else:
                st.info("Solar data not available.")

        st.markdown("---")

        # ---- Hourly Load Forecast Table ----
        st.markdown('<p class="section-label">HOURLY LOAD FORECAST TABLE</p>', unsafe_allow_html=True)
        forecast_table = hourly_stats.copy()
        forecast_table['lower_bound'] = (forecast_table['expected'] - forecast_table['uncertainty']).clip(lower=0)
        forecast_table['upper_bound'] = forecast_table['expected'] + forecast_table['uncertainty']
        forecast_table = forecast_table.rename(columns={
            'hour': 'Hour', 'expected': 'Expected (kWh)', 'uncertainty': 'Std Dev',
            'median': 'Median (kWh)', 'lower_bound': 'Lower Bound', 'upper_bound': 'Upper Bound'
        })
        st.dataframe(
            forecast_table.style.format({
                'Expected (kWh)': '{:.4f}', 'Std Dev': '{:.4f}', 'Median (kWh)': '{:.4f}',
                'Lower Bound': '{:.4f}', 'Upper Bound': '{:.4f}'
            }),
            use_container_width=True, hide_index=True
        )
        st.caption("Tabular summary of the pattern-based 24-hour energy consumption forecast. Lower/Upper bounds represent \u00b11\u03c3 from the mean.")
    else:
        st.info("Insufficient data for forecasting. Need at least 100 data points.")



# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 12px; color: #475569; font-size: 0.7rem; letter-spacing: 0.05em;">
    EED SMARTGRID ANALYTICS v1.0 &middot; National Institute of Technology, Warangal &middot;
    Streamlit &middot; Plotly &middot; scikit-learn &middot; XGBoost &middot; TensorFlow
</div>
""", unsafe_allow_html=True)
