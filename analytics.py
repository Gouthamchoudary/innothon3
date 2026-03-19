"""
EED SmartGrid Analytics — Advanced Analytics
==============================================
Anomaly detection, peak load analysis, solar optimization,
and feeder-level insights.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def detect_anomalies(df_merged, contamination=0.05):
    """Detect anomalies in energy consumption using Isolation Forest."""
    print("\n🚨 Running Anomaly Detection (Isolation Forest)...")

    features_for_anomaly = [
        'active_power_kw', 'current_avg', 'voltage_ll_avg',
        'power_factor', 'reactive_power_kvar'
    ]
    available = [f for f in features_for_anomaly if f in df_merged.columns]

    df_clean = df_merged.dropna(subset=available).copy()
    X_anomaly = df_clean[available].values

    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
        n_jobs=-1
    )
    df_clean['anomaly_label'] = iso_forest.fit_predict(X_anomaly)
    df_clean['anomaly_score'] = iso_forest.decision_function(X_anomaly)

    # -1 = anomaly, 1 = normal
    anomalies = df_clean[df_clean['anomaly_label'] == -1].copy()

    # Assign severity based on score
    anomalies['severity'] = pd.cut(
        anomalies['anomaly_score'],
        bins=[-np.inf, -0.3, -0.15, 0],
        labels=['🔴 Critical', '🟠 Warning', '🟡 Minor']
    )

    print(f"   ✅ Found {len(anomalies)} anomalies out of {len(df_clean)} records ({len(anomalies)/len(df_clean)*100:.1f}%)")
    print(f"   Severity breakdown:")
    if len(anomalies) > 0:
        for sev, count in anomalies['severity'].value_counts().items():
            print(f"      {sev}: {count}")

    # Save anomalies
    if 'timestamp' in anomalies.columns:
        anomalies_out = anomalies[['timestamp'] + available + ['anomaly_score', 'severity']].copy()
    else:
        anomalies_out = anomalies[available + ['anomaly_score', 'severity']].copy()
    anomalies_out.to_csv(os.path.join(BASE_DIR, 'anomalies.csv'), index=False)
    print(f"   💾 Saved anomalies.csv")

    return df_clean, anomalies


def peak_load_analysis(df_merged):
    """Analyze peak consumption patterns and provide recommendations."""
    print("\n📈 Running Peak Load Analysis...")

    df = df_merged.dropna(subset=['active_power_kw']).copy()

    # Hourly average power
    hourly_avg = df.groupby('hour')['active_power_kw'].agg(['mean', 'std', 'max']).round(3)
    hourly_avg.columns = ['avg_power_kw', 'std_power_kw', 'peak_power_kw']

    # Day of week average
    dow_avg = df.groupby('day_of_week')['active_power_kw'].mean().round(3)
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_avg.index = [dow_names[i] for i in dow_avg.index]

    # Weekend vs weekday
    weekday_avg = df[df['is_weekend'] == 0]['active_power_kw'].mean()
    weekend_avg = df[df['is_weekend'] == 1]['active_power_kw'].mean()

    # Peak hours
    peak_hours = hourly_avg.nlargest(5, 'avg_power_kw')
    offpeak_hours = hourly_avg.nsmallest(5, 'avg_power_kw')

    print(f"   ⚡ Peak hours (top 5):")
    for h, row in peak_hours.iterrows():
        print(f"      {h:02d}:00 — Avg: {row['avg_power_kw']:.2f} kW, Peak: {row['peak_power_kw']:.2f} kW")

    print(f"   🌙 Off-peak hours (bottom 5):")
    for h, row in offpeak_hours.iterrows():
        print(f"      {h:02d}:00 — Avg: {row['avg_power_kw']:.2f} kW")

    print(f"\n   📊 Weekday avg: {weekday_avg:.2f} kW vs Weekend avg: {weekend_avg:.2f} kW")
    savings_pct = ((weekday_avg - weekend_avg) / weekday_avg * 100) if weekday_avg > 0 else 0
    print(f"   💡 Weekend is {abs(savings_pct):.1f}% {'lower' if savings_pct > 0 else 'higher'} than weekday")

    return {
        'hourly_avg': hourly_avg,
        'dow_avg': dow_avg,
        'weekday_avg': weekday_avg,
        'weekend_avg': weekend_avg,
        'peak_hours': peak_hours,
        'offpeak_hours': offpeak_hours,
    }


def solar_analysis(df_merged):
    """Analyze solar generation vs grid consumption."""
    print("\n☀️  Running Solar Analytics...")

    df = df_merged.copy()

    # Solar generation stats
    if 'solar_active_power_kw' not in df.columns:
        print("   ⚠️ No solar data available")
        return {}

    solar_cols = ['solar_active_power_kw', 'solar_active_energy_kwh']
    df_solar = df.dropna(subset=['solar_active_power_kw']).copy()

    total_solar = df_solar['solar_active_power_kw'].abs().sum()
    total_grid = df_solar['active_power_kw'].sum()

    # Solar hours (power > 0.5 kW)
    solar_hours = df_solar[df_solar['solar_active_power_kw'].abs() > 0.5]
    avg_solar_gen = solar_hours['solar_active_power_kw'].abs().mean() if len(solar_hours) > 0 else 0
    peak_solar_gen = df_solar['solar_active_power_kw'].abs().max()

    # Solar by hour
    hourly_solar = df_solar.groupby(df_solar['timestamp'].dt.hour)['solar_active_power_kw'].mean().abs()

    # Self-sufficiency ratio (how much of consumption is covered by solar)
    df_solar['self_sufficiency'] = (
        df_solar['solar_active_power_kw'].abs() /
        df_solar['active_power_kw'].replace(0, np.nan)
    ).clip(0, 1)
    avg_self_sufficiency = df_solar['self_sufficiency'].mean()

    print(f"   ⚡ Total solar generation: {total_solar:.1f} kW (cumulative)")
    print(f"   🔌 Total grid consumption: {total_grid:.1f} kW (cumulative)")
    print(f"   ☀️  Average solar during daylight: {avg_solar_gen:.2f} kW")
    print(f"   🏆 Peak solar generation: {peak_solar_gen:.2f} kW")
    print(f"   📊 Average self-sufficiency: {avg_self_sufficiency*100:.1f}%")

    return {
        'total_solar': total_solar,
        'total_grid': total_grid,
        'avg_solar_gen': avg_solar_gen,
        'peak_solar_gen': peak_solar_gen,
        'hourly_solar': hourly_solar,
        'avg_self_sufficiency': avg_self_sufficiency,
    }


def feeder_analysis(df_daily):
    """Analyze consumption patterns by feeder."""
    print("\n🔌 Running Feeder-Level Analysis...")

    if df_daily is None or len(df_daily) == 0:
        print("   ⚠️ No daily data available")
        return {}

    # Total consumption by feeder
    feeder_total = df_daily.groupby('feeder')['daily_kwh'].agg(['sum', 'mean', 'std', 'max']).round(2)
    feeder_total.columns = ['total_kwh', 'avg_daily_kwh', 'std_daily_kwh', 'peak_daily_kwh']
    feeder_total = feeder_total.sort_values('total_kwh', ascending=False)

    print("   📊 Feeder consumption ranking:")
    for feeder, row in feeder_total.iterrows():
        print(f"      {feeder}: Total={row['total_kwh']:.0f} kWh, Avg={row['avg_daily_kwh']:.1f} kWh/day")

    # Consumption vs Generation feeders
    consumption_feeders = ['EED_Incomer_1', 'EED_Incomer_2', 'Research_Wing_Incomer_1',
                          'Research_Wing_Incomer_2', 'EED_Load_Feeder', 'Civil_Load_Feeder']
    generation_feeders = ['Research_Wing_Solar', 'EED_Solar']

    total_consumption = df_daily[df_daily['feeder'].isin(consumption_feeders)]['daily_kwh'].sum()
    total_generation = df_daily[df_daily['feeder'].isin(generation_feeders)]['daily_kwh'].sum()

    print(f"\n   ⚡ Total grid consumption: {total_consumption:.0f} kWh")
    print(f"   ☀️  Total solar generation: {total_generation:.0f} kWh")
    if total_consumption > 0:
        print(f"   📊 Solar meets {total_generation/total_consumption*100:.1f}% of demand")

    return {
        'feeder_total': feeder_total,
        'total_consumption': total_consumption,
        'total_generation': total_generation,
    }


def run_all_analytics(data):
    """Run all analytics modules."""
    print("\n" + "=" * 70)
    print("📊 EED SmartGrid Analytics — Advanced Analytics Suite")
    print("=" * 70)

    df_merged = data.get('merged')
    df_daily = data.get('daily')

    results = {}

    if df_merged is not None and len(df_merged) > 0:
        df_clean, anomalies = detect_anomalies(df_merged)
        results['anomalies'] = anomalies
        results['df_clean'] = df_clean
        results['peak_load'] = peak_load_analysis(df_merged)
        results['solar'] = solar_analysis(df_merged)

    if df_daily is not None and len(df_daily) > 0:
        results['feeder'] = feeder_analysis(df_daily)

    return results


if __name__ == "__main__":
    from data_pipeline import load_all_data
    data = load_all_data()
    analytics = run_all_analytics(data)
