"""
EED SmartGrid Analytics — Data Pipeline
========================================
Loads, cleans, merges, and engineers features from NIT Warangal EED building
electrical data for ML model training.

Data Sources:
  - 32 Daily Report Excel files (feeder-level daily consumption)
  - eedincomer.xlsx (15-min interval incomer data)
  - researchwindsolar.xlsx (15-min interval solar generation data)
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DAILY_DIR = os.path.join(BASE_DIR, "Daily data from dec18_to_jan18")
INCOMER_FILE = os.path.join(BASE_DIR, "eedincomer.xlsx")
SOLAR_FILE = os.path.join(BASE_DIR, "researchwindsolar.xlsx")

# Feeder name mapping
FEEDER_NAMES = {
    1: "EED_Incomer_1",
    2: "EED_Incomer_2",
    3: "Research_Wing_Incomer_1",
    4: "Research_Wing_Incomer_2",
    5: "EED_Load_Feeder",
    6: "Civil_Load_Feeder",
    7: "Research_Wing_Solar",
    8: "EED_Solar",
}


# ============================================================================
# 1. LOAD DAILY REPORTS
# ============================================================================
def load_daily_reports():
    """Load all 32 daily report Excel files and combine into a single DataFrame."""
    print("📂 Loading daily reports...")
    records = []

    files = sorted([f for f in os.listdir(DAILY_DIR) if f.endswith('.xlsx')])
    print(f"   Found {len(files)} daily report files")

    for fname in files:
        # Extract date from filename: Daily Report_DD-MM-YYYY.xlsx
        match = re.search(r'(\d{2})-(\d{2})-(\d{4})', fname)
        if not match:
            continue
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        date = datetime(year, month, day)

        filepath = os.path.join(DAILY_DIR, fname)
        try:
            wb = pd.ExcelFile(filepath, engine='openpyxl')

            # Read Daily_Consumption sheet
            df_cons = pd.read_excel(
                wb, sheet_name='Daily_Consumption',
                header=None, skiprows=2
            )

            for idx, row in df_cons.iterrows():
                meter_no = row.iloc[0]
                if pd.isna(meter_no) or not isinstance(meter_no, (int, float)):
                    continue
                meter_no = int(meter_no)
                if meter_no not in FEEDER_NAMES:
                    continue

                feeder = FEEDER_NAMES[meter_no]
                init_kwh = row.iloc[2] if len(row) > 2 else None
                final_kwh = row.iloc[3] if len(row) > 3 else None
                daily_kwh = row.iloc[4] if len(row) > 4 else None

                records.append({
                    'date': date,
                    'feeder': feeder,
                    'meter_no': meter_no,
                    'init_reading_kwh': init_kwh,
                    'final_reading_kwh': final_kwh,
                    'daily_kwh': daily_kwh,
                })
        except Exception as e:
            print(f"   ⚠️ Error loading {fname}: {e}")

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'meter_no']).reset_index(drop=True)

    # Convert to numeric
    for col in ['init_reading_kwh', 'final_reading_kwh', 'daily_kwh']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"   ✅ Loaded {len(df)} records from {df['date'].nunique()} days")
    return df


# ============================================================================
# 2. LOAD 15-MIN INTERVAL INCOMER DATA
# ============================================================================
def _parse_timestamp(ts_str):
    """Parse timestamp like '16-12-2025@15:00:00.000' to datetime."""
    if pd.isna(ts_str) or not isinstance(ts_str, str):
        return pd.NaT
    try:
        return pd.to_datetime(ts_str, format='%d-%m-%Y@%H:%M:%S.%f')
    except:
        try:
            return pd.to_datetime(ts_str)
        except:
            return pd.NaT


def load_incomer_data():
    """Load EED Incomer 2 data (15-min intervals)."""
    print("⚡ Loading incomer data...")

    wb = pd.ExcelFile(INCOMER_FILE, engine='openpyxl')
    df_raw = pd.read_excel(wb, sheet_name='EED Incomer 2', header=None)

    # Based on exploration: Row 10 (0-indexed: 9) has headers
    # Columns of interest (0-indexed): C=2, D=3, F=5, G=6, J=9, K=10, N=13, O=14, P=15, Q=16, T=19, U=20
    # Data starts at row 11 (0-indexed: 10)

    records = []
    for idx in range(10, len(df_raw)):
        row = df_raw.iloc[idx]

        # Voltage timestamp & value (cols C=2, D=3)
        ts_voltage = _parse_timestamp(row.iloc[2] if len(row) > 2 else None)
        voltage = row.iloc[3] if len(row) > 3 else None

        # Current timestamp & value (cols F=5, G=6)
        ts_current = _parse_timestamp(row.iloc[5] if len(row) > 5 else None)
        current = row.iloc[6] if len(row) > 6 else None

        # Power Factor (cols J=9, K=10)
        ts_pf = _parse_timestamp(row.iloc[9] if len(row) > 9 else None)
        pf = row.iloc[10] if len(row) > 10 else None

        # Power values (cols N=13, O=14, P=15, Q=16)
        ts_power = _parse_timestamp(row.iloc[13] if len(row) > 13 else None)
        active_power = row.iloc[14] if len(row) > 14 else None
        apparent_power = row.iloc[15] if len(row) > 15 else None
        reactive_power = row.iloc[16] if len(row) > 16 else None

        # Energy (cols T=19, U=20)
        ts_energy = _parse_timestamp(row.iloc[19] if len(row) > 19 else None)
        active_energy = row.iloc[20] if len(row) > 20 else None

        # Use the most common timestamp as primary
        timestamps = [ts_voltage, ts_current, ts_pf, ts_power, ts_energy]
        valid_ts = [t for t in timestamps if pd.notna(t)]
        if not valid_ts:
            continue
        primary_ts = valid_ts[0]

        records.append({
            'timestamp': primary_ts,
            'voltage_ll_avg': voltage,
            'current_avg': current,
            'power_factor': pf,
            'active_power_kw': active_power,
            'apparent_power_kva': apparent_power,
            'reactive_power_kvar': reactive_power,
            'active_energy_kwh': active_energy,
        })

    df = pd.DataFrame(records)
    for col in df.columns:
        if col != 'timestamp':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df = df.drop_duplicates(subset=['timestamp'], keep='first')

    print(f"   ✅ Loaded {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


# ============================================================================
# 3. LOAD SOLAR GENERATION DATA
# ============================================================================
def load_solar_data():
    """Load Research Wing Solar data (15-min intervals)."""
    print("☀️  Loading solar generation data...")

    wb = pd.ExcelFile(SOLAR_FILE, engine='openpyxl')
    df_raw = pd.read_excel(wb, sheet_name='EED Research wing solar', header=None)

    # Based on exploration: Row 7 (0-indexed: 6) has headers
    # Columns (0-indexed): B=1, C=2, F=5, G=6, J=9, K=10, N=13, O=14, P=15, Q=16, S=18, T=19
    # Data starts at row 8 (0-indexed: 7)

    records = []
    for idx in range(7, len(df_raw)):
        row = df_raw.iloc[idx]

        ts_voltage = _parse_timestamp(row.iloc[1] if len(row) > 1 else None)
        voltage = row.iloc[2] if len(row) > 2 else None

        ts_current = _parse_timestamp(row.iloc[5] if len(row) > 5 else None)
        current = row.iloc[6] if len(row) > 6 else None

        ts_pf = _parse_timestamp(row.iloc[9] if len(row) > 9 else None)
        pf = row.iloc[10] if len(row) > 10 else None

        ts_power = _parse_timestamp(row.iloc[13] if len(row) > 13 else None)
        active_power = row.iloc[14] if len(row) > 14 else None
        apparent_power = row.iloc[15] if len(row) > 15 else None
        reactive_power = row.iloc[16] if len(row) > 16 else None

        ts_energy = _parse_timestamp(row.iloc[18] if len(row) > 18 else None)
        active_energy = row.iloc[19] if len(row) > 19 else None

        timestamps = [ts_voltage, ts_current, ts_pf, ts_power, ts_energy]
        valid_ts = [t for t in timestamps if pd.notna(t)]
        if not valid_ts:
            continue
        primary_ts = valid_ts[0]

        records.append({
            'timestamp': primary_ts,
            'solar_voltage_ll_avg': voltage,
            'solar_current_avg': current,
            'solar_power_factor': pf,
            'solar_active_power_kw': active_power,
            'solar_apparent_power_kva': apparent_power,
            'solar_reactive_power_kvar': reactive_power,
            'solar_active_energy_kwh': active_energy,
        })

    df = pd.DataFrame(records)
    for col in df.columns:
        if col != 'timestamp':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df = df.drop_duplicates(subset=['timestamp'], keep='first')

    print(f"   ✅ Loaded {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


# ============================================================================
# 4. MERGE & ENGINEER FEATURES
# ============================================================================
def merge_interval_data(df_incomer, df_solar):
    """Merge incomer and solar data on nearest timestamp, then engineer features."""
    print("🔧 Merging and engineering features...")

    # Merge on timestamp (use merge_asof for nearest match)
    df_incomer = df_incomer.sort_values('timestamp')
    df_solar = df_solar.sort_values('timestamp')

    df = pd.merge_asof(
        df_incomer, df_solar,
        on='timestamp',
        tolerance=pd.Timedelta('15min'),
        direction='nearest'
    )

    # --- Temporal Features ---
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Is daylight (approx 6 AM to 6 PM)
    df['is_daylight'] = ((df['hour'] >= 6) & (df['hour'] < 18)).astype(int)

    # --- Electrical Features ---
    df['apparent_vs_active_ratio'] = (
        df['apparent_power_kva'] / df['active_power_kw'].replace(0, np.nan)
    )
    df['reactive_power_ratio'] = (
        df['reactive_power_kvar'].abs() / df['active_power_kw'].replace(0, np.nan)
    )
    df['voltage_deviation'] = (df['voltage_ll_avg'] - df['voltage_ll_avg'].mean()).abs()

    # --- Solar Features ---
    df['solar_contribution_ratio'] = (
        df['solar_active_power_kw'].abs() /
        (df['active_power_kw'].abs() + df['solar_active_power_kw'].abs()).replace(0, np.nan)
    )
    df['net_grid_power'] = df['active_power_kw'] - df['solar_active_power_kw'].abs().fillna(0)

    # --- Energy Consumption (target variable) ---
    # Calculate interval consumption from cumulative energy readings
    df['energy_consumption'] = df['active_energy_kwh'].diff()
    # Remove negative values (meter resets) and extreme outliers
    df.loc[df['energy_consumption'] < 0, 'energy_consumption'] = np.nan
    q99 = df['energy_consumption'].quantile(0.99)
    df.loc[df['energy_consumption'] > q99 * 3, 'energy_consumption'] = np.nan

    # --- Lag Features ---
    df['energy_lag_1'] = df['energy_consumption'].shift(1)   # 15 min ago
    df['energy_lag_4'] = df['energy_consumption'].shift(4)   # 1 hour ago
    df['energy_lag_96'] = df['energy_consumption'].shift(96)  # 24 hours ago

    df['power_lag_1'] = df['active_power_kw'].shift(1)
    df['power_lag_4'] = df['active_power_kw'].shift(4)
    df['voltage_lag_1'] = df['voltage_ll_avg'].shift(1)

    # --- Rolling Features ---
    df['energy_roll_mean_4'] = df['energy_consumption'].rolling(4, min_periods=1).mean()
    df['energy_roll_std_4'] = df['energy_consumption'].rolling(4, min_periods=1).std()
    df['energy_roll_mean_24'] = df['energy_consumption'].rolling(96, min_periods=1).mean()
    df['power_roll_mean_4'] = df['active_power_kw'].rolling(4, min_periods=1).mean()
    df['power_roll_std_4'] = df['active_power_kw'].rolling(4, min_periods=1).std()

    # Clean up infinities
    df = df.replace([np.inf, -np.inf], np.nan)

    print(f"   ✅ Merged dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   📊 Features: {list(df.columns)}")
    return df


# ============================================================================
# 5. TRAIN / TEST SPLIT
# ============================================================================
def train_test_split_timeseries(df, test_ratio=0.25):
    """Chronological train/test split (no shuffling for time-series!)."""
    print("✂️  Splitting train/test (chronological)...")

    n = len(df)
    split_idx = int(n * (1 - test_ratio))

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    print(f"   Train: {len(train)} rows ({train['timestamp'].min()} → {train['timestamp'].max()})")
    print(f"   Test:  {len(test)} rows ({test['timestamp'].min()} → {test['timestamp'].max()})")
    return train, test


# ============================================================================
# 6. PREPARE ML-READY FEATURES
# ============================================================================
FEATURE_COLUMNS = [
    'hour', 'minute', 'day_of_week', 'day_of_month', 'month',
    'is_weekend', 'is_daylight',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'voltage_ll_avg', 'current_avg', 'power_factor',
    'active_power_kw', 'apparent_power_kva', 'reactive_power_kvar',
    'voltage_deviation', 'apparent_vs_active_ratio', 'reactive_power_ratio',
    'solar_active_power_kw', 'solar_current_avg', 'solar_power_factor',
    'solar_contribution_ratio', 'net_grid_power',
    'energy_lag_1', 'energy_lag_4', 'energy_lag_96',
    'power_lag_1', 'power_lag_4', 'voltage_lag_1',
    'energy_roll_mean_4', 'energy_roll_std_4', 'energy_roll_mean_24',
    'power_roll_mean_4', 'power_roll_std_4',
]

TARGET_COLUMN = 'energy_consumption'


def prepare_ml_data(train, test):
    """Prepare feature matrices X and target vectors y for ML."""
    print("🎯 Preparing ML-ready data...")

    # Select features that exist
    available_features = [f for f in FEATURE_COLUMNS if f in train.columns]

    # Drop rows where target is NaN
    train_clean = train.dropna(subset=[TARGET_COLUMN])
    test_clean = test.dropna(subset=[TARGET_COLUMN])

    X_train = train_clean[available_features].copy()
    y_train = train_clean[TARGET_COLUMN].copy()
    X_test = test_clean[available_features].copy()
    y_test = test_clean[TARGET_COLUMN].copy()

    # Fill NaN features with median
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=available_features, index=X_train.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=available_features, index=X_test.index)

    print(f"   ✅ X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   ✅ X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    print(f"   📊 Features used: {len(available_features)}")

    return X_train, y_train, X_test, y_test, available_features


# ============================================================================
# 7. MASTER LOADER
# ============================================================================
def load_all_data():
    """Master function: load, merge, engineer, split, and return everything."""
    print("=" * 70)
    print("⚡ EED SmartGrid Analytics — Data Pipeline")
    print("=" * 70)

    # Load individual sources
    df_daily = load_daily_reports()
    df_incomer = load_incomer_data()
    df_solar = load_solar_data()

    # Merge interval data
    df_merged = merge_interval_data(df_incomer, df_solar)

    # Train/test split
    train, test = train_test_split_timeseries(df_merged)

    # Prepare ML data
    X_train, y_train, X_test, y_test, features = prepare_ml_data(train, test)

    return {
        'daily': df_daily,
        'incomer': df_incomer,
        'solar': df_solar,
        'merged': df_merged,
        'train': train,
        'test': test,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'features': features,
    }


if __name__ == "__main__":
    data = load_all_data()
    print("\n" + "=" * 70)
    print("📋 DATA PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Daily reports:     {len(data['daily'])} records, {data['daily']['date'].nunique()} days")
    print(f"Incomer (15-min):  {len(data['incomer'])} records")
    print(f"Solar (15-min):    {len(data['solar'])} records")
    print(f"Merged dataset:    {data['merged'].shape}")
    print(f"Train set:         {data['X_train'].shape}")
    print(f"Test set:          {data['X_test'].shape}")
    print(f"Target stats:\n{data['y_train'].describe()}")
    print(f"\nFeatures ({len(data['features'])}):")
    for f in data['features']:
        print(f"  • {f}")

    # Save processed data
    data['merged'].to_csv(os.path.join(BASE_DIR, 'processed_data.csv'), index=False)
    data['daily'].to_csv(os.path.join(BASE_DIR, 'daily_consumption.csv'), index=False)
    print("\n💾 Saved processed_data.csv and daily_consumption.csv")
