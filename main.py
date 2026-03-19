"""
EED SmartGrid Analytics — Main Runner
=======================================
Orchestrates the entire pipeline: data loading → ML training → analytics → dashboard.
"""

import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_pipeline():
    """Run the complete analytics pipeline."""
    print("=" * 70)
    print("⚡ EED SmartGrid Analytics — Full Pipeline Runner")
    print("=" * 70)
    start = time.time()

    # 1. Data Pipeline
    print("\n" + "=" * 70)
    print("STEP 1: DATA PIPELINE")
    print("=" * 70)
    from data_pipeline import load_all_data
    data = load_all_data()

    # 2. ML Model Training
    print("\n" + "=" * 70)
    print("STEP 2: ML MODEL TRAINING")
    print("=" * 70)
    from ml_models import train_all_models
    ml_results = train_all_models(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        data['features']
    )

    # 3. Advanced Analytics
    print("\n" + "=" * 70)
    print("STEP 3: ADVANCED ANALYTICS")
    print("=" * 70)
    from analytics import run_all_analytics
    analytics_results = run_all_analytics(data)

    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print(f"✅ PIPELINE COMPLETE in {elapsed:.1f}s")
    print("=" * 70)
    print("\nGenerated files:")
    for f in ['processed_data.csv', 'daily_consumption.csv', 'model_comparison.csv',
              'predictions.csv', 'feature_importances.csv', 'anomalies.csv']:
        fp = os.path.join(BASE_DIR, f)
        if os.path.exists(fp):
            size = os.path.getsize(fp)
            print(f"  ✅ {f} ({size:,} bytes)")
        else:
            print(f"  ❌ {f} (not found)")

    print(f"\n🚀 To launch the dashboard:")
    print(f"   streamlit run dashboard.py")

    return data, ml_results, analytics_results


if __name__ == "__main__":
    run_pipeline()
