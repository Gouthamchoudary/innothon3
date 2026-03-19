"""
EED SmartGrid Analytics — ML Models
=====================================
Trains 7 ML algorithms on the energy consumption data
and produces comparison metrics.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE, handling zeros."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float('inf')
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(y_true, y_pred, model_name):
    """Compute all evaluation metrics for a model."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return {
        'model': model_name,
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R2_Score': round(r2, 4),
        'MAPE_%': round(mape, 2),
    }


def train_all_models(X_train, y_train, X_test, y_test, feature_names):
    """Train all 7 ML models and return comparison results."""
    print("\n" + "=" * 70)
    print("🤖 EED SmartGrid Analytics — ML Model Training")
    print("=" * 70)

    # Scale features for SVR and LSTM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    predictions = {}
    models = {}
    feature_importances = {}

    # ======================================================================
    # 1. LINEAR REGRESSION
    # ======================================================================
    print("\n📊 [1/7] Training Linear Regression...")
    t0 = time.time()
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    elapsed = time.time() - t0
    results.append({**evaluate_model(y_test, pred_lr, 'Linear Regression'), 'time_sec': round(elapsed, 2)})
    predictions['Linear Regression'] = pred_lr
    models['Linear Regression'] = lr
    print(f"   ✅ Done in {elapsed:.2f}s | R²={results[-1]['R2_Score']} | RMSE={results[-1]['RMSE']}")

    # ======================================================================
    # 2. RIDGE REGRESSION
    # ======================================================================
    print("\n📊 [2/7] Training Ridge Regression...")
    t0 = time.time()
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    pred_ridge = ridge.predict(X_test)
    elapsed = time.time() - t0
    results.append({**evaluate_model(y_test, pred_ridge, 'Ridge Regression'), 'time_sec': round(elapsed, 2)})
    predictions['Ridge Regression'] = pred_ridge
    models['Ridge Regression'] = ridge
    print(f"   ✅ Done in {elapsed:.2f}s | R²={results[-1]['R2_Score']} | RMSE={results[-1]['RMSE']}")

    # ======================================================================
    # 3. DECISION TREE
    # ======================================================================
    print("\n🌲 [3/7] Training Decision Tree...")
    t0 = time.time()
    dt = DecisionTreeRegressor(max_depth=10, min_samples_split=10, random_state=42)
    dt.fit(X_train, y_train)
    pred_dt = dt.predict(X_test)
    elapsed = time.time() - t0
    results.append({**evaluate_model(y_test, pred_dt, 'Decision Tree'), 'time_sec': round(elapsed, 2)})
    predictions['Decision Tree'] = pred_dt
    models['Decision Tree'] = dt
    feature_importances['Decision Tree'] = dict(zip(feature_names, dt.feature_importances_))
    print(f"   ✅ Done in {elapsed:.2f}s | R²={results[-1]['R2_Score']} | RMSE={results[-1]['RMSE']}")

    # ======================================================================
    # 4. RANDOM FOREST
    # ======================================================================
    print("\n🌳 [4/7] Training Random Forest...")
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    elapsed = time.time() - t0
    results.append({**evaluate_model(y_test, pred_rf, 'Random Forest'), 'time_sec': round(elapsed, 2)})
    predictions['Random Forest'] = pred_rf
    models['Random Forest'] = rf
    feature_importances['Random Forest'] = dict(zip(feature_names, rf.feature_importances_))
    print(f"   ✅ Done in {elapsed:.2f}s | R²={results[-1]['R2_Score']} | RMSE={results[-1]['RMSE']}")

    # ======================================================================
    # 5. XGBOOST
    # ======================================================================
    print("\n🚀 [5/7] Training XGBoost...")
    t0 = time.time()
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0, n_jobs=-1
        )
        xgb.fit(X_train, y_train)
        pred_xgb = xgb.predict(X_test)
        elapsed = time.time() - t0
        results.append({**evaluate_model(y_test, pred_xgb, 'XGBoost'), 'time_sec': round(elapsed, 2)})
        predictions['XGBoost'] = pred_xgb
        models['XGBoost'] = xgb
        feature_importances['XGBoost'] = dict(zip(feature_names, xgb.feature_importances_))
        print(f"   ✅ Done in {elapsed:.2f}s | R²={results[-1]['R2_Score']} | RMSE={results[-1]['RMSE']}")
    except ImportError:
        print("   ⚠️ XGBoost not installed, skipping")

    # ======================================================================
    # 6. SVR (Support Vector Regression)
    # ======================================================================
    print("\n🔮 [6/7] Training SVR...")
    t0 = time.time()
    # Use subset for SVR if data is too large (SVR is O(n²))
    n_svr = min(len(X_train_scaled), 3000)
    svr = SVR(kernel='rbf', C=10, epsilon=0.1, gamma='scale')
    svr.fit(X_train_scaled[:n_svr], y_train.iloc[:n_svr])
    pred_svr = svr.predict(X_test_scaled)
    elapsed = time.time() - t0
    results.append({**evaluate_model(y_test, pred_svr, 'SVR (RBF)'), 'time_sec': round(elapsed, 2)})
    predictions['SVR (RBF)'] = pred_svr
    models['SVR (RBF)'] = svr
    print(f"   ✅ Done in {elapsed:.2f}s | R²={results[-1]['R2_Score']} | RMSE={results[-1]['RMSE']}")

    # ======================================================================
    # 7. LSTM Neural Network
    # ======================================================================
    print("\n🧠 [7/7] Training LSTM Neural Network...")
    t0 = time.time()
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping

        # Reshape for LSTM [samples, timesteps, features]
        n_steps = 4  # Use 4 previous steps (1 hour of data)
        n_features = X_train_scaled.shape[1]

        def create_sequences(X, y, n_steps):
            Xs, ys = [], []
            for i in range(n_steps, len(X)):
                Xs.append(X[i - n_steps:i])
                ys.append(y.iloc[i] if hasattr(y, 'iloc') else y[i])
            return np.array(Xs), np.array(ys)

        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, n_steps)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, n_steps)

        model = Sequential([
            LSTM(64, input_shape=(n_steps, n_features), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(
            X_train_seq, y_train_seq,
            epochs=50, batch_size=32,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0
        )

        pred_lstm = model.predict(X_test_seq, verbose=0).flatten()
        elapsed = time.time() - t0

        # Adjust test indices for sequences
        y_test_lstm = y_test.iloc[n_steps:].values if len(y_test) > n_steps else y_test.values
        if len(pred_lstm) < len(y_test_lstm):
            y_test_lstm = y_test_lstm[:len(pred_lstm)]
        elif len(pred_lstm) > len(y_test_lstm):
            pred_lstm = pred_lstm[:len(y_test_lstm)]

        results.append({**evaluate_model(y_test_lstm, pred_lstm, 'LSTM'), 'time_sec': round(elapsed, 2)})
        predictions['LSTM'] = pred_lstm
        models['LSTM'] = model
        print(f"   ✅ Done in {elapsed:.2f}s | R²={results[-1]['R2_Score']} | RMSE={results[-1]['RMSE']}")
    except Exception as e:
        print(f"   ⚠️ LSTM failed: {e}")

    # ======================================================================
    # RESULTS COMPARISON
    # ======================================================================
    print("\n" + "=" * 70)
    print("📊 MODEL COMPARISON RESULTS")
    print("=" * 70)

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('R2_Score', ascending=False).reset_index(drop=True)
    df_results.index = df_results.index + 1
    df_results.index.name = 'Rank'
    print(df_results.to_string())

    best_model = df_results.iloc[0]['model']
    print(f"\n🏆 Best Model: {best_model} (R² = {df_results.iloc[0]['R2_Score']}, RMSE = {df_results.iloc[0]['RMSE']})")

    # Save results
    df_results.to_csv(os.path.join(BASE_DIR, 'model_comparison.csv'), index=True)

    # Save predictions
    pred_df = pd.DataFrame({'actual': y_test.values})
    for name, preds in predictions.items():
        if name == 'LSTM':
            # LSTM has fewer predictions due to sequences - pad with NaN
            padded = np.full(len(y_test), np.nan)
            padded[4:4 + len(preds)] = preds  # n_steps=4
            pred_df[name] = padded
        else:
            pred_df[name] = preds
    pred_df.to_csv(os.path.join(BASE_DIR, 'predictions.csv'), index=False)

    # Save feature importances
    if feature_importances:
        fi_df = pd.DataFrame(feature_importances)
        fi_df.index.name = 'feature'
        fi_df.to_csv(os.path.join(BASE_DIR, 'feature_importances.csv'))

    # Save models
    models_dir = os.path.join(BASE_DIR, 'saved_models')
    os.makedirs(models_dir, exist_ok=True)
    for name, model in models.items():
        if name == 'LSTM':
            model.save(os.path.join(models_dir, 'lstm_model.keras'))
        else:
            safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            joblib.dump(model, os.path.join(models_dir, f'{safe_name}.pkl'))

    # Save scaler
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))

    print(f"\n💾 Saved: model_comparison.csv, predictions.csv, feature_importances.csv")
    print(f"💾 Saved models to: {models_dir}/")

    return {
        'results': df_results,
        'predictions': predictions,
        'models': models,
        'feature_importances': feature_importances,
        'scaler': scaler,
    }


if __name__ == "__main__":
    from data_pipeline import load_all_data
    data = load_all_data()
    ml_results = train_all_models(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        data['features']
    )
