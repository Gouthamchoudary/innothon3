# EED SmartGrid Analytics — Comprehensive Documentation

Welcome to the comprehensive documentation for the **EED SmartGrid Analytics Platform**. This document explains every aspect of the project, including the core concepts, data sources, machine learning models, analytics logic, and the AI chatbot integration.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Data Sources & Pipeline (`data_pipeline.py`)](#2-data-sources--pipeline)
3. [Machine Learning Models (`ml_models.py`)](#3-machine-learning-models)
4. [Advanced Analytics (`analytics.py`)](#4-advanced-analytics)
5. [AI Chatbot Integration (`chatbot.py`)](#5-ai-chatbot-integration)
6. [Pipeline Execution (`main.py`)](#6-pipeline-execution)

---

## 1. Project Overview

The **EED SmartGrid Analytics Platform** is a data-driven system built for the National Institute of Technology (NIT) Warangal's Electrical Engineering Department (EED) building. Its primary purpose is to:
- Monitor and forecast electrical energy consumption.
- Detect abnormal energy usage (anomalies).
- Analyze the behavior of grid consumption vs. solar generation.
- Provide an interactive, context-aware AI assistant to interpret the data based on the academic calendar.

It processes raw electrical grid data, trains predictive algorithms, runs deep analytics, and presents the findings in a professional dashboard.

---

## 2. Data Sources & Pipeline
**File:** `data_pipeline.py`

The data pipeline is responsible for collecting dirty, raw data from multiple files, aligning them accurately, and generating powerful features for the AI to learn from.

### 2.1 Where is the Data Taken From?
The system utilizes three primary raw data sources:
1. **Daily Data Reports (`Daily data from dec18_to_jan18/`):** 32 Excel files representing daily consumption (kWh) over a month for explicit feeder meters (e.g., EED Load, Civil Load, Solar).
2. **Incomer Data (`eedincomer.xlsx`):** Granular 15-minute interval data of grid electrical metrics (Voltage, Current, Power Factor, Active Power, etc.).
3. **Solar Generation Data (`researchwindsolar.xlsx`):** Granular 15-minute interval data of research wing solar panel generation.

### 2.2 How Does the Pipeline Work?
- **Data Loading & Parsing:** The system reads Excel files, drops missing values, and strictly formats timestamps (e.g., `16-12-2025@15:00:00.000` to standard Pandas datetime).
- **Merging (`merge_asof`):** It intelligently merges the 15-minute Incomer data with the 15-minute Solar data based on the *closest timestamp*.
- **Feature Engineering:** We create new columns from the existing data to help ML models find patterns:
  - **Temporal Features:** Hour, Minute, Day of Week, Month, Is Weekend, Is Daylight (6 AM - 6 PM).
  - **Cyclical Features:** `hour_sin`, `hour_cos` (helps the model understand that 23:59 is chronologically next to 00:00).
  - **Electrical Features:** Apparent vs Active ratio (efficiency indicator), Voltage deviation from the mean, Net Grid Power (Grid - Solar).
  - **Lag Features:** Energy consumed 1 step ago, 4 steps ago (1 hour ago), 96 steps ago (24 hours ago). 
  - **Rolling Features:** Rolling averages and standard deviations over the last 1 hour and 24 hours.

### 2.3 Train / Test Split
Since this is **Time-Series Data**, we *cannot* randomly shuffle the data. Instead, a chronological split is used (e.g., first 75% of time for training, last 25% for testing) to prevent the model from "peeking into the future".

---

## 3. Machine Learning Models
**File:** `ml_models.py`

Once data is prepped, we train 7 distinct machine learning algorithms to forecast `energy_consumption` based on historical and electrical features. Testing multiple models helps us pick the most accurate one.

### 3.1 The Models Used

#### 1. Linear Regression
- **Concept:** Fits a straight straight line through the data.
- **Why it's used here:** Serves as a fundamental baseline. It is very fast and easy to interpret. If energy scales linearly with time of day or voltage, this catches it.

#### 2. Ridge Regression
- **Concept:** An extension of Linear Regression that shrinks large coefficients to prevent overfitting (adding a penalty to complex models).
- **Why it's used here:** Because we engineered many features (lags, rolling means), some might be highly correlated (multicollinearity). Ridge stabilizes the model.

#### 3. Decision Tree
- **Concept:** Splits the data through a series of "If-Else" rules (e.g., *Is hour > 18? If yes, is day_of_week = Sunday?*).
- **Why it's used here:** Good at capturing non-linear behavior (like sudden load drops at night).
- **Basic Example:** If it's a weekend and time is 2 AM, predict low consumption. 

#### 4. Random Forest (Ensemble Model)
- **Concept:** Builds multiple (100+) Decision Trees using random subsets of data, then averages their predictions. 
- **Why it's used here:** Usually highly accurate and robust against overfitting compared to a single Decision Tree. Excellent for complex multi-variable tabular data.

#### 5. XGBoost (Extreme Gradient Boosting)
- **Concept:** Also builds multiple trees, but *sequentially*. Each new tree actively tries to correct the errors made by the previous trees.
- **Why it's used here:** It is often the top-performing model in tabular data competitions (like Kaggle). It handles missing values well and runs incredibly fast.

#### 6. SVR (Support Vector Regression)
- **Concept:** Tries to fit a "tube" around the data points in high-dimensional space. Points inside the tube are ignored; points outside define the boundary. uses an RBF (Radial Basis Function) kernel for complex curves.
- **Why it's used here:** Can map highly complex, non-linear relationships in limited, scaled datasets.

#### 7. LSTM (Long Short-Term Memory Neural Network)
- **Concept:** A type of Deep Neural Network architecture specifically designed to "remember" sequential patterns over time.
- **Why it's used here:** Best suited for time-series forecasting. It explicitly looks at the sequence of the past 4 timesteps (1 hour) to predict the next step.
- **How it works:** Contains gates (Input, Output, Forget gates) that decide which information from the past is relevant to carry forward to the future.

### 3.2 Evaluation Metrics
How do we know which model is best? We use four metrics:
- **MAE (Mean Absolute Error):** On average, how many kWh were our predictions off by?
- **RMSE (Root Mean Squared Error):** Penalizes very large errors heavily.
- **R² Score:** The percentage of variance explained by the model (1.0 is perfect, 0.0 is predicting the mean).
- **MAPE:** Mean Absolute Percentage Error (Percentage off by average).

---

## 4. Advanced Analytics
**File:** `analytics.py`

This module generates actionable business insights from the ML and statistical data.

### 4.1 Anomaly Detection (Isolation Forest)
- **Concept:** The `Isolation Forest` algorithm builds random decision trees. Anomalies (outliers) are points that are isolated close to the root of the tree (it takes fewer splits to separate them from the rest of the data).
- **Why it's used:** To automatically detect hardware faults, power spikes, meter resets, or suspicious after-hours usage without explicit human labeling. 
- **Example:** High active power at 3:00 AM on a Sunday immediately gets flagged as a `🔴 Critical` anomaly.

### 4.2 Peak Load & Feeder Analysis
- Aggregates the data to show average power at each hour, comparing `weekday vs weekend`.
- Shows which feeder (e.g., Civil vs EED) consumes the most power.

### 4.3 Solar Analytics
- Calculates the `Self-Sufficiency Ratio` (How much of the grid consumption was met by our internal solar panels).

---

## 5. AI Chatbot Integration
**File:** `chatbot.py`

The dashboard isn't just static charts. It features an intelligent AI Assistant using the **Google Gemini (1.5 Flash)** Large Language Model.

### 5.1 How It Works
- **System Prompting:** We silently inject a giant context block behind the scenes. We tell the AI: *"You are an assistant. Here is the NIT Warangal Academic Calendar..."*.
- **Academic Context:** The context contains specific dates (Mid-semesters, Spring Spree cultural fest, holidays).
- **Inference Strategy:** If a user clicks an anomaly on February 28th and asks the bot "Why did power spike?", the AI cross-references the systemic prompt and realizes *"Feb 28 is Spring Spree, high nighttime load is expected due to the cultural festival"* rather than assuming a technical baseline fault.

### 5.2 Why It's Used Here
Raw graphs are intimidating to non-engineers. The AI acts as a translation layer, bridging raw power metrics (like poor Power Factor) and real-world campus events to explain *why* the metrics look the way they do.

---

## 6. Pipeline Execution
**File:** `main.py`

This is the central execution script. It linearly orchestrates the system:
1. Calls `data_pipeline.py` to ingest and process data.
2. Passes processed data to `ml_models.py` to train AI models.
3. Passes data to `analytics.py` to compute anomalies and solar statistics.
4. Generates standard `.csv` files (e.g., `processed_data.csv`, `predictions.csv`) which the interactive `dashboard.py` Streamlit UI securely consumes.

---
**Document End**
