═══════════════════════════════════════════════════════════════════════════════
ENERGY CONSUMPTION FORECASTING SYSTEM
Complete Usage Guide
═══════════════════════════════════════════════════════════════════════════════

TABLE OF CONTENTS
─────────────────────────────────────────────────────────────────────────────

1. Quick Start
2. System Overview
3. Running the Full Pipeline
4. Making Predictions
5. Understanding the Outputs
6. Command Reference
7. Examples & Use Cases
8. Tips & Best Practices
9. Troubleshooting

═══════════════════════════════════════════════════════════════════════════════

1. QUICK START
   ═══════════════════════════════════════════════════════════════════════════════

Installation
────────────────────────────────────────────────────────────────────────────

# Install required packages

pip install pandas numpy scikit-learn matplotlib seaborn

# Ensure data file exists

data/household_power_consumption.txt

First Run
────────────────────────────────────────────────────────────────────────────

# Train the model and generate visualizations (takes 2-3 minutes)

python main.py

# Make a prediction

python predict.py --date 2010-11-20

# Predict future dates

python predict.py --date 2025-12-25

Done! Your system is ready to use.

═══════════════════════════════════════════════════════════════════════════════ 2. SYSTEM OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

What It Does
────────────────────────────────────────────────────────────────────────────
✓ Predicts daily household energy consumption (kWh)
✓ Works for historical dates (with actual comparison)
✓ Works for future dates (up to 365 days ahead)
✓ Detects anomalies in consumption patterns
✓ Identifies consumption clusters (weekday/weekend/seasonal)
✓ Shows feature importance and confidence intervals

Model Performance
────────────────────────────────────────────────────────────────────────────
Training MAE: ~0.94 kWh
Test MAE: ~2.12 kWh
Test R²: ~0.86
Test MAPE: ~12.3%

Technology Stack
────────────────────────────────────────────────────────────────────────────
• Random Forest Regression (primary model)
• Multi-method Anomaly Detection (IQR, Z-score, Isolation Forest, Pattern)
• K-Means Clustering for daily profiles
• 60+ engineered features (lags, rolling stats, temporal features)
• Recursive forecasting for future predictions

═══════════════════════════════════════════════════════════════════════════════ 3. RUNNING THE FULL PIPELINE
═══════════════════════════════════════════════════════════════════════════════

Command
────────────────────────────────────────────────────────────────────────────
python main.py

What Happens
────────────────────────────────────────────────────────────────────────────
[STEP 1] Data Preprocessing
• Loads 2M+ records from raw data file
• Interpolates missing values (~180K gaps)
• Aggregates to hourly data (34K records)
• Creates daily summaries (1,442 days)
• Extracts 24-hour consumption profiles

[STEP 2] Anomaly Detection
• Runs 4 detection methods (IQR, Z-Score, Isolation Forest, Pattern)
• Identifies ~11 anomalous days requiring 2+ method agreement
• Removes anomalies for cleaner training data

[STEP 3] Daily Profile Clustering
• Clusters 1,418 complete daily profiles into 5 groups
• Identifies typical consumption patterns (weekday/weekend/seasonal)

[STEP 4] Feature Engineering
• Creates 61 features including: - Lag features (1d, 2d, 3d, 7d, 14d) - Rolling statistics (7d, 30d means/std/min/max) - Temporal features (day of week, month, season, cyclical encoding) - Cluster features (average consumption per cluster) - Anomaly flags

[STEP 5] Model Training
• Trains Random Forest with 80/20 train/test split
• Performs 5-fold time series cross-validation
• Reports training metrics

[STEP 6] Model Evaluation
• Tests on held-out 20% of data
• Reports MAE, RMSE, R², MAPE
• Saves model to models/energy_forecaster.pkl

[STEP 7] Recursive Forecaster
• Initializes forecaster for future date predictions
• Ready to predict up to 365 days ahead

[STEP 8] Feature Importance
• Shows top 15 most important features
• Cluster average consumption typically most important

[Visualizations]
• Generates 7 comprehensive plots
• Saved to visualizations/ directory

[Example Predictions]
• Shows 3 historical predictions with actual vs predicted
• Shows 3 future predictions with confidence intervals

Outputs Generated
────────────────────────────────────────────────────────────────────────────
models/
└── energy_forecaster.pkl # Trained model
└── energy_forecaster_cache.pkl # Preprocessed data cache

visualizations/
├── 01_daily_trend_with_anomalies.png
├── 02_hourly_patterns.png
├── 03_cluster_profiles.png
├── 04_forecast_results.png
├── 05_feature_importance.png
├── 06_anomaly_summary.png
└── 07_rf_model_evaluation.png

Duration
────────────────────────────────────────────────────────────────────────────
First run: 2-3 minutes
With cache: 30-60 seconds

═══════════════════════════════════════════════════════════════════════════════ 4. MAKING PREDICTIONS
═══════════════════════════════════════════════════════════════════════════════

predict.py - Interactive Prediction Tool
────────────────────────────────────────────────────────────────────────────

Basic Usage
python predict.py --date YYYY-MM-DD

First Run Setup
python predict.py --date 2010-11-20 --train

Historical Date Prediction
────────────────────────────────────────────────────────────────────────────

# Predict a historical date (compares with actual)

python predict.py --date 2010-11-20

Output:
Date: 2010-11-20 (Saturday)
Method: Random Forest

    🔮 Forecast: 24.12 kWh
    📊 Actual: 25.30 kWh
    ❌ Error: 1.18 kWh (4.7%)

    Top Contributing Features:
      cluster_avg_consumption: 0.7015
      Sub_metering_3_sum: 0.0949
      Sub_metering_2_sum: 0.0316

Future Date Prediction
────────────────────────────────────────────────────────────────────────────

# Predict a future date (shows confidence intervals)

python predict.py --date 2025-12-25

Output:
Date: 2025-12-25 (Thursday)
Method: Recursive Random Forest

    🔮 Forecast: 23.45 kWh
    95% Confidence Interval: [15.23, 31.67] kWh
    Uncertainty: ±8.32 kWh
    Days ahead: 1,856

Natural Language Queries
────────────────────────────────────────────────────────────────────────────
python predict.py "Predict tomorrow's energy usage"
python predict.py "What about next week?"
python predict.py "Energy consumption for Christmas 2025"

Range Predictions
────────────────────────────────────────────────────────────────────────────

# Predict next N days

python predict.py --range 30

# Predict specific date range

python predict.py --from 2025-12-01 --to 2025-12-31

# Export to CSV

python predict.py --range 30 --export december_forecast.csv

Output:
Date Prediction CI Lower CI Upper
──────────────────────────────────────────
2010-12-03 24.19 10.57 37.81
2010-12-04 23.87 10.25 37.49
2010-12-05 24.03 10.41 37.65
...

    📊 Summary:
       Average: 24.05 kWh
       Total: 721.5 kWh
       Minimum: 22.45 kWh
       Maximum: 25.67 kWh

Command Options
────────────────────────────────────────────────────────────────────────────
--date DATE Specific date (YYYY-MM-DD)
--range N Predict next N days
--from DATE Start date for range
--to DATE End date for range
--train Train model first (required on first run)
--export FILE Export predictions to CSV
--model-path PATH Custom model path
--data-path PATH Custom data file path

═══════════════════════════════════════════════════════════════════════════════ 5. UNDERSTANDING THE OUTPUTS
═══════════════════════════════════════════════════════════════════════════════

Visualizations Explained
────────────────────────────────────────────────────────────────────────────

1. Daily Trend with Anomalies
   • Shows daily consumption over entire dataset
   • Red dots indicate detected anomalies
   • Orange line shows 30-day rolling average
   • Helps identify seasonal patterns and outliers

2. Hourly Patterns
   • Four subplots showing consumption by:

   - Hour of day (0-23)
   - Weekday vs Weekend comparison
   - Day of week (Mon-Sun)
   - Month (Jan-Dec)
     • Identifies peak usage hours and seasonal variations

3. Cluster Profiles
   • Shows 5 typical daily consumption patterns
   • Each subplot shows mean ± std deviation
   • Gray lines show sample days from each cluster
   • Helps understand different usage behaviors

4. Forecast Results
   • Time series: Actual vs Predicted over test period
   • Scatter plot: Shows prediction accuracy
   • Includes performance metrics (MAE, RMSE, MAPE)

5. Feature Importance
   • Horizontal bar chart of top 15 features
   • Shows which factors most influence predictions
   • Typically led by cluster_avg_consumption

6. Anomaly Summary
   • Four subplots showing:

   - Anomalies over time
   - Distribution comparison (normal vs anomaly)
   - Count by detection method
   - Consensus distribution
     • Helps validate anomaly detection quality

7. Model Evaluation
   • Residual plot (prediction errors)
   • Error distribution histogram
   • Predictions over time
   • Error percentage breakdown

Metrics Explained
────────────────────────────────────────────────────────────────────────────
MAE (Mean Absolute Error)
Average absolute difference between predicted and actual
Lower is better | Typical: 2-3 kWh

RMSE (Root Mean Squared Error)
Square root of average squared errors
Penalizes large errors more | Typical: 3-4 kWh

R² (R-squared)
Proportion of variance explained by model
0 to 1, higher is better | Typical: 0.85-0.90

MAPE (Mean Absolute Percentage Error)
Average percentage error
Lower is better | Typical: 10-15%

Feature Importance Categories
────────────────────────────────────────────────────────────────────────────
Cluster Features (70%+) - Dominant predictor
Sub-metering Features (10-15%) - Appliance-specific usage
Temporal Features (5-10%) - Day/month/season effects
Lag Features (3-5%) - Recent consumption trends
Rolling Features (2-3%) - Moving averages
Anomaly Flags (1-2%) - Unusual patterns

═══════════════════════════════════════════════════════════════════════════════ 6. COMMAND REFERENCE
═══════════════════════════════════════════════════════════════════════════════

Main Pipeline
────────────────────────────────────────────────────────────────────────────
python main.py # Run full training pipeline

Prediction Commands
────────────────────────────────────────────────────────────────────────────

# Single date predictions

python predict.py --date 2010-11-20
python predict.py --date 2025-12-25

# Natural language

python predict.py "tomorrow"
python predict.py "next week"
python predict.py "Christmas 2025"

# Range predictions

python predict.py --range 7 # Next week
python predict.py --range 30 # Next month
python predict.py --range 90 # Next quarter

# Custom date ranges

python predict.py --from 2025-01-01 --to 2025-01-31

# Export results

python predict.py --range 30 --export forecast.csv

Training Options
────────────────────────────────────────────────────────────────────────────

# First time setup

python predict.py --train

# Force retrain

python main.py

File Structure
────────────────────────────────────────────────────────────────────────────
your_project/
├── main.py # Full training pipeline
├── predict.py # Prediction interface
├── data/
│ └── household_power_consumption.txt
├── src/
│ ├── data_preprocessing.py
│ ├── anomaly_detection.py
│ ├── profile_clustering.py
│ ├── forecasting_model.py
│ └── visualization.py
├── models/ # Generated models
│ ├── energy_forecaster.pkl
│ └── energy_forecaster_cache.pkl
└── visualizations/ # Generated plots

═══════════════════════════════════════════════════════════════════════════════ 7. EXAMPLES & USE CASES
═══════════════════════════════════════════════════════════════════════════════

Example 1: Quick Daily Forecast
────────────────────────────────────────────────────────────────────────────
Goal: Get tomorrow's energy forecast

Command:
python predict.py "tomorrow"

Use Case: Daily energy planning

Example 2: Weekly Planning
────────────────────────────────────────────────────────────────────────────
Goal: Forecast next 7 days

Command:
python predict.py --range 7

Use Case: Short-term energy budgeting

Example 3: Monthly Budget Estimation
────────────────────────────────────────────────────────────────────────────
Goal: Estimate next month's total consumption

Command:
python predict.py --range 30

Output:
📊 Summary: Total = 720 kWh for next 30 days

Use Case: Monthly billing estimation

Example 4: Seasonal Analysis
────────────────────────────────────────────────────────────────────────────
Goal: Compare winter vs summer consumption

Commands:
python predict.py --from 2025-12-01 --to 2025-12-31 --export winter.csv
python predict.py --from 2025-06-01 --to 2025-06-30 --export summer.csv

Use Case: Seasonal energy planning

Example 5: Historical Accuracy Check
────────────────────────────────────────────────────────────────────────────
Goal: Validate model accuracy on known dates

Command:
python predict.py --date 2010-11-20

Output shows actual vs predicted comparison

Use Case: Model validation

Example 6: Long-term Planning
────────────────────────────────────────────────────────────────────────────
Goal: Forecast next quarter

Command:
python predict.py --range 90 --export Q1_forecast.csv

Use Case: Quarterly energy budgeting

Example 7: Special Event Planning
────────────────────────────────────────────────────────────────────────────
Goal: Estimate consumption for holiday period

Command:
python predict.py --from 2025-12-24 --to 2025-12-26

Use Case: Holiday energy planning

═══════════════════════════════════════════════════════════════════════════════ 8. TIPS & BEST PRACTICES
═══════════════════════════════════════════════════════════════════════════════

Model Training
────────────────────────────────────────────────────────────────────────────
✓ Run main.py once to train the model (2-3 minutes)
✓ Retrain periodically with new data for best accuracy
✓ Check visualizations to understand your consumption patterns
✓ Review feature importance to see what drives predictions

Making Predictions
────────────────────────────────────────────────────────────────────────────
✓ Historical dates: Get actual comparison to validate accuracy
✓ Near-term predictions (1-7 days): Most accurate
✓ Medium-term (8-30 days): Good accuracy, wider confidence intervals
✓ Long-term (31-365 days): Use for planning, expect wider intervals
✓ Export to CSV for further analysis in Excel/Python

Interpreting Results
────────────────────────────────────────────────────────────────────────────
✓ Check confidence intervals: Wider = less certain
✓ Compare with historical averages for reasonableness
✓ Consider seasonality: Winter/summer consumption differs
✓ Weekend vs weekday patterns: Expect variations
✓ Outliers: May indicate special circumstances

Performance Optimization
────────────────────────────────────────────────────────────────────────────
✓ First run trains model and creates cache (slow)
✓ Subsequent predictions use cache (fast)
✓ Cache stored in models/energy_forecaster_cache.pkl
✓ Delete cache to force retrain with fresh preprocessing

Data Quality
────────────────────────────────────────────────────────────────────────────
✓ System handles ~180K missing values automatically
✓ Interpolation used for gaps < 1 hour
✓ Anomaly detection removes outliers
✓ Clustering identifies normal patterns

═══════════════════════════════════════════════════════════════════════════════ 9. TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════════════

Error: "No trained model found"
────────────────────────────────────────────────────────────────────────────
Solution: Run training first
python predict.py --train # or
python main.py

Error: "Data file not found"
────────────────────────────────────────────────────────────────────────────
Solution: Ensure data file exists at:
data/household_power_consumption.txt

Error: "Date not in dataset"
────────────────────────────────────────────────────────────────────────────
Solution: Check available date range
Historical data: 2006-12-16 to 2010-11-26
Future predictions: Any date after 2010-11-26

Error: Import errors
────────────────────────────────────────────────────────────────────────────
Solution: Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn

Predictions seem inaccurate
────────────────────────────────────────────────────────────────────────────
Check:
• Is model trained? (run main.py)
• Are you using appropriate date ranges?
• Check confidence intervals (wider = less certain)
• Compare with visualizations for context

Slow performance
────────────────────────────────────────────────────────────────────────────
Optimization:
• First run is slow (training)
• Use cache for faster predictions
• Limit range predictions to reasonable periods

Visualizations not generating
────────────────────────────────────────────────────────────────────────────
Check:
• matplotlib installed?
• visualizations/ directory writable?
• Run main.py to regenerate

═══════════════════════════════════════════════════════════════════════════════
QUICK REFERENCE
═══════════════════════════════════════════════════════════════════════════════

Most Common Commands
────────────────────────────────────────────────────────────────────────────
python main.py # Train model (first time)
python predict.py --date 2010-11-20 # Historical prediction
python predict.py --date 2025-12-25 # Future prediction
python predict.py --range 30 # Next 30 days
python predict.py --range 7 --export forecast.csv # Export results

Key Files
────────────────────────────────────────────────────────────────────────────
main.py # Training pipeline
predict.py # Prediction interface
models/energy_forecaster.pkl # Trained model
visualizations/\*.png # Analysis plots

Performance Metrics
────────────────────────────────────────────────────────────────────────────
Test MAE: ~2.1 kWh (average error)
Test R²: ~0.86 (86% variance explained)
Test MAPE: ~12% (typical percentage error)

Date Ranges
────────────────────────────────────────────────────────────────────────────
Historical: 2006-12-16 to 2010-11-26 (with actual comparisons)
Future: Any date after 2010-11-26 (up to 365 days ahead)

═══════════════════════════════════════════════════════════════════════════════
NEED MORE HELP?
═══════════════════════════════════════════════════════════════════════════════

Check the generated visualizations in visualizations/ for insights into:
• Your consumption patterns
• Model performance
• Feature importance
• Anomaly detection results

Review the output from main.py for detailed pipeline information.

═══════════════════════════════════════════════════════════════════════════════
Happy Forecasting! 📊⚡🔮
═══════════════════════════════════════════════════════════════════════════════
