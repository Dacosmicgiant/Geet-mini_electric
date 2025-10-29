# Energy Usage Analysis & Forecasting System

A comprehensive machine learning system for analyzing and predicting household/industrial energy consumption using daily pattern analysis and time series features.

## Problem Statement

Predict and analyze household energy consumption patterns to support energy-saving initiatives and provide insights into temporal consumption behaviors.

## Key Innovation: Daily Pattern-Based Analysis

Unlike traditional approaches that treat energy data as one continuous time series, this system recognizes that energy consumption follows distinct **daily patterns**. By analyzing each day as a separate 24-hour profile, we can:

- Capture intra-day fluctuations (morning surge, evening peak, night low)
- Compare similar days (weekdays vs weekends, seasonal variations)
- Identify typical consumption profiles through clustering
- Make more accurate predictions by understanding daily behavioral patterns

## Features

- **Hourly Data Aggregation**: Converts minute-level data to hourly resolution
- **Anomaly Detection**: Multi-method approach (IQR, Z-score, Isolation Forest, Pattern-based)
- **Daily Profile Clustering**: Groups similar consumption patterns
- **Advanced Feature Engineering**: 40+ temporal, lag, and pattern-based features
- **Random Forest Forecasting**: Predicts daily consumption with high accuracy
- **Comprehensive Visualizations**: 6+ visualization types for pattern analysis
- **Simple Query Interface**: Natural language prediction queries

## Technical Stack

- **Python 3.8+**
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning models
- **Matplotlib/Seaborn** - Visualizations
- **NumPy** - Numerical computations
- **Statsmodels** - Statistical analysis

## Installation

```bash
# Clone repository
git clone <repository-url>
cd Geet-mini_electric

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data models visualizations
```

## Dataset Structure

Place your dataset in `data/household_power_consumption.txt` with the following format:

```
Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3
16/12/2006;17:24:00;4.216;0.418;234.840;18.400;0.000;1.000;17.000
16/12/2006;17:25:00;5.360;0.436;233.630;23.000;0.000;1.000;16.000
...
```

**Data Source**: UCI Machine Learning Repository - Individual Household Electric Power Consumption Dataset

## Quick Start

### Option 1: Run Full Pipeline

```bash
# Run complete analysis (training + visualization + predictions)
python main.py
```

This will:
1. Load and preprocess data
2. Detect and remove anomalies
3. Cluster daily profiles
4. Train forecasting model
5. Generate visualizations
6. Show example predictions

### Option 2: Simple Prediction Interface

```bash
# First run: Train the model
python predict.py --train

# Make predictions with natural language
python predict.py "Predict tomorrow's energy usage for Household 1001"

# Predict specific date
python predict.py --date 2007-12-15

# Use different data file
python predict.py --date 2007-11-20 --data-path path/to/data.txt
```

### Sample Output

```
Query: Predict tomorrow's energy usage for Household 1001.
======================================================================

Date: 2007-12-15 (Saturday)

Forecast: 24.5 kWh

Actual: 24.8 kWh
Error: 0.3 kWh
Error %: 1.21%

Top Contributing Features:
  lag_1d: 23.456
  same_day_last_week: 24.123
  rolling_7d_mean: 22.890
  day_of_week: 5
  is_weekend: 1

======================================================================
```

## System Architecture

### 1. Data Preprocessing (`src/data_preprocessing.py`)

- Loads semicolon-separated data
- Handles missing values with interpolation
- Aggregates to hourly resolution (24 points/day)
- Extracts daily summaries and 24-hour profiles
- Adds temporal features (day of week, season, etc.)

### 2. Anomaly Detection (`src/anomaly_detection.py`)

Uses 4 complementary methods:
- **IQR Method**: Statistical outlier detection
- **Z-Score**: Standard deviation-based detection
- **Isolation Forest**: ML-based anomaly detection
- **Pattern Analysis**: Detects days with unusual hourly patterns

Consensus approach: Flag as anomaly if ≥2 methods agree

### 3. Profile Clustering (`src/profile_clustering.py`)

- Clusters days with similar 24-hour consumption patterns
- Auto-determines optimal cluster count (elbow + silhouette)
- Identifies typical profiles for weekdays, weekends, seasons
- Enables similarity-based forecasting

### 4. Forecasting Model (`src/forecasting_model.py`)

**Model**: Random Forest Regressor (200 trees)

**Features** (40+):
- **Temporal**: Day of week, month, season, cyclical encodings
- **Lag**: Previous 1, 2, 3, 7, 14 days consumption
- **Rolling Statistics**: 7-day and 30-day mean/std/min/max
- **Trend**: Week-over-week changes
- **Pattern**: Ratio to typical consumption, day-of-week averages
- **Cluster**: Cluster membership and cluster averages

**Target**: Total daily consumption (kWh)

### 5. Visualization (`src/visualization.py`)

Generates 6 key visualizations:
1. Daily consumption trend with anomalies
2. Hourly patterns (by hour, day of week, month)
3. Cluster profiles with typical patterns
4. Forecast results (actual vs predicted)
5. Feature importance ranking
6. Anomaly detection summary

## Project Structure

```
Geet-mini_electric/
├── data/
│   └── household_power_consumption.txt    # Dataset (not included)
├── src/
│   ├── data_preprocessing.py              # Data loading & preprocessing
│   ├── anomaly_detection.py               # Multi-method anomaly detection
│   ├── profile_clustering.py              # Daily pattern clustering
│   ├── forecasting_model.py               # ML forecasting model
│   └── visualization.py                   # Plotting utilities
├── models/
│   └── energy_forecaster.pkl              # Saved trained model
├── visualizations/
│   ├── 01_daily_trend_with_anomalies.png
│   ├── 02_hourly_patterns.png
│   ├── 03_cluster_profiles.png
│   ├── 04_forecast_results.png
│   ├── 05_feature_importance.png
│   └── 06_anomaly_summary.png
├── main.py                                 # Full pipeline execution
├── predict.py                              # Simple prediction interface
├── requirements.txt                        # Dependencies
└── README.md                               # This file
```

## Usage Examples

### Example 1: Custom Date Prediction

```python
from main import EnergyForecastingSystem

# Initialize system
system = EnergyForecastingSystem("data/household_power_consumption.txt")
system.run_full_pipeline()

# Predict specific date
result = system.predict_day("2007-12-15")
print(f"Predicted: {result['predicted_consumption']:.2f} kWh")
```

### Example 2: Analyze Cluster Characteristics

```python
from src.profile_clustering import DailyProfileClusterer
from src.data_preprocessing import EnergyDataPreprocessor

# Load data
preprocessor = EnergyDataPreprocessor("data/household_power_consumption.txt")
hourly, daily, profiles = preprocessor.process_pipeline()

# Cluster profiles
clusterer = DailyProfileClusterer()
profiles_clustered = clusterer.cluster_profiles(profiles)
summary = clusterer.get_cluster_characteristics(profiles_clustered)
print(summary)
```

### Example 3: Custom Anomaly Detection

```python
from src.anomaly_detection import EnergyAnomalyDetector

# Detect anomalies with custom threshold
detector = EnergyAnomalyDetector(contamination=0.03)
daily_with_anomalies, stats = detector.detect_all_anomalies(
    daily, profiles, consensus_threshold=3
)
print(f"Detected {stats['anomalies_detected']} anomalies")
```

## Performance Metrics

Typical performance on test set (20% holdout):

- **MAE**: 1.5-2.5 kWh
- **RMSE**: 2.0-3.5 kWh
- **R²**: 0.85-0.92
- **MAPE**: 8-12%

*Performance varies based on dataset size and quality*

## Benefits

1. **Energy-Saving Initiatives**: Identify high-consumption patterns and optimize usage
2. **Demand Forecasting**: Predict future consumption for grid planning
3. **Anomaly Detection**: Identify unusual consumption (equipment malfunction, data errors)
4. **Pattern Understanding**: Learn household behavior through daily profiles
5. **Educational**: Teaches time-series analysis, clustering, and feature engineering

## Key Insights

The daily pattern approach reveals:

- **Morning Peak** (7-9 AM): Waking up, breakfast, heating
- **Midday Plateau** (10 AM-4 PM): Background consumption
- **Evening Peak** (6-9 PM): Cooking, entertainment, lighting
- **Night Low** (11 PM-6 AM): Sleep mode, minimal usage
- **Weekend Shift**: Later morning peak, higher midday usage
- **Seasonal Variation**: Higher consumption in winter/summer

## Troubleshooting

**Issue**: Model accuracy is low
- Ensure dataset has sufficient history (>1 year recommended)
- Check for data quality issues (excessive missing values)
- Try adjusting anomaly detection threshold

**Issue**: Memory errors during processing
- Reduce dataset size or process in chunks
- Use lower resolution (2-hour instead of 1-hour intervals)

**Issue**: Predictions not available for recent dates
- Check if date is within dataset range
- Ensure sufficient historical data for feature engineering

## Future Enhancements

- [ ] Weather data integration (temperature, humidity)
- [ ] Holiday calendar for special day handling
- [ ] Multi-horizon forecasting (predict next 7 days)
- [ ] Real-time prediction API
- [ ] Deep learning models (LSTM, Transformer)
- [ ] Multi-household comparative analysis
- [ ] Energy cost optimization recommendations

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Dataset: UCI Machine Learning Repository
- Inspired by real-world energy analytics challenges
- Built with daily pattern analysis innovation

## Contact

For questions or suggestions, please open an issue in the repository.

---

**Made with Daily Pattern Analysis** ⚡
