# Quick Start Guide

This guide will help you get started with the Energy Forecasting System in 5 minutes.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 1-2 GB free disk space

## Step 1: Installation (2 minutes)

```bash
# Install required packages
pip install -r requirements.txt

# Create directories
mkdir -p data models visualizations
```

## Step 2: Get Dataset (1 minute)

Download the household power consumption dataset:

**Option A**: Use UCI ML Repository dataset
- Visit: https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption
- Download: household_power_consumption.zip
- Extract and place `household_power_consumption.txt` in the `data/` folder

**Option B**: Use your own data
- Format your data to match the expected structure (see README.md)
- Place it in `data/household_power_consumption.txt`

## Step 3: Train and Predict (2 minutes)

```bash
# Option 1: Run full pipeline with visualizations
python main.py

# Option 2: Train model only and make quick prediction
python predict.py --train --date 2007-12-15
```

## What Happens Next?

### During Training:
1. Data preprocessing: ~30 seconds
2. Anomaly detection: ~15 seconds
3. Profile clustering: ~20 seconds
4. Model training: ~45 seconds
5. Evaluation: ~10 seconds
6. Visualization: ~30 seconds

**Total time: ~2-3 minutes** (depends on dataset size)

### You'll Get:
- Trained model saved in `models/energy_forecaster.pkl`
- 6 visualizations in `visualizations/` folder
- Summary report with model performance
- Example predictions printed to console

## Making Predictions

After training, predict energy consumption for any date:

```bash
# Natural language query
python predict.py "Predict tomorrow's energy usage for Household 1001"

# Specific date
python predict.py --date 2007-12-15

# Multiple predictions
python predict.py --date 2007-11-20
```

## Understanding the Output

```
Date: 2007-12-15 (Saturday)

Forecast: 24.5 kWh          ← Predicted daily consumption
Actual: 24.8 kWh            ← Actual consumption (if available)
Error: 0.3 kWh              ← Prediction error
Error %: 1.21%              ← Percentage error

Top Contributing Features:   ← Most important factors
  lag_1d: 23.456            ← Yesterday's consumption
  same_day_last_week: 24.123 ← Last Saturday's consumption
  rolling_7d_mean: 22.890   ← 7-day average
  day_of_week: 5            ← Saturday = 5
  is_weekend: 1             ← Weekend flag
```

## Exploring Visualizations

Check the `visualizations/` folder for:

1. **01_daily_trend_with_anomalies.png**: See consumption over time with anomalies marked
2. **02_hourly_patterns.png**: Understand typical consumption by hour, day, month
3. **03_cluster_profiles.png**: View different daily consumption patterns
4. **04_forecast_results.png**: Compare predictions vs actual values
5. **05_feature_importance.png**: See which features matter most
6. **06_anomaly_summary.png**: Analyze detected anomalies

## Common Questions

**Q: How much data do I need?**
A: Minimum 1 month, recommended 1+ year for best results

**Q: Can I use my own dataset?**
A: Yes! Format it as semicolon-separated with datetime and consumption columns

**Q: How accurate are the predictions?**
A: Typically 8-12% MAPE (Mean Absolute Percentage Error)

**Q: Can I predict future dates not in the dataset?**
A: Currently only dates within the dataset range are supported

**Q: How do I improve accuracy?**
A: Use more historical data, ensure data quality, consider weather data integration

## Next Steps

1. **Analyze patterns**: Review visualizations to understand consumption behavior
2. **Experiment**: Try different dates, adjust anomaly thresholds
3. **Customize**: Modify parameters in `main.py` for your use case
4. **Extend**: Add new features, integrate weather data, try different models

## Troubleshooting

**Error: "No such file or directory: data/household_power_consumption.txt"**
- Make sure you've placed the dataset in the correct location

**Error: "ModuleNotFoundError"**
- Run `pip install -r requirements.txt` again

**Warning: "Low memory"**
- Try reducing dataset size or using a machine with more RAM

**Issue: Predictions seem inaccurate**
- Check data quality (missing values, anomalies)
- Ensure sufficient training data (>6 months)

## Support

Need help? Check:
- Full documentation in `README.md`
- Example code in `main.py` and `predict.py`
- Module documentation in `src/` folder

Ready to dive deeper? Read the full README.md!

---

Happy forecasting! ⚡
