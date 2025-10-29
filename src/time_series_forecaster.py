"""
Time Series Forecasting Module using SARIMA
Enables prediction of future dates beyond the dataset using seasonal patterns.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import joblib
warnings.filterwarnings('ignore')


class TimeSeriesForecaster:
    """
    SARIMA-based forecaster for predicting energy consumption on future dates.
    """

    def __init__(self):
        """Initialize SARIMA forecaster."""
        self.model = None
        self.sarima_order = None
        self.seasonal_order = None
        self.is_fitted = False
        self.last_date = None
        self.freq = 'D'  # Daily frequency

    def find_best_sarima_params(self, timeseries: pd.Series,
                                max_p=3, max_d=2, max_q=3,
                                seasonal_period=7) -> tuple:
        """
        Find best SARIMA parameters using grid search on AIC.

        Args:
            timeseries: Time series data
            max_p, max_d, max_q: Maximum values for ARIMA parameters
            seasonal_period: Seasonal period (7 for weekly patterns)

        Returns:
            Tuple of (best_order, best_seasonal_order)
        """
        print("Finding optimal SARIMA parameters...")

        best_aic = np.inf
        best_order = None
        best_seasonal_order = None

        # Quick grid search
        for p in range(0, max_p + 1):
            for d in range(0, max_d + 1):
                for q in range(0, max_q + 1):
                    try:
                        # Try simple seasonal component
                        model = SARIMAX(
                            timeseries,
                            order=(p, d, q),
                            seasonal_order=(1, 0, 1, seasonal_period),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        results = model.fit(disp=False, maxiter=100)

                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = (p, d, q)
                            best_seasonal_order = (1, 0, 1, seasonal_period)

                    except:
                        continue

        print(f"Best SARIMA order: {best_order}")
        print(f"Best seasonal order: {best_seasonal_order}")
        print(f"Best AIC: {best_aic:.2f}")

        return best_order, best_seasonal_order

    def fit(self, daily_data: pd.DataFrame,
            target_col='total_daily_consumption',
            auto_params=True):
        """
        Fit SARIMA model on historical data.

        Args:
            daily_data: DataFrame with daily consumption
            target_col: Target column name
            auto_params: Auto-find best parameters
        """
        print("\nTraining SARIMA model for future predictions...")

        # Prepare time series
        ts = daily_data[target_col].copy()
        ts.index = pd.to_datetime(daily_data.index)
        ts = ts.sort_index()

        # Remove any NaN
        ts = ts.dropna()

        self.last_date = ts.index[-1]
        print(f"Training data: {len(ts)} days from {ts.index[0].date()} to {ts.index[-1].date()}")

        # Find or use default parameters
        if auto_params:
            self.sarima_order, self.seasonal_order = self.find_best_sarima_params(
                ts, max_p=2, max_d=1, max_q=2, seasonal_period=7
            )
        else:
            # Default parameters for daily energy consumption
            self.sarima_order = (1, 1, 1)
            self.seasonal_order = (1, 0, 1, 7)  # Weekly seasonality

        print(f"\nFitting SARIMA{self.sarima_order}x{self.seasonal_order}...")

        # Fit the model
        self.model = SARIMAX(
            ts,
            order=self.sarima_order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        self.fitted_model = self.model.fit(disp=False, maxiter=200)
        self.is_fitted = True

        print(f"SARIMA model fitted successfully!")
        print(f"AIC: {self.fitted_model.aic:.2f}")

        return self.fitted_model

    def predict_future(self, target_date: pd.Timestamp) -> float:
        """
        Predict consumption for a future date.

        Args:
            target_date: Future date to predict

        Returns:
            Predicted consumption value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Calculate days ahead
        days_ahead = (target_date - self.last_date).days

        if days_ahead <= 0:
            raise ValueError(f"Target date must be after {self.last_date.date()}")

        # Make forecast
        forecast = self.fitted_model.forecast(steps=days_ahead)

        # Return the last value (target date)
        prediction = forecast.iloc[-1]

        return float(prediction)

    def predict_range(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
        """
        Predict consumption for a range of dates.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Series with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Calculate days ahead
        days_ahead = (end_date - self.last_date).days

        if days_ahead <= 0:
            end_date = self.last_date + pd.Timedelta(days=30)
            days_ahead = 30

        # Make forecast
        forecast = self.fitted_model.forecast(steps=days_ahead)

        # Filter to requested range
        forecast_dates = pd.date_range(
            start=self.last_date + pd.Timedelta(days=1),
            periods=days_ahead,
            freq='D'
        )
        forecast.index = forecast_dates

        return forecast[(forecast.index >= start_date) & (forecast.index <= end_date)]

    def evaluate_on_test(self, test_data: pd.DataFrame,
                        target_col='total_daily_consumption') -> dict:
        """
        Evaluate model on test set.

        Args:
            test_data: Test DataFrame
            target_col: Target column

        Returns:
            Dictionary with metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        ts_test = test_data[target_col].copy()
        ts_test.index = pd.to_datetime(test_data.index)
        ts_test = ts_test.sort_index()

        # Forecast for test period
        steps = len(ts_test)
        forecast = self.fitted_model.forecast(steps=steps)

        # Calculate metrics
        mae = mean_absolute_error(ts_test, forecast)
        rmse = np.sqrt(mean_squared_error(ts_test, forecast))
        mape = np.mean(np.abs((ts_test - forecast) / ts_test)) * 100

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }

        print(f"\nSARIMA Test Performance:")
        print(f"  MAE: {mae:.3f} kWh")
        print(f"  RMSE: {rmse:.3f} kWh")
        print(f"  MAPE: {mape:.2f}%")

        return metrics

    def save_model(self, filepath: str):
        """Save fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        joblib.dump({
            'fitted_model': self.fitted_model,
            'sarima_order': self.sarima_order,
            'seasonal_order': self.seasonal_order,
            'last_date': self.last_date
        }, filepath)
        print(f"SARIMA model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load fitted model."""
        data = joblib.load(filepath)
        self.fitted_model = data['fitted_model']
        self.sarima_order = data['sarima_order']
        self.seasonal_order = data['seasonal_order']
        self.last_date = data['last_date']
        self.is_fitted = True
        print(f"SARIMA model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import EnergyDataPreprocessor

    print("Testing SARIMA Forecasting Module...\n")

    # Load data
    preprocessor = EnergyDataPreprocessor("data/household_power_consumption.txt")
    hourly, daily, profiles = preprocessor.process_pipeline()

    # Split into train/test
    split_idx = int(len(daily) * 0.8)
    train_daily = daily.iloc[:split_idx]
    test_daily = daily.iloc[split_idx:]

    # Fit SARIMA
    ts_forecaster = TimeSeriesForecaster()
    ts_forecaster.fit(train_daily, auto_params=True)

    # Evaluate
    ts_forecaster.evaluate_on_test(test_daily)

    # Predict future date
    future_date = pd.Timestamp('2025-10-29')
    prediction = ts_forecaster.predict_future(future_date)
    print(f"\nPrediction for {future_date.date()}: {prediction:.2f} kWh")
