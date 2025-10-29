"""
Main Energy Consumption Forecasting System
Complete pipeline for analyzing and predicting household energy consumption.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import EnergyDataPreprocessor
from anomaly_detection import EnergyAnomalyDetector
from profile_clustering import DailyProfileClusterer
from forecasting_model import EnergyForecaster
from time_series_forecaster import TimeSeriesForecaster
from visualization import EnergyVisualizer


class EnergyForecastingSystem:
    """
    Complete energy forecasting system with all components integrated.
    """

    def __init__(self, data_filepath: str):
        """
        Initialize the forecasting system.

        Args:
            data_filepath: Path to household_power_consumption.txt
        """
        self.data_filepath = data_filepath
        self.preprocessor = None
        self.detector = None
        self.clusterer = None
        self.forecaster = None
        self.ts_forecaster = None  # SARIMA for future predictions
        self.visualizer = EnergyVisualizer()

        # Data containers
        self.hourly_data = None
        self.daily_data = None
        self.daily_profiles = None
        self.daily_clean = None
        self.profiles_clustered = None
        self.df_features = None

        # Model performance
        self.train_metrics = None
        self.test_metrics = None
        self.predictions = None

        print("="*70)
        print("Energy Consumption Forecasting System")
        print("="*70)

    def run_full_pipeline(self, save_model: bool = True,
                         model_path: str = 'models/energy_forecaster.pkl'):
        """
        Run complete analysis and forecasting pipeline.

        Args:
            save_model: Whether to save trained model
            model_path: Path to save model
        """
        print("\n[STEP 1] Data Preprocessing")
        print("-" * 70)
        self.preprocessor = EnergyDataPreprocessor(self.data_filepath)
        self.hourly_data, self.daily_data, self.daily_profiles = \
            self.preprocessor.process_pipeline()

        print("\n[STEP 2] Anomaly Detection")
        print("-" * 70)
        self.detector = EnergyAnomalyDetector(contamination=0.05)
        self.daily_data, anomaly_stats = self.detector.detect_all_anomalies(
            self.daily_data, self.daily_profiles, consensus_threshold=2
        )
        self.daily_clean = self.detector.get_clean_data(
            self.daily_data, remove_anomalies=True
        )

        print("\n[STEP 3] Daily Profile Clustering")
        print("-" * 70)
        self.clusterer = DailyProfileClusterer()
        self.profiles_clustered = self.clusterer.cluster_profiles(self.daily_profiles)
        cluster_summary = self.clusterer.get_cluster_characteristics(self.profiles_clustered)

        print("\n[STEP 4] Feature Engineering")
        print("-" * 70)
        self.forecaster = EnergyForecaster(model_type='random_forest')
        self.df_features = self.forecaster.create_features(
            self.daily_clean, self.profiles_clustered
        )
        print(f"Created feature set with {len(self.df_features.columns)} features")

        print("\n[STEP 5] Model Training")
        print("-" * 70)
        X_train, X_test, y_train, y_test = self.forecaster.prepare_train_test(
            self.df_features, test_size=0.2
        )

        self.train_metrics = self.forecaster.train(X_train, y_train, use_cross_validation=True)

        print("\n[STEP 6] Model Evaluation")
        print("-" * 70)
        self.test_metrics, self.predictions = self.forecaster.evaluate(X_test, y_test)

        # Save model
        if save_model:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.forecaster.save_model(model_path)

        print("\n[STEP 7] Training SARIMA for Future Predictions")
        print("-" * 70)
        self.ts_forecaster = TimeSeriesForecaster()
        self.ts_forecaster.fit(self.daily_clean, auto_params=False)  # Use default params for speed

        # Save SARIMA model
        if save_model:
            sarima_path = model_path.replace('.pkl', '_sarima.pkl')
            self.ts_forecaster.save_model(sarima_path)

        print("\n[STEP 8] Feature Importance Analysis")
        print("-" * 70)
        print("\nTop 15 Most Important Features:")
        print(self.forecaster.get_top_features(15).to_string(index=False))

        print("\n" + "="*70)
        print("Pipeline Complete!")
        print("="*70)

    def predict_day(self, target_date: str) -> dict:
        """
        Predict energy consumption for any date (historical or future).
        Uses Random Forest for historical dates, SARIMA for future dates.

        Args:
            target_date: Date string in format 'YYYY-MM-DD'

        Returns:
            Dictionary with prediction and metadata
        """
        if self.forecaster is None:
            raise ValueError("Must run pipeline first")

        target_date = pd.to_datetime(target_date)
        target_datetime = target_date

        # Get last available date in dataset
        daily_index = pd.to_datetime(self.daily_data.index)
        last_date = daily_index.max()

        # Determine prediction method
        is_future = target_date > last_date
        prediction_method = 'SARIMA' if is_future else 'Random Forest'

        if is_future:
            # Use SARIMA for future predictions
            if self.ts_forecaster is None or not self.ts_forecaster.is_fitted:
                raise ValueError("SARIMA model not trained. Please run full pipeline first.")

            prediction = self.ts_forecaster.predict_future(target_datetime)
            features_used = {'prediction_method': 'SARIMA (time series)'}
            actual = None

        else:
            # Use Random Forest for historical predictions
            if self.df_features is None:
                raise ValueError("Features not available. Please run full pipeline first.")

            # Convert index to DatetimeIndex for consistent handling
            df_features_index = pd.to_datetime(self.df_features.index)

            # Check if date is in feature set
            if target_date.date() not in [d.date() for d in df_features_index]:
                raise ValueError(f"Date {target_date.date()} not in dataset features. Available range: {df_features_index.min().date()} to {df_features_index.max().date()}")
            else:
                # Find the matching datetime index
                matching_dates = [d for d in df_features_index if d.date() == target_date.date()]
                if not matching_dates:
                    raise ValueError(f"No matching date found for {target_date.date()}")

                # Get the original index value (might be date or datetime)
                target_idx = df_features_index.get_loc(matching_dates[0])
                target_date_key = self.df_features.index[target_idx]

                # Make prediction with Random Forest
                prediction, features_used = self.forecaster.predict_future_day(
                    target_date_key, self.df_features
                )

                # Get actual value if available
                actual = None
                matching_daily = [d for d in daily_index if d.date() == target_datetime.date()]
                if matching_daily:
                    daily_idx = daily_index.get_loc(matching_daily[0])
                    actual = self.daily_data.iloc[daily_idx]['total_daily_consumption']

        # Format features_used for display (handle non-numeric values)
        top_features = {}
        for k, v in list(features_used.items())[:5]:
            if isinstance(v, (int, float)):
                top_features[k] = round(v, 3)
            else:
                top_features[k] = v

        result = {
            'date': target_datetime,
            'predicted_consumption': round(prediction, 2),
            'actual_consumption': round(actual, 2) if actual is not None else None,
            'error': round(abs(prediction - actual), 2) if actual is not None else None,
            'day_of_week': target_datetime.strftime('%A'),
            'prediction_method': prediction_method,
            'top_features': top_features
        }

        return result

    def generate_visualizations(self):
        """
        Generate all visualizations.
        """
        if self.daily_data is None:
            raise ValueError("Must run pipeline first")

        print("\n[Generating Visualizations]")
        print("-" * 70)

        # 1. Daily consumption trend with anomalies
        print("1. Daily consumption trend with anomalies...")
        self.visualizer.plot_daily_consumption_trend(
            self.daily_data,
            anomalies=self.daily_data['is_anomaly'],
            save_filename='01_daily_trend_with_anomalies.png'
        )

        # 2. Hourly patterns
        print("2. Hourly consumption patterns...")
        self.visualizer.plot_hourly_patterns(
            self.hourly_data,
            save_filename='02_hourly_patterns.png'
        )

        # 3. Cluster profiles
        print("3. Daily profile clusters...")
        self.visualizer.plot_cluster_profiles(
            self.profiles_clustered,
            save_filename='03_cluster_profiles.png'
        )

        # 4. Forecast results
        if self.predictions is not None:
            print("4. Forecast results...")
            X_train, X_test, y_train, y_test = self.forecaster.prepare_train_test(
                self.df_features, test_size=0.2
            )
            self.visualizer.plot_forecast_results(
                y_test, self.predictions,
                title="Test Set Predictions",
                save_filename='04_forecast_results.png'
            )

        # 5. Feature importance
        print("5. Feature importance...")
        self.visualizer.plot_feature_importance(
            self.forecaster.feature_importance,
            top_n=15,
            save_filename='05_feature_importance.png'
        )

        # 6. Anomaly detection summary
        print("6. Anomaly detection summary...")
        self.visualizer.plot_anomaly_detection_summary(
            self.daily_data,
            save_filename='06_anomaly_summary.png'
        )

        # 7. Random Forest model evaluation
        if self.predictions is not None:
            print("7. Random Forest model evaluation...")
            X_train, X_test, y_train, y_test = self.forecaster.prepare_train_test(
                self.df_features, test_size=0.2
            )
            self.visualizer.plot_model_evaluation(
                y_test, self.predictions,
                model_name="Random Forest",
                save_filename='07_rf_model_evaluation.png'
            )

        # 8. SARIMA forecast visualization
        if self.ts_forecaster and self.ts_forecaster.is_fitted:
            print("8. SARIMA forecast visualization...")
            # Get historical data
            ts = self.daily_clean['total_daily_consumption'].copy()
            ts.index = pd.to_datetime(self.daily_clean.index)

            # Generate 30-day forecast
            forecast = self.ts_forecaster.fitted_model.forecast(steps=30)
            forecast_dates = pd.date_range(
                start=ts.index[-1] + pd.Timedelta(days=1),
                periods=30,
                freq='D'
            )
            forecast_series = pd.Series(forecast, index=forecast_dates)

            # Get confidence intervals
            forecast_obj = self.ts_forecaster.fitted_model.get_forecast(steps=30)
            conf_int = forecast_obj.conf_int()

            self.visualizer.plot_sarima_forecast(
                ts,
                forecast_series,
                confidence_intervals=(conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values),
                save_filename='08_sarima_forecast.png'
            )

        # 9. Model comparison
        if self.test_metrics:
            print("9. Model comparison...")
            sarima_metrics = None
            if self.ts_forecaster and self.ts_forecaster.is_fitted:
                # Evaluate SARIMA on a test set
                try:
                    split_idx = int(len(self.daily_clean) * 0.8)
                    test_daily = self.daily_clean.iloc[split_idx:]
                    sarima_metrics = self.ts_forecaster.evaluate_on_test(test_daily)
                except:
                    pass

            self.visualizer.plot_model_comparison(
                self.test_metrics,
                sarima_metrics,
                save_filename='09_model_comparison.png'
            )

        print("\nAll visualizations saved to 'visualizations/' directory")

    def print_summary_report(self):
        """
        Print comprehensive summary report.
        """
        print("\n" + "="*70)
        print("SUMMARY REPORT")
        print("="*70)

        if self.daily_data is not None:
            print(f"\nDataset Overview:")
            print(f"  Date range: {self.daily_data.index.min().date()} to {self.daily_data.index.max().date()}")
            print(f"  Total days: {len(self.daily_data)}")
            print(f"  Clean days: {len(self.daily_clean)}")
            print(f"  Anomalies removed: {len(self.daily_data) - len(self.daily_clean)}")

        if self.profiles_clustered is not None:
            print(f"\nClustering:")
            print(f"  Number of clusters: {self.profiles_clustered['cluster'].nunique()}")
            print(f"  Clustered profiles: {len(self.profiles_clustered)}")

        if self.train_metrics is not None:
            print(f"\nTraining Performance:")
            print(f"  MAE: {self.train_metrics['mae']:.3f} kWh")
            print(f"  RMSE: {self.train_metrics['rmse']:.3f} kWh")
            print(f"  R²: {self.train_metrics['r2']:.3f}")
            if 'cv_mae' in self.train_metrics:
                print(f"  Cross-Validation MAE: {self.train_metrics['cv_mae']:.3f} kWh")

        if self.test_metrics is not None:
            print(f"\nTest Performance:")
            print(f"  MAE: {self.test_metrics['mae']:.3f} kWh")
            print(f"  RMSE: {self.test_metrics['rmse']:.3f} kWh")
            print(f"  R²: {self.test_metrics['r2']:.3f}")
            print(f"  MAPE: {self.test_metrics['mape']:.2f}%")

        print("\n" + "="*70)


def main():
    """
    Main execution function with example usage.
    """
    # Initialize system
    system = EnergyForecastingSystem("data/household_power_consumption.txt")

    # Run full pipeline
    system.run_full_pipeline(save_model=True)

    # Generate visualizations
    system.generate_visualizations()

    # Print summary
    system.print_summary_report()

    # Example predictions
    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS")
    print("="*70)

    # Predict some specific dates
    try:
        # Get dates from features (which are cleaned and have lag features)
        features_index = pd.to_datetime(system.df_features.index)

        # Historical predictions using Random Forest
        print("\nHistorical Predictions (Random Forest):")
        print("-" * 70)

        historical_dates = [
            features_index[-30].strftime('%Y-%m-%d'),
            features_index[-15].strftime('%Y-%m-%d'),
            features_index[-5].strftime('%Y-%m-%d'),
        ]

        for date_str in historical_dates:
            try:
                result = system.predict_day(date_str)

                if hasattr(result['date'], 'date'):
                    date_display = result['date'].date()
                else:
                    date_display = result['date']

                print(f"\n{date_display} ({result['day_of_week']}) - {result['prediction_method']}")
                print(f"  Predicted: {result['predicted_consumption']:.2f} kWh", end='')
                if result['actual_consumption']:
                    error_pct = (result['error'] / result['actual_consumption']) * 100
                    print(f" | Actual: {result['actual_consumption']:.2f} kWh | Error: {error_pct:.1f}%")
                else:
                    print()
            except Exception as e:
                print(f"\nError for {date_str}: {e}")

        # Future predictions using SARIMA
        if system.ts_forecaster and system.ts_forecaster.is_fitted:
            print("\n\nFuture Predictions (SARIMA):")
            print("-" * 70)

            last_date = features_index.max()
            future_dates = [
                (last_date + pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
                (last_date + pd.Timedelta(days=30)).strftime('%Y-%m-%d'),
                (last_date + pd.Timedelta(days=365)).strftime('%Y-%m-%d'),
            ]

            for date_str in future_dates:
                try:
                    result = system.predict_day(date_str)

                    if hasattr(result['date'], 'date'):
                        date_display = result['date'].date()
                    else:
                        date_display = result['date']

                    print(f"\n{date_display} ({result['day_of_week']}) - {result['prediction_method']}")
                    print(f"  Predicted: {result['predicted_consumption']:.2f} kWh")
                except Exception as e:
                    print(f"\nError for {date_str}: {e}")

    except Exception as e:
        print(f"\nError generating example predictions: {e}")

    print("\n" + "="*70)
    print("Analysis complete! Check 'visualizations/' for plots.")
    print("="*70)


if __name__ == "__main__":
    main()
