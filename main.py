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

        print("\n[STEP 7] Feature Importance Analysis")
        print("-" * 70)
        print("\nTop 15 Most Important Features:")
        print(self.forecaster.get_top_features(15).to_string(index=False))

        print("\n" + "="*70)
        print("Pipeline Complete!")
        print("="*70)

    def predict_day(self, target_date: str) -> dict:
        """
        Predict energy consumption for a specific date.

        Args:
            target_date: Date string in format 'YYYY-MM-DD'

        Returns:
            Dictionary with prediction and metadata
        """
        if self.forecaster is None or self.df_features is None:
            raise ValueError("Must run pipeline first")

        target_date = pd.to_datetime(target_date)

        # Check if date is in feature set
        if target_date.date() not in [d.date() for d in self.df_features.index]:
            raise ValueError(f"Date {target_date.date()} not in dataset")

        # Find the matching datetime index
        matching_dates = [d for d in self.df_features.index if d.date() == target_date.date()]
        if not matching_dates:
            raise ValueError(f"No matching date found for {target_date.date()}")

        target_date = matching_dates[0]

        # Make prediction
        prediction, features_used = self.forecaster.predict_future_day(
            target_date, self.df_features
        )

        # Get actual value if available
        actual = None
        if target_date in self.daily_data.index:
            actual = self.daily_data.loc[target_date, 'total_daily_consumption']

        result = {
            'date': target_date,
            'predicted_consumption': round(prediction, 2),
            'actual_consumption': round(actual, 2) if actual else None,
            'error': round(abs(prediction - actual), 2) if actual else None,
            'day_of_week': target_date.strftime('%A'),
            'top_features': {k: round(v, 3) for k, v in list(features_used.items())[:5]}
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
    example_dates = [
        system.daily_data.index[-30].strftime('%Y-%m-%d'),  # 30 days ago
        system.daily_data.index[-15].strftime('%Y-%m-%d'),  # 15 days ago
        system.daily_data.index[-7].strftime('%Y-%m-%d'),   # 7 days ago
    ]

    for date_str in example_dates:
        try:
            result = system.predict_day(date_str)
            print(f"\nDate: {result['date'].date()} ({result['day_of_week']})")
            print(f"  Predicted: {result['predicted_consumption']:.2f} kWh")
            if result['actual_consumption']:
                print(f"  Actual: {result['actual_consumption']:.2f} kWh")
                print(f"  Error: {result['error']:.2f} kWh")
        except Exception as e:
            print(f"\nCould not predict for {date_str}: {e}")

    print("\n" + "="*70)
    print("Analysis complete! Check 'visualizations/' for plots.")
    print("="*70)


if __name__ == "__main__":
    main()
