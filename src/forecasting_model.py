"""
Energy Consumption Forecasting Model
Predicts daily consumption using profile-based features and time series analysis.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Dict, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')


class EnergyForecaster:
    """
    Forecasts daily energy consumption using multiple features and ensemble methods.
    """

    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize forecaster.

        Args:
            model_type: 'random_forest' or 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.feature_names = []

        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )

    def create_features(self, daily_data: pd.DataFrame,
                       profiles_clustered: pd.DataFrame = None) -> pd.DataFrame:
        """
        Engineer features for forecasting.

        Args:
            daily_data: DataFrame with daily consumption and metadata
            profiles_clustered: DataFrame with cluster assignments (optional)

        Returns:
            DataFrame with engineered features
        """
        df = daily_data.copy()

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Sort by date
        df = df.sort_index()

        # ============ Time-based features ============
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['week_of_year'] = df.index.isocalendar().week
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)

        # Season (0=Winter, 1=Spring, 2=Summer, 3=Fall)
        df['season'] = (df['month'] % 12 // 3)

        # Cyclical encoding for periodic features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # ============ Lag features (historical consumption) ============
        target_col = 'total_daily_consumption'

        # Previous days
        for lag in [1, 2, 3, 7, 14]:
            df[f'lag_{lag}d'] = df[target_col].shift(lag)

        # Same day last week
        df['same_day_last_week'] = df[target_col].shift(7)

        # Previous week average
        df['last_week_avg'] = df[target_col].rolling(window=7, min_periods=1).mean().shift(1)

        # Previous 2 weeks average
        df['last_2weeks_avg'] = df[target_col].rolling(window=14, min_periods=1).mean().shift(1)

        # Rolling statistics (7-day window)
        df['rolling_7d_mean'] = df[target_col].rolling(window=7, min_periods=1).mean().shift(1)
        df['rolling_7d_std'] = df[target_col].rolling(window=7, min_periods=1).std().shift(1)
        df['rolling_7d_min'] = df[target_col].rolling(window=7, min_periods=1).min().shift(1)
        df['rolling_7d_max'] = df[target_col].rolling(window=7, min_periods=1).max().shift(1)

        # Rolling statistics (30-day window)
        df['rolling_30d_mean'] = df[target_col].rolling(window=30, min_periods=1).mean().shift(1)
        df['rolling_30d_std'] = df[target_col].rolling(window=30, min_periods=1).std().shift(1)

        # ============ Trend features ============
        # Day number (continuous time)
        df['day_number'] = (df.index - df.index[0]).days

        # Recent trend (difference from last week)
        df['trend_7d'] = df[target_col].shift(1) - df[target_col].shift(8)

        # ============ Pattern features ============
        # Ratio to typical day
        df['ratio_to_weekly_avg'] = df[target_col].shift(1) / (df['rolling_7d_mean'] + 1e-6)

        # Day of week average consumption
        dow_avg = df.groupby('day_of_week')[target_col].transform('mean')
        df['dow_typical_consumption'] = dow_avg

        # ============ Cluster features (if available) ============
        if profiles_clustered is not None:
            # Add cluster information
            cluster_map = profiles_clustered['cluster'].to_dict()

            # Convert index to date for matching
            df['date_key'] = df.index.date
            profiles_clustered['date_key'] = profiles_clustered.index.date

            # Merge cluster info
            df = df.merge(
                profiles_clustered[['date_key', 'cluster']],
                on='date_key',
                how='left'
            )

            # Fill missing clusters with most common
            if 'cluster' in df.columns:
                df['cluster'] = df['cluster'].fillna(df['cluster'].mode()[0] if len(df['cluster'].mode()) > 0 else 0)
                df['cluster'] = df['cluster'].astype(int)

                # Cluster-based features
                # Average consumption for each cluster
                cluster_avg = df.groupby('cluster')[target_col].transform('mean')
                df['cluster_avg_consumption'] = cluster_avg

                # Yesterday's cluster
                df['yesterday_cluster'] = df['cluster'].shift(1).fillna(df['cluster'])

            df = df.drop('date_key', axis=1)
            if 'date_key' in profiles_clustered.columns:
                profiles_clustered = profiles_clustered.drop('date_key', axis=1)

        # ============ External features ============
        # Use voltage as proxy for external factors (temperature, etc.)
        if 'Voltage_mean' in df.columns:
            df['voltage_avg'] = df['Voltage_mean']
            df['voltage_7d_avg'] = df['Voltage_mean'].rolling(window=7, min_periods=1).mean().shift(1)

        return df

    def prepare_train_test(self, df_features: pd.DataFrame,
                          target_col: str = 'total_daily_consumption',
                          test_size: float = 0.2) -> Tuple:
        """
        Split data into train and test sets (time series split).

        Args:
            df_features: DataFrame with features
            target_col: Target column name
            test_size: Proportion of data for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Drop rows with NaN in target
        df_clean = df_features.dropna(subset=[target_col]).copy()

        # Separate target
        y = df_clean[target_col]

        # Select feature columns (exclude target and metadata)
        exclude_cols = [
            target_col,
            'avg_hourly_consumption',
            'consumption_volatility',
            'peak_consumption',
            'min_consumption',
            'Global_reactive_power_mean',
            'Global_reactive_power_sum',
            'Voltage_mean',
            'Voltage_std',
            'Global_intensity_mean',
            'Global_intensity_max',
            'Sub_metering_1',
            'Sub_metering_2',
            'Sub_metering_3'
        ]

        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

        # Remove any remaining NaN
        df_clean = df_clean[feature_cols + [target_col]].dropna()

        X = df_clean[feature_cols]
        y = df_clean[target_col]

        # Time series split
        split_idx = int(len(df_clean) * (1 - test_size))

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        self.feature_names = feature_cols

        print(f"Training set: {len(X_train)} days")
        print(f"Test set: {len(X_test)} days")
        print(f"Features: {len(feature_cols)}")

        return X_train, X_test, y_train, y_test

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             use_cross_validation: bool = True) -> Dict:
        """
        Train the forecasting model.

        Args:
            X_train: Training features
            y_train: Training targets
            use_cross_validation: Whether to perform cross-validation

        Returns:
            Dictionary with training metrics
        """
        print(f"\nTraining {self.model_type} model...")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

        # Training metrics
        train_pred = self.model.predict(X_train_scaled)
        train_metrics = {
            'mae': mean_absolute_error(y_train, train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'r2': r2_score(y_train, train_pred)
        }

        print(f"Training MAE: {train_metrics['mae']:.3f} kWh")
        print(f"Training RMSE: {train_metrics['rmse']:.3f} kWh")
        print(f"Training R²: {train_metrics['r2']:.3f}")

        # Cross-validation
        if use_cross_validation:
            print("\nPerforming time series cross-validation...")
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train,
                cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1
            )
            train_metrics['cv_mae'] = -cv_scores.mean()
            train_metrics['cv_mae_std'] = cv_scores.std()
            print(f"CV MAE: {train_metrics['cv_mae']:.3f} ± {train_metrics['cv_mae_std']:.3f} kWh")

        return train_metrics

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model on test set.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with test metrics
        """
        print("\nEvaluating on test set...")

        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)

        metrics = {
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'r2': r2_score(y_test, predictions),
            'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
        }

        print(f"Test MAE: {metrics['mae']:.3f} kWh")
        print(f"Test RMSE: {metrics['rmse']:.3f} kWh")
        print(f"Test R²: {metrics['r2']:.3f}")
        print(f"Test MAPE: {metrics['mape']:.2f}%")

        return metrics, predictions

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for new data.

        Args:
            X: Features for prediction

        Returns:
            Array of predictions
        """
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions

    def predict_future_day(self, target_date: pd.Timestamp,
                          df_features: pd.DataFrame) -> Tuple[float, Dict]:
        """
        Predict consumption for a specific future date.

        Args:
            target_date: Date to predict
            df_features: Full feature DataFrame including historical data

        Returns:
            Tuple of (prediction, feature_dict)
        """
        if target_date not in df_features.index:
            raise ValueError(f"Date {target_date} not in feature DataFrame")

        # Get features for target date
        features = df_features.loc[[target_date], self.feature_names]

        # Make prediction
        prediction = self.predict(features)[0]

        # Return features used
        feature_dict = features.iloc[0].to_dict()

        return prediction, feature_dict

    def get_top_features(self, n: int = 15) -> pd.DataFrame:
        """
        Get top N most important features.

        Args:
            n: Number of features to return

        Returns:
            DataFrame with top features and their importance
        """
        if self.feature_importance is None:
            raise ValueError("Model must be trained first")

        return self.feature_importance.head(n)

    def save_model(self, filepath: str):
        """Save trained model and scaler."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model and scaler."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.feature_importance = data['feature_importance']
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import EnergyDataPreprocessor
    from anomaly_detection import EnergyAnomalyDetector
    from profile_clustering import DailyProfileClusterer

    print("Testing Forecasting Model...\n")

    # Load and preprocess data
    preprocessor = EnergyDataPreprocessor("data/household_power_consumption.txt")
    hourly, daily, profiles = preprocessor.process_pipeline()

    # Detect and remove anomalies
    detector = EnergyAnomalyDetector(contamination=0.05)
    daily_with_anomalies, _ = detector.detect_all_anomalies(daily, profiles)
    daily_clean = detector.get_clean_data(daily_with_anomalies, remove_anomalies=True)

    # Cluster profiles
    clusterer = DailyProfileClusterer()
    profiles_clustered = clusterer.cluster_profiles(profiles)

    # Create features
    forecaster = EnergyForecaster(model_type='random_forest')
    df_features = forecaster.create_features(daily_clean, profiles_clustered)

    # Train-test split
    X_train, X_test, y_train, y_test = forecaster.prepare_train_test(df_features)

    # Train
    train_metrics = forecaster.train(X_train, y_train)

    # Evaluate
    test_metrics, predictions = forecaster.evaluate(X_test, y_test)

    # Top features
    print("\nTop 10 Most Important Features:")
    print(forecaster.get_top_features(10))
