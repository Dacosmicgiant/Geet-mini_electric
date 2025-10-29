"""
Anomaly Detection Module for Daily Energy Consumption
Identifies unusual consumption patterns that may skew forecasting models.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Tuple, Dict


class EnergyAnomalyDetector:
    """
    Detects anomalies in daily energy consumption using multiple methods.
    """

    def __init__(self, contamination=0.05):
        """
        Initialize anomaly detector.

        Args:
            contamination: Expected proportion of anomalies (0.01 to 0.1)
        """
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.isolation_forest = None
        self.anomaly_scores = {}

    def detect_statistical_anomalies(self, daily_data: pd.DataFrame,
                                     column: str = 'total_daily_consumption',
                                     method: str = 'iqr',
                                     threshold: float = 3.0) -> pd.Series:
        """
        Detect anomalies using statistical methods.

        Args:
            daily_data: DataFrame with daily consumption data
            column: Column to check for anomalies
            method: 'iqr' (Interquartile Range) or 'zscore'
            threshold: Number of IQRs or standard deviations

        Returns:
            Boolean Series indicating anomalies (True = anomaly)
        """
        values = daily_data[column].values

        if method == 'iqr':
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            anomalies = (values < lower_bound) | (values > upper_bound)

            print(f"IQR Method: Detected {anomalies.sum()} anomalies")
            print(f"  Range: [{lower_bound:.2f}, {upper_bound:.2f}]")

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(values, nan_policy='omit'))
            anomalies = z_scores > threshold

            print(f"Z-Score Method: Detected {anomalies.sum()} anomalies")
            print(f"  Threshold: {threshold} standard deviations")

        return pd.Series(anomalies, index=daily_data.index, name=f'{method}_anomaly')

    def detect_isolation_forest_anomalies(self,
                                          daily_data: pd.DataFrame,
                                          features: list = None) -> pd.Series:
        """
        Detect anomalies using Isolation Forest algorithm.

        Args:
            daily_data: DataFrame with daily data
            features: List of feature columns to use (None = use all numeric)

        Returns:
            Boolean Series indicating anomalies (True = anomaly)
        """
        if features is None:
            # Use relevant features for anomaly detection
            features = [
                'total_daily_consumption',
                'avg_hourly_consumption',
                'consumption_volatility',
                'peak_consumption',
                'min_consumption'
            ]

        # Ensure all features exist
        features = [f for f in features if f in daily_data.columns]

        X = daily_data[features].copy()

        # Handle any remaining NaN values
        X = X.fillna(X.mean())

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Fit Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )

        predictions = self.isolation_forest.fit_predict(X_scaled)
        anomaly_scores = self.isolation_forest.score_samples(X_scaled)

        # -1 means anomaly, 1 means normal
        anomalies = predictions == -1

        print(f"Isolation Forest: Detected {anomalies.sum()} anomalies")
        print(f"  Contamination parameter: {self.contamination}")

        self.anomaly_scores['isolation_forest'] = anomaly_scores

        return pd.Series(anomalies, index=daily_data.index, name='isolation_forest_anomaly')

    def detect_pattern_anomalies(self, daily_profiles: pd.DataFrame,
                                 threshold: float = 2.5) -> pd.Series:
        """
        Detect days with unusual hourly patterns compared to similar days.

        Args:
            daily_profiles: DataFrame with 24-hour profiles (h0-h23 columns)
            threshold: Standard deviations from mean pattern

        Returns:
            Boolean Series indicating pattern anomalies
        """
        print("Detecting pattern-based anomalies...")

        profile_cols = [f'h{i}' for i in range(24)]
        profiles = daily_profiles[profile_cols].values

        # Compare each day to the average pattern of similar days
        # (same day of week)
        anomalies = []

        for idx, row in daily_profiles.iterrows():
            day_of_week = row['day_of_week']

            # Get all similar days (same day of week)
            similar_days = daily_profiles[
                (daily_profiles['day_of_week'] == day_of_week) &
                (daily_profiles.index != idx)
            ]

            if len(similar_days) < 3:
                # Not enough data to compare
                anomalies.append(False)
                continue

            # Mean pattern for this day of week
            mean_pattern = similar_days[profile_cols].mean().values
            std_pattern = similar_days[profile_cols].std().values + 1e-6

            # Calculate normalized distance
            current_pattern = row[profile_cols].values
            distances = np.abs(current_pattern - mean_pattern) / std_pattern
            max_deviation = distances.max()

            # Flag as anomaly if maximum deviation exceeds threshold
            is_anomaly = max_deviation > threshold
            anomalies.append(is_anomaly)

        anomalies = np.array(anomalies)
        print(f"Pattern Method: Detected {anomalies.sum()} anomalies")

        return pd.Series(anomalies, index=daily_profiles.index, name='pattern_anomaly')

    def detect_all_anomalies(self, daily_data: pd.DataFrame,
                            daily_profiles: pd.DataFrame,
                            consensus_threshold: int = 2) -> Tuple[pd.DataFrame, Dict]:
        """
        Run all anomaly detection methods and combine results.

        Args:
            daily_data: DataFrame with daily summary statistics
            daily_profiles: DataFrame with 24-hour profiles
            consensus_threshold: Number of methods that must agree (1-4)

        Returns:
            Tuple of (daily_data with anomaly flags, statistics dict)
        """
        print("\n" + "="*60)
        print("Running Anomaly Detection")
        print("="*60 + "\n")

        results = daily_data.copy()

        # Method 1: IQR on total consumption
        iqr_anomalies = self.detect_statistical_anomalies(
            daily_data, 'total_daily_consumption', method='iqr', threshold=2.5
        )
        results['anomaly_iqr'] = iqr_anomalies

        # Method 2: Z-score on total consumption
        zscore_anomalies = self.detect_statistical_anomalies(
            daily_data, 'total_daily_consumption', method='zscore', threshold=3.0
        )
        results['anomaly_zscore'] = zscore_anomalies

        # Method 3: Isolation Forest on multiple features
        if_anomalies = self.detect_isolation_forest_anomalies(daily_data)
        results['anomaly_isolation_forest'] = if_anomalies

        # Method 4: Pattern-based anomalies
        # Align with daily_data index
        pattern_anomalies = self.detect_pattern_anomalies(daily_profiles)

        # Ensure indices match
        results['anomaly_pattern'] = False
        for idx in pattern_anomalies.index:
            if idx in results.index:
                results.loc[idx, 'anomaly_pattern'] = pattern_anomalies[idx]

        # Consensus: Count how many methods flagged each day
        anomaly_cols = ['anomaly_iqr', 'anomaly_zscore',
                       'anomaly_isolation_forest', 'anomaly_pattern']
        results['anomaly_count'] = results[anomaly_cols].sum(axis=1)
        results['is_anomaly'] = results['anomaly_count'] >= consensus_threshold

        # Statistics
        stats_dict = {
            'total_days': len(results),
            'anomalies_detected': results['is_anomaly'].sum(),
            'anomaly_percentage': (results['is_anomaly'].sum() / len(results)) * 100,
            'by_method': {
                'iqr': iqr_anomalies.sum(),
                'zscore': zscore_anomalies.sum(),
                'isolation_forest': if_anomalies.sum(),
                'pattern': pattern_anomalies.sum()
            },
            'consensus_threshold': consensus_threshold
        }

        print("\n" + "="*60)
        print("Anomaly Detection Summary")
        print("="*60)
        print(f"Total days analyzed: {stats_dict['total_days']}")
        print(f"Anomalies detected: {stats_dict['anomalies_detected']} "
              f"({stats_dict['anomaly_percentage']:.2f}%)")
        print(f"\nBy method:")
        for method, count in stats_dict['by_method'].items():
            print(f"  {method:20s}: {count:4d}")
        print(f"\nConsensus threshold: {consensus_threshold} methods must agree")

        return results, stats_dict

    def get_clean_data(self, daily_data: pd.DataFrame,
                      remove_anomalies: bool = True) -> pd.DataFrame:
        """
        Return dataset with anomalies either removed or flagged.

        Args:
            daily_data: DataFrame with anomaly flags (must have 'is_anomaly' column)
            remove_anomalies: If True, remove anomalies; if False, just return flagged data

        Returns:
            Clean dataset
        """
        if 'is_anomaly' not in daily_data.columns:
            raise ValueError("Must run detect_all_anomalies() first")

        if remove_anomalies:
            clean_data = daily_data[~daily_data['is_anomaly']].copy()
            print(f"Removed {daily_data['is_anomaly'].sum()} anomalous days")
            print(f"Clean dataset: {len(clean_data)} days")
            return clean_data
        else:
            return daily_data


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import EnergyDataPreprocessor

    print("Testing Anomaly Detection Module...\n")

    preprocessor = EnergyDataPreprocessor("data/household_power_consumption.txt")
    hourly, daily, profiles = preprocessor.process_pipeline()

    detector = EnergyAnomalyDetector(contamination=0.05)
    daily_with_anomalies, stats = detector.detect_all_anomalies(
        daily, profiles, consensus_threshold=2
    )

    clean_daily = detector.get_clean_data(daily_with_anomalies, remove_anomalies=True)

    print("\nExample anomalous days:")
    print(daily_with_anomalies[daily_with_anomalies['is_anomaly']][
        ['total_daily_consumption', 'anomaly_count']
    ].head(10))
