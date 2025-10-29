"""
Data Preprocessing Module for Energy Consumption Analysis
Handles loading, cleaning, and hourly aggregation of household power consumption data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional


class EnergyDataPreprocessor:
    """
    Preprocesses energy consumption data with focus on daily patterns.
    Aggregates minute-level data to hourly resolution.
    """

    def __init__(self, filepath: str):
        """
        Initialize preprocessor with dataset filepath.

        Args:
            filepath: Path to household_power_consumption.txt
        """
        self.filepath = filepath
        self.raw_data = None
        self.hourly_data = None
        self.daily_data = None

    def load_data(self) -> pd.DataFrame:
        """
        Load raw data from file with proper parsing.

        Returns:
            DataFrame with parsed datetime and numeric columns
        """
        print("Loading raw data...")

        # Read the semicolon-separated file
        df = pd.read_csv(
            self.filepath,
            sep=';',
            low_memory=False,
            na_values=['?', '']
        )

        # Parse datetime
        df['datetime'] = pd.to_datetime(
            df['Date'] + ' ' + df['Time'],
            format='%d/%m/%Y %H:%M:%S',
            errors='coerce'
        )

        # Drop original Date and Time columns
        df = df.drop(['Date', 'Time'], axis=1)

        # Convert numeric columns
        numeric_cols = [
            'Global_active_power', 'Global_reactive_power', 'Voltage',
            'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)

        # Set datetime as index
        df = df.set_index('datetime')

        print(f"Loaded {len(df):,} records from {df.index.min()} to {df.index.max()}")
        print(f"Missing values: {df.isnull().sum().sum():,}")

        self.raw_data = df
        return df

    def handle_missing_values(self, method='interpolate') -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            method: 'interpolate', 'forward_fill', or 'drop'

        Returns:
            DataFrame with missing values handled
        """
        df = self.raw_data.copy()

        if method == 'interpolate':
            print("Interpolating missing values...")
            df = df.interpolate(method='time', limit=60)  # Limit to 1 hour gaps
            df = df.fillna(method='ffill', limit=10)  # Fill remaining with forward fill
            df = df.fillna(method='bfill', limit=10)  # Backward fill

        elif method == 'forward_fill':
            df = df.fillna(method='ffill')

        elif method == 'drop':
            df = df.dropna()

        print(f"Remaining missing values: {df.isnull().sum().sum()}")

        self.raw_data = df
        return df

    def aggregate_to_hourly(self) -> pd.DataFrame:
        """
        Aggregate minute-level data to hourly resolution.

        Returns:
            DataFrame with hourly aggregated data
        """
        print("Aggregating to hourly data...")

        df = self.raw_data.copy()

        # Resample to hourly with mean
        hourly = df.resample('H').agg({
            'Global_active_power': 'mean',
            'Global_reactive_power': 'mean',
            'Voltage': 'mean',
            'Global_intensity': 'mean',
            'Sub_metering_1': 'sum',  # Energy meters are summed
            'Sub_metering_2': 'sum',
            'Sub_metering_3': 'sum'
        })

        # Drop any rows that are all NaN (no data for that hour)
        hourly = hourly.dropna(how='all')

        # Add time-based features
        hourly['hour'] = hourly.index.hour
        hourly['day_of_week'] = hourly.index.dayofweek  # 0=Monday, 6=Sunday
        hourly['day_of_month'] = hourly.index.day
        hourly['month'] = hourly.index.month
        hourly['year'] = hourly.index.year
        hourly['is_weekend'] = (hourly['day_of_week'] >= 5).astype(int)

        # Season: 0=Winter, 1=Spring, 2=Summer, 3=Fall
        hourly['season'] = (hourly['month'] % 12 // 3)

        print(f"Generated {len(hourly):,} hourly records")
        print(f"Date range: {hourly.index.min()} to {hourly.index.max()}")

        self.hourly_data = hourly
        return hourly

    def create_daily_summary(self) -> pd.DataFrame:
        """
        Create daily summary statistics from hourly data.

        Returns:
            DataFrame with daily aggregated metrics
        """
        print("Creating daily summary...")

        if self.hourly_data is None:
            raise ValueError("Must run aggregate_to_hourly() first")

        hourly = self.hourly_data.copy()

        # Group by date
        daily = hourly.groupby(hourly.index.date).agg({
            'Global_active_power': ['mean', 'std', 'min', 'max', 'sum'],
            'Global_reactive_power': ['mean', 'sum'],
            'Voltage': ['mean', 'std'],
            'Global_intensity': ['mean', 'max'],
            'Sub_metering_1': 'sum',
            'Sub_metering_2': 'sum',
            'Sub_metering_3': 'sum',
            'day_of_week': 'first',
            'month': 'first',
            'year': 'first',
            'is_weekend': 'first',
            'season': 'first'
        })

        # Flatten column names
        daily.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                        for col in daily.columns.values]

        # Rename for clarity
        daily = daily.rename(columns={
            'Global_active_power_sum': 'total_daily_consumption',
            'Global_active_power_mean': 'avg_hourly_consumption',
            'Global_active_power_std': 'consumption_volatility',
            'Global_active_power_max': 'peak_consumption',
            'Global_active_power_min': 'min_consumption'
        })

        # Convert index to datetime
        daily.index = pd.to_datetime(daily.index)

        print(f"Generated {len(daily)} daily summaries")

        self.daily_data = daily
        return daily

    def get_daily_profiles(self) -> pd.DataFrame:
        """
        Extract 24-hour consumption profiles for each day.
        Each row represents one day with 24 hourly values.

        Returns:
            DataFrame where each row is a day and columns are hours (0-23)
        """
        print("Extracting daily profiles...")

        if self.hourly_data is None:
            raise ValueError("Must run aggregate_to_hourly() first")

        hourly = self.hourly_data[['Global_active_power', 'day_of_week',
                                   'is_weekend', 'month', 'season']].copy()

        # Pivot to get 24-hour profiles
        hourly['date'] = hourly.index.date
        hourly['hour'] = hourly.index.hour

        profiles = hourly.pivot_table(
            index='date',
            columns='hour',
            values='Global_active_power',
            aggfunc='mean'
        )

        # Only keep days with complete 24-hour data
        profiles = profiles.dropna()

        # Rename columns to h0, h1, ..., h23
        profiles.columns = [f'h{i}' for i in range(24)]

        # Add metadata
        metadata = hourly.groupby('date')[['day_of_week', 'is_weekend',
                                           'month', 'season']].first()
        profiles = profiles.join(metadata)

        print(f"Extracted {len(profiles)} complete daily profiles")

        return profiles

    def process_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run complete preprocessing pipeline.

        Returns:
            Tuple of (hourly_data, daily_data, daily_profiles)
        """
        self.load_data()
        self.handle_missing_values()
        hourly = self.aggregate_to_hourly()
        daily = self.create_daily_summary()
        profiles = self.get_daily_profiles()

        print("\n" + "="*60)
        print("Preprocessing Complete!")
        print("="*60)

        return hourly, daily, profiles


if __name__ == "__main__":
    # Example usage
    preprocessor = EnergyDataPreprocessor("data/household_power_consumption.txt")
    hourly, daily, profiles = preprocessor.process_pipeline()

    print("\nSample hourly data:")
    print(hourly.head())
    print("\nSample daily data:")
    print(daily.head())
    print("\nSample daily profile:")
    print(profiles.head(1))
