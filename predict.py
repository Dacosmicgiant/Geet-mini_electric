"""
Simple Prediction Interface
Usage: python predict.py "Predict tomorrow's energy usage for Household 1001"
       python predict.py --date 2007-12-15
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main import EnergyForecastingSystem


def parse_query(query: str) -> dict:
    """
    Parse natural language query into prediction parameters.

    Args:
        query: Natural language query

    Returns:
        Dictionary with parsed parameters
    """
    query_lower = query.lower()

    # Extract date
    target_date = None

    if 'tomorrow' in query_lower:
        # Use a recent date from dataset as "tomorrow"
        target_date = 'latest'
    elif 'today' in query_lower:
        target_date = 'latest'
    elif 'yesterday' in query_lower:
        target_date = 'latest-1'
    else:
        # Try to extract date in various formats
        import re
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
        ]

        for pattern in date_patterns:
            match = re.search(pattern, query)
            if match:
                target_date = match.group()
                break

    return {
        'date': target_date,
        'query': query
    }


def format_prediction_output(result: dict, query: str = None):
    """
    Format prediction result for display.

    Args:
        result: Prediction result dictionary
        query: Original query (optional)
    """
    print("\n" + "="*70)
    if query:
        print(f"Query: {query}")
        print("="*70)

    # Handle date formatting - might be datetime or date object
    if hasattr(result['date'], 'strftime'):
        date_str = result['date'].strftime('%Y-%m-%d')
    else:
        date_str = str(result['date'])

    print(f"\nDate: {date_str} ({result['day_of_week']})")
    print(f"\nForecast: {result['predicted_consumption']:.2f} kWh")

    if result['actual_consumption']:
        print(f"Actual: {result['actual_consumption']:.2f} kWh")
        print(f"Error: {result['error']:.2f} kWh")
        error_pct = (result['error'] / result['actual_consumption']) * 100
        print(f"Error %: {error_pct:.2f}%")

    print("\nTop Contributing Features:")
    for feature, value in result['top_features'].items():
        print(f"  {feature}: {value}")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Predict household energy consumption',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py "Predict tomorrow's energy usage for Household 1001"
  python predict.py --date 2007-12-15
  python predict.py --date 2007-11-20 --train
        """
    )

    parser.add_argument(
        'query',
        nargs='?',
        default=None,
        help='Natural language query (e.g., "Predict tomorrow\'s energy usage")'
    )

    parser.add_argument(
        '--date',
        type=str,
        help='Specific date to predict (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--train',
        action='store_true',
        help='Train model (required on first run)'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='models/energy_forecaster.pkl',
        help='Path to saved model'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='data/household_power_consumption.txt',
        help='Path to data file'
    )

    args = parser.parse_args()

    # Check if model exists
    model_exists = os.path.exists(args.model_path)

    if not model_exists and not args.train:
        print("No trained model found. Please run with --train flag first.")
        print("Example: python predict.py --train")
        sys.exit(1)

    # Initialize system
    print("Initializing Energy Forecasting System...")
    system = EnergyForecastingSystem(args.data_path)

    # Train or load model
    if args.train or not model_exists:
        print("\nTraining model (this may take a few minutes)...")
        system.run_full_pipeline(save_model=True, model_path=args.model_path)
        print("\nModel trained successfully!")
    else:
        print("\nLoading existing model...")
        # Load preprocessed data
        system.preprocessor = EnergyDataPreprocessor(args.data_path)
        system.hourly_data, system.daily_data, system.daily_profiles = \
            system.preprocessor.process_pipeline()

        system.detector = EnergyAnomalyDetector(contamination=0.05)
        system.daily_data, _ = system.detector.detect_all_anomalies(
            system.daily_data, system.daily_profiles, consensus_threshold=2
        )
        system.daily_clean = system.detector.get_clean_data(
            system.daily_data, remove_anomalies=True
        )

        system.clusterer = DailyProfileClusterer()
        system.profiles_clustered = system.clusterer.cluster_profiles(
            system.daily_profiles, n_clusters=5
        )

        system.forecaster = EnergyForecaster(model_type='random_forest')
        system.df_features = system.forecaster.create_features(
            system.daily_clean, system.profiles_clustered
        )
        system.forecaster.load_model(args.model_path)
        print("Model loaded successfully!")

    # Determine target date
    target_date = None

    if args.date:
        target_date = args.date
    elif args.query:
        parsed = parse_query(args.query)
        target_date = parsed['date']
        query_text = args.query
    else:
        # Use latest date as example
        target_date = 'latest'
        query_text = None

    # Handle special date keywords
    # Convert index to datetime for consistent handling
    daily_index = pd.to_datetime(system.daily_data.index)

    if target_date == 'latest':
        target_date = daily_index[-10].strftime('%Y-%m-%d')
    elif target_date and target_date.startswith('latest-'):
        days_back = int(target_date.split('-')[1])
        target_date = daily_index[-(10 + days_back)].strftime('%Y-%m-%d')

    # Make prediction
    try:
        result = system.predict_day(target_date)
        format_prediction_output(result, query=args.query if args.query else None)

        # Show additional example predictions
        print("\nAdditional Example Predictions:")
        print("-" * 70)

        for i in [15, 20, 25]:
            try:
                date_str = daily_index[-i].strftime('%Y-%m-%d')
                result = system.predict_day(date_str)

                # Handle date formatting
                if hasattr(result['date'], 'date'):
                    date_display = result['date'].date()
                else:
                    date_display = result['date']

                print(f"\n{date_display} ({result['day_of_week']}):")
                print(f"  Forecast: {result['predicted_consumption']:.2f} kWh", end='')
                if result['actual_consumption']:
                    print(f" | Actual: {result['actual_consumption']:.2f} kWh | Error: {result['error']:.2f} kWh")
                else:
                    print()
            except:
                continue

        print("\n" + "="*70)

    except Exception as e:
        print(f"\nError making prediction: {e}")
        print("\nAvailable date range:")
        # Safe date formatting
        try:
            min_date = pd.to_datetime(system.daily_data.index.min())
            max_date = pd.to_datetime(system.daily_data.index.max())
            print(f"  From: {min_date.date()}")
            print(f"  To: {max_date.date()}")
        except:
            print(f"  From: {system.daily_data.index.min()}")
            print(f"  To: {system.daily_data.index.max()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
