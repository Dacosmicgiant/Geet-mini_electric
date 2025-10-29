"""
Enhanced Prediction Interface - FIXED VERSION
Now supports BOTH historical and future date predictions!

Fixes applied:
- Index preservation when loading from cache
- Better error handling for out-of-range dates
- Fixed recursive forecaster predictions storage
- All command flags work properly

Usage: 
    python predict.py --date 2010-11-20              # Historical (with actual)
    python predict.py --date 2010-12-25              # Near future (works!)
    python predict.py --range 30                     # Next 30 days
    python predict.py --from 2010-12-01 --to 2010-12-31
    python predict.py --train                        # First run
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main import EnergyForecastingSystem
from data_preprocessing import EnergyDataPreprocessor
from anomaly_detection import EnergyAnomalyDetector
from profile_clustering import DailyProfileClusterer
from forecasting_model import EnergyForecaster


def parse_query(query: str) -> dict:
    """Parse natural language query into prediction parameters."""
    query_lower = query.lower()
    target_date = None

    if 'tomorrow' in query_lower:
        target_date = 'latest+1'
    elif 'today' in query_lower:
        target_date = 'latest'
    elif 'yesterday' in query_lower:
        target_date = 'latest-1'
    elif 'next week' in query_lower:
        target_date = 'latest+7'
    elif 'next month' in query_lower:
        target_date = 'latest+30'
    else:
        import re
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{4}',
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
    """Format prediction result for display."""
    print("\n" + "="*70)
    if query:
        print(f"Query: {query}")
        print("="*70)

    date_str = result['date'].strftime('%Y-%m-%d') if hasattr(result['date'], 'strftime') else str(result['date'])
    
    print(f"\nDate: {date_str} ({result['day_of_week']})")
    print(f"Method: {result.get('method', 'Unknown').replace('_', ' ').title()}")
    
    print(f"\n🔮 Forecast: {result['predicted_consumption']:.2f} kWh")

    if result['actual_consumption']:
        print(f"📊 Actual: {result['actual_consumption']:.2f} kWh")
        print(f"❌ Error: {result['error']:.2f} kWh ({(result['error']/result['actual_consumption']*100):.1f}%)")
    
    # Show confidence interval for future predictions
    if 'confidence_interval' in result and result['confidence_interval']:
        ci_lower, ci_upper = result['confidence_interval']
        print(f"\n95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}] kWh")
        if 'uncertainty' in result:
            print(f"Uncertainty: ±{result['uncertainty']:.2f} kWh")
        if 'days_ahead' in result:
            print(f"Days ahead: {result['days_ahead']}")

    if 'top_features' in result and result['top_features']:
        print("\nTop Contributing Features:")
        for feature, value in result['top_features'].items():
            print(f"  {feature}: {value:.4f}")

    print("\n" + "="*70)


def fix_dataframe_index(df, reference_dates):
    """
    Fix DataFrame index to ensure it's DatetimeIndex.
    
    Args:
        df: DataFrame that may have wrong index
        reference_dates: Source of correct dates
        
    Returns:
        DataFrame with corrected DatetimeIndex
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        print("⚠️  Fixing DataFrame index...")
        original_dates = pd.to_datetime(reference_dates)
        
        if len(df) <= len(original_dates):
            df.index = original_dates[-len(df):]
        else:
            df.index = pd.date_range(
                start=original_dates[0], 
                periods=len(df), 
                freq='D'
            )
        
        print(f"✓ Fixed: Index now spans {df.index.min().date()} to {df.index.max().date()}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Predict household energy consumption for ANY date',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Historical dates (with actual comparison)
  python predict.py --date 2010-11-20
  
  # Future dates (with confidence intervals)
  python predict.py --date 2010-12-25
  python predict.py "Predict tomorrow's energy usage"
  python predict.py "What about next week?"
  
  # Date ranges
  python predict.py --range 30                        # Next 30 days
  python predict.py --from 2010-12-01 --to 2010-12-31
  python predict.py --range 30 --export forecast.csv  # Export to CSV
  
  # First run (required)
  python predict.py --train
  python predict.py --date 2010-12-25 --train
        """
    )

    parser.add_argument(
        'query',
        nargs='?',
        default=None,
        help='Natural language query'
    )

    parser.add_argument(
        '--date',
        type=str,
        help='Specific date to predict (YYYY-MM-DD) - works for past AND future!'
    )
    
    parser.add_argument(
        '--range',
        type=int,
        help='Predict next N days (1-365)'
    )
    
    parser.add_argument(
        '--from',
        dest='date_from',
        type=str,
        help='Start date for range prediction (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--to',
        dest='date_to',
        type=str,
        help='End date for range prediction (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--train',
        action='store_true',
        help='Train model first (required on first run)'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='models/energy_forecaster.pkl',
        help='Path to saved model (default: models/energy_forecaster.pkl)'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='data/household_power_consumption.txt',
        help='Path to data file (default: data/household_power_consumption.txt)'
    )
    
    parser.add_argument(
        '--export',
        type=str,
        help='Export predictions to CSV file'
    )

    args = parser.parse_args()

    # Check if model exists
    model_exists = os.path.exists(args.model_path)

    if not model_exists and not args.train:
        print("⚠️  No trained model found. Please run with --train flag first.")
        print("\nExample: python predict.py --train")
        print("\nThis will:")
        print("  • Train the model (takes 2-3 minutes)")
        print("  • Save model to", args.model_path)
        print("  • Create cache for fast predictions")
        sys.exit(1)

    # Initialize system
    print("🚀 Initializing Energy Forecasting System...")
    system = EnergyForecastingSystem(args.data_path)

    # Train or load model
    if args.train or not model_exists:
        print("\n📚 Training model (this may take a few minutes)...")
        system.run_full_pipeline(save_model=True, model_path=args.model_path)
        print("\n✅ Model trained successfully!")

        # Save preprocessed data with FIXED index
        print("\n💾 Saving preprocessed data for faster future predictions...")
        cache_path = args.model_path.replace('.pkl', '_cache.pkl')
        
        # CRITICAL: Fix df_features index before saving to cache
        if not isinstance(system.df_features.index, pd.DatetimeIndex):
            print("⚠️  Fixing df_features index before caching...")
            system.df_features = fix_dataframe_index(
                system.df_features, 
                system.daily_clean.index
            )
        
        import joblib
        joblib.dump({
            'daily_data': system.daily_data,
            'daily_clean': system.daily_clean,
            'daily_profiles': system.daily_profiles,
            'profiles_clustered': system.profiles_clustered,
            'df_features': system.df_features
        }, cache_path)
        print(f"✅ Cache saved to {cache_path}")

    else:
        print("\n📂 Loading existing model...")
        cache_path = args.model_path.replace('.pkl', '_cache.pkl')
        use_cache = os.path.exists(cache_path)

        if use_cache:
            print("⚡ Loading preprocessed data from cache (fast mode)...")
            import joblib
            cache_data = joblib.load(cache_path)
            system.daily_data = cache_data['daily_data']
            system.daily_clean = cache_data['daily_clean']
            system.daily_profiles = cache_data['daily_profiles']
            system.profiles_clustered = cache_data['profiles_clustered']
            system.df_features = cache_data['df_features']
            
            # CRITICAL FIX: Ensure df_features has proper DatetimeIndex
            system.df_features = fix_dataframe_index(
                system.df_features,
                system.daily_clean.index
            )
            
            # Also fix daily_data and daily_clean if needed
            if not isinstance(system.daily_data.index, pd.DatetimeIndex):
                system.daily_data.index = pd.to_datetime(system.daily_data.index)
            if not isinstance(system.daily_clean.index, pd.DatetimeIndex):
                system.daily_clean.index = pd.to_datetime(system.daily_clean.index)
            
            print("✅ Preprocessed data loaded!")
        else:
            print("⚠️  No cache found, running full preprocessing...")
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
            
            # Fix index before saving
            system.df_features = fix_dataframe_index(
                system.df_features,
                system.daily_clean.index
            )

            import joblib
            joblib.dump({
                'daily_data': system.daily_data,
                'daily_clean': system.daily_clean,
                'daily_profiles': system.daily_profiles,
                'profiles_clustered': system.profiles_clustered,
                'df_features': system.df_features
            }, cache_path)
            print(f"✅ Cache saved to {cache_path}")

        system.forecaster = EnergyForecaster(model_type='random_forest')
        system.forecaster.load_model(args.model_path)

        print("✅ Model loaded successfully!")

    # Determine what to predict
    target_date = None
    daily_index = pd.to_datetime(system.daily_data.index)
    last_date = daily_index[-1]

    # Handle range predictions
    if args.range:
        if args.range < 1 or args.range > 365:
            print(f"\n❌ Error: Range must be between 1 and 365 days. You specified: {args.range}")
            sys.exit(1)
            
        start_date = last_date + timedelta(days=1)
        end_date = last_date + timedelta(days=args.range)
        
        print(f"\n📅 Forecasting range: {start_date.date()} to {end_date.date()}")
        print(f"   ({args.range} days ahead)\n")
        
        try:
            forecast_df = system.predict_future_range(start_date, end_date)
            
            print("="*70)
            print(f"{'Date':<12} {'Prediction':<12} {'CI Lower':<12} {'CI Upper':<12}")
            print("="*70)
            
            for idx, row in forecast_df.head(10).iterrows():
                print(f"{idx.date()} {row['prediction']:>10.2f}  {row['ci_lower']:>10.2f}  {row['ci_upper']:>10.2f}")
            
            if len(forecast_df) > 10:
                print(f"... ({len(forecast_df) - 10} more days)")
            
            print("="*70)
            print(f"\n📊 Summary Statistics:")
            print(f"   Average: {forecast_df['prediction'].mean():.2f} kWh")
            print(f"   Minimum: {forecast_df['prediction'].min():.2f} kWh")
            print(f"   Maximum: {forecast_df['prediction'].max():.2f} kWh")
            print(f"   Total: {forecast_df['prediction'].sum():.2f} kWh")
            
            if args.export:
                forecast_df.to_csv(args.export)
                print(f"\n💾 Exported to {args.export}")
        
        except Exception as e:
            print(f"\n❌ Error during range prediction: {e}")
            print(f"\n💡 Note: Future predictions work best for 1-90 days ahead")
            print(f"   Your range: {args.range} days")
            if args.range > 90:
                print(f"   Consider using --range 90 or less for more reliable predictions")
            sys.exit(1)
        
        return
    
    # Handle date range
    if args.date_from and args.date_to:
        start_date = pd.to_datetime(args.date_from)
        end_date = pd.to_datetime(args.date_to)
        
        # Check if range is reasonable
        days_range = (end_date - start_date).days + 1
        if days_range < 1:
            print(f"\n❌ Error: End date must be after start date")
            sys.exit(1)
        if days_range > 365:
            print(f"\n❌ Error: Date range too large ({days_range} days). Maximum: 365 days")
            sys.exit(1)
        
        print(f"\n📅 Forecasting range: {start_date.date()} to {end_date.date()}")
        print(f"   ({days_range} days)\n")
        
        try:
            forecast_df = system.predict_future_range(start_date, end_date)
            
            print("="*70)
            print(f"{'Date':<12} {'Prediction':<12} {'CI Lower':<12} {'CI Upper':<12}")
            print("="*70)
            
            for idx, row in forecast_df.head(20).iterrows():
                print(f"{idx.date()} {row['prediction']:>10.2f}  {row['ci_lower']:>10.2f}  {row['ci_upper']:>10.2f}")
            
            if len(forecast_df) > 20:
                print(f"... ({len(forecast_df) - 20} more days)")
            
            print("="*70)
            print(f"\n📊 Summary: Avg={forecast_df['prediction'].mean():.2f} kWh, "
                  f"Total={forecast_df['prediction'].sum():.2f} kWh")
            
            if args.export:
                forecast_df.to_csv(args.export)
                print(f"\n💾 Exported to {args.export}")
        
        except Exception as e:
            print(f"\n❌ Error during range prediction: {e}")
            sys.exit(1)
        
        return

    # Single date prediction
    if args.date:
        target_date = args.date
    elif args.query:
        parsed = parse_query(args.query)
        target_date = parsed['date']
        query_text = args.query
    else:
        target_date = 'latest'
        query_text = None

    # Handle special keywords
    if target_date == 'latest':
        target_date = daily_index[-10].strftime('%Y-%m-%d')
    elif target_date == 'latest+1':
        target_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    elif target_date == 'latest+7':
        target_date = (last_date + timedelta(days=7)).strftime('%Y-%m-%d')
    elif target_date == 'latest+30':
        target_date = (last_date + timedelta(days=30)).strftime('%Y-%m-%d')
    elif target_date and target_date.startswith('latest-'):
        days_back = int(target_date.split('-')[1])
        target_date = daily_index[-(10 + days_back)].strftime('%Y-%m-%d')

    # Make prediction
    try:
        # Check if date is too far in future
        target_dt = pd.to_datetime(target_date)
        days_ahead = (target_dt - last_date).days
        
        if days_ahead > 365:
            print(f"\n❌ Error: Cannot predict more than 365 days ahead")
            print(f"   Last historical date: {last_date.date()}")
            print(f"   Requested date: {target_dt.date()}")
            print(f"   Days ahead: {days_ahead}")
            print(f"\n💡 Maximum future date: {(last_date + timedelta(days=365)).date()}")
            sys.exit(1)
        
        result = system.predict_day(target_date)
        format_prediction_output(result, query=args.query if args.query else None)
        
        # Show recent examples
        print("\n💡 Recent Example Predictions:")
        print("-" * 70)
        
        # Mix of historical and future
        example_dates = []
        
        # Add some historical
        for i in [15, 20]:
            if len(daily_index) >= i:
                example_dates.append(daily_index[-i])
        
        # Add some future (only if reasonable)
        if days_ahead <= 30:  # Only show future examples if user's query is near-term
            for i in [7, 14, 30]:
                example_dates.append(last_date + timedelta(days=i))
        
        for date in example_dates:
            try:
                result = system.predict_day(date)
                date_display = result['date'].date() if hasattr(result['date'], 'date') else result['date']
                
                print(f"\n{date_display} ({result['day_of_week']}) - {result['method'].upper()}:")
                print(f"  Forecast: {result['predicted_consumption']:.2f} kWh", end='')
                
                if result['actual_consumption']:
                    print(f" | Actual: {result['actual_consumption']:.2f} kWh | "
                          f"Error: {result['error']:.2f} kWh")
                elif 'confidence_interval' in result:
                    ci = result['confidence_interval']
                    print(f" | 95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
                else:
                    print()
            except:
                continue

        print("\n" + "="*70)

    except Exception as e:
        print(f"\n❌ Error making prediction: {e}")
        
        print("\n📅 Available date information:")
        try:
            min_date = pd.to_datetime(system.daily_data.index.min())
            max_date = pd.to_datetime(system.daily_data.index.max())
            print(f"  Historical data: {min_date.date()} to {max_date.date()}")
            print(f"  Future predictions: {max_date.date()} onwards (up to 365 days)")
            print(f"  Maximum future date: {(max_date + timedelta(days=365)).date()}")
        except:
            print(f"  From: {system.daily_data.index.min()}")
            print(f"  To: {system.daily_data.index.max()}")
        
        print("\n💡 Examples:")
        print("  Historical: python predict.py --date 2010-11-20")
        print("  Future: python predict.py --date 2010-12-25")
        print("  Range: python predict.py --range 30")
        print("  Export: python predict.py --range 30 --export forecast.csv")
        
        import traceback
        print("\n🔍 Debug info:")
        print(traceback.format_exc())
        
        sys.exit(1)


if __name__ == "__main__":
    main()