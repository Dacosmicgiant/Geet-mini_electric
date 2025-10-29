"""
Visualization Module for Energy Consumption Analysis
Creates comprehensive plots for patterns, clusters, and forecasts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 10


class EnergyVisualizer:
    """
    Creates visualizations for energy consumption analysis.
    """

    def __init__(self, save_dir: str = 'visualizations'):
        """
        Initialize visualizer.

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir

    def plot_daily_consumption_trend(self, daily_data: pd.DataFrame,
                                    anomalies: pd.Series = None,
                                    save_filename: str = None):
        """
        Plot daily consumption over time with anomalies highlighted.

        Args:
            daily_data: DataFrame with daily consumption
            anomalies: Boolean series indicating anomalies
            save_filename: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(16, 6))

        # Plot normal days
        if anomalies is not None:
            normal_data = daily_data[~anomalies]
            anomaly_data = daily_data[anomalies]

            ax.plot(normal_data.index, normal_data['total_daily_consumption'],
                   color='steelblue', linewidth=1, label='Normal', alpha=0.7)

            ax.scatter(anomaly_data.index, anomaly_data['total_daily_consumption'],
                      color='red', s=30, label='Anomaly', zorder=5, alpha=0.8)
        else:
            ax.plot(daily_data.index, daily_data['total_daily_consumption'],
                   color='steelblue', linewidth=1, label='Daily Consumption')

        # Add rolling average
        rolling_avg = daily_data['total_daily_consumption'].rolling(window=30).mean()
        ax.plot(daily_data.index, rolling_avg,
               color='orange', linewidth=2, label='30-day Average', alpha=0.8)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Daily Consumption (kWh)', fontsize=12)
        ax.set_title('Daily Energy Consumption Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_filename:
            plt.savefig(f"{self.save_dir}/{save_filename}", dpi=300, bbox_inches='tight')
            print(f"Saved: {save_filename}")

        plt.show()

    def plot_hourly_patterns(self, hourly_data: pd.DataFrame,
                            save_filename: str = None):
        """
        Plot average hourly consumption patterns.

        Args:
            hourly_data: DataFrame with hourly data
            save_filename: Filename to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Average by hour of day
        hourly_avg = hourly_data.groupby('hour')['Global_active_power'].mean()
        axes[0, 0].plot(hourly_avg.index, hourly_avg.values,
                       marker='o', color='steelblue', linewidth=2)
        axes[0, 0].fill_between(hourly_avg.index, hourly_avg.values,
                               alpha=0.3, color='steelblue')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Average Consumption (kW)')
        axes[0, 0].set_title('Average Consumption by Hour', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticks(range(0, 24, 2))

        # Weekday vs Weekend
        weekday_avg = hourly_data[hourly_data['is_weekend'] == 0].groupby('hour')['Global_active_power'].mean()
        weekend_avg = hourly_data[hourly_data['is_weekend'] == 1].groupby('hour')['Global_active_power'].mean()

        axes[0, 1].plot(weekday_avg.index, weekday_avg.values,
                       marker='o', label='Weekday', linewidth=2)
        axes[0, 1].plot(weekend_avg.index, weekend_avg.values,
                       marker='s', label='Weekend', linewidth=2)
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Average Consumption (kW)')
        axes[0, 1].set_title('Weekday vs Weekend Patterns', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xticks(range(0, 24, 2))

        # By day of week
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dow_avg = hourly_data.groupby('day_of_week')['Global_active_power'].mean()

        axes[1, 0].bar(range(7), dow_avg.values, color='steelblue', alpha=0.7)
        axes[1, 0].set_xlabel('Day of Week')
        axes[1, 0].set_ylabel('Average Consumption (kW)')
        axes[1, 0].set_title('Average Consumption by Day of Week', fontweight='bold')
        axes[1, 0].set_xticks(range(7))
        axes[1, 0].set_xticklabels(dow_names)
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # By month
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_avg = hourly_data.groupby('month')['Global_active_power'].mean()

        axes[1, 1].bar(month_avg.index, month_avg.values, color='orange', alpha=0.7)
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Average Consumption (kW)')
        axes[1, 1].set_title('Average Consumption by Month', fontweight='bold')
        axes[1, 1].set_xticks(range(1, 13))
        axes[1, 1].set_xticklabels(month_names, rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_filename:
            plt.savefig(f"{self.save_dir}/{save_filename}", dpi=300, bbox_inches='tight')
            print(f"Saved: {save_filename}")

        plt.show()

    def plot_cluster_profiles(self, profiles_clustered: pd.DataFrame,
                             n_clusters: int = None,
                             save_filename: str = None):
        """
        Plot typical consumption profiles for each cluster.

        Args:
            profiles_clustered: DataFrame with cluster assignments
            n_clusters: Number of clusters to plot (None = all)
            save_filename: Filename to save plot
        """
        profile_cols = [f'h{i}' for i in range(24)]
        unique_clusters = sorted(profiles_clustered['cluster'].unique())

        if n_clusters:
            unique_clusters = unique_clusters[:n_clusters]

        n_clusters_plot = len(unique_clusters)
        n_cols = min(3, n_clusters_plot)
        n_rows = (n_clusters_plot + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))

        if n_clusters_plot == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, cluster_id in enumerate(unique_clusters):
            cluster_data = profiles_clustered[profiles_clustered['cluster'] == cluster_id]

            # Mean profile
            mean_profile = cluster_data[profile_cols].mean()
            std_profile = cluster_data[profile_cols].std()

            hours = range(24)

            axes[idx].plot(hours, mean_profile.values,
                          linewidth=2, label='Mean', color='steelblue')
            axes[idx].fill_between(hours,
                                  mean_profile.values - std_profile.values,
                                  mean_profile.values + std_profile.values,
                                  alpha=0.3, color='steelblue', label='±1 Std Dev')

            # Sample profiles
            sample_profiles = cluster_data.sample(min(5, len(cluster_data)))
            for _, sample in sample_profiles.iterrows():
                axes[idx].plot(hours, sample[profile_cols].values,
                              alpha=0.2, color='gray', linewidth=1)

            axes[idx].set_xlabel('Hour of Day')
            axes[idx].set_ylabel('Consumption (kW)')
            axes[idx].set_title(f'Cluster {cluster_id} ({len(cluster_data)} days)',
                               fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xticks(range(0, 24, 3))
            axes[idx].legend(loc='best', fontsize=8)

        # Hide unused subplots
        for idx in range(n_clusters_plot, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_filename:
            plt.savefig(f"{self.save_dir}/{save_filename}", dpi=300, bbox_inches='tight')
            print(f"Saved: {save_filename}")

        plt.show()

    def plot_forecast_results(self, y_true: pd.Series, y_pred: np.ndarray,
                             title: str = "Forecast Results",
                             save_filename: str = None):
        """
        Plot actual vs predicted values.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            title: Plot title
            save_filename: Filename to save plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # Time series plot
        axes[0].plot(y_true.index, y_true.values,
                    label='Actual', color='steelblue', linewidth=2, alpha=0.7)
        axes[0].plot(y_true.index, y_pred,
                    label='Predicted', color='orange', linewidth=2, alpha=0.7)

        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Daily Consumption (kWh)')
        axes[0].set_title(f'{title} - Time Series', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Scatter plot
        axes[1].scatter(y_true.values, y_pred, alpha=0.5, s=30)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1].plot([min_val, max_val], [min_val, max_val],
                    'r--', linewidth=2, label='Perfect Prediction')

        axes[1].set_xlabel('Actual Consumption (kWh)')
        axes[1].set_ylabel('Predicted Consumption (kWh)')
        axes[1].set_title(f'{title} - Scatter Plot', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Add metrics
        mae = np.mean(np.abs(y_true.values - y_pred))
        rmse = np.sqrt(np.mean((y_true.values - y_pred) ** 2))
        mape = np.mean(np.abs((y_true.values - y_pred) / y_true.values)) * 100

        metrics_text = f'MAE: {mae:.2f} kWh\nRMSE: {rmse:.2f} kWh\nMAPE: {mape:.2f}%'
        axes[1].text(0.05, 0.95, metrics_text,
                    transform=axes[1].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=11)

        plt.tight_layout()

        if save_filename:
            plt.savefig(f"{self.save_dir}/{save_filename}", dpi=300, bbox_inches='tight')
            print(f"Saved: {save_filename}")

        plt.show()

    def plot_feature_importance(self, feature_importance: pd.DataFrame,
                               top_n: int = 15,
                               save_filename: str = None):
        """
        Plot feature importance.

        Args:
            feature_importance: DataFrame with features and importance scores
            top_n: Number of top features to display
            save_filename: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        top_features = feature_importance.head(top_n)

        ax.barh(range(len(top_features)), top_features['importance'].values,
               color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Most Important Features', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_filename:
            plt.savefig(f"{self.save_dir}/{save_filename}", dpi=300, bbox_inches='tight')
            print(f"Saved: {save_filename}")

        plt.show()

    def plot_anomaly_detection_summary(self, daily_data: pd.DataFrame,
                                      save_filename: str = None):
        """
        Plot anomaly detection summary.

        Args:
            daily_data: DataFrame with anomaly flags
            save_filename: Filename to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Time series with anomalies
        normal = daily_data[~daily_data['is_anomaly']]
        anomalies = daily_data[daily_data['is_anomaly']]

        axes[0, 0].plot(normal.index, normal['total_daily_consumption'],
                       color='steelblue', linewidth=1, alpha=0.7, label='Normal')
        axes[0, 0].scatter(anomalies.index, anomalies['total_daily_consumption'],
                          color='red', s=30, label='Anomaly', zorder=5)
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Daily Consumption (kWh)')
        axes[0, 0].set_title('Anomalies Detected Over Time', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Distribution comparison
        axes[0, 1].hist(normal['total_daily_consumption'], bins=50,
                       alpha=0.7, label='Normal', color='steelblue', density=True)
        axes[0, 1].hist(anomalies['total_daily_consumption'], bins=20,
                       alpha=0.7, label='Anomaly', color='red', density=True)
        axes[0, 1].set_xlabel('Daily Consumption (kWh)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Distribution Comparison', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Anomaly count by method
        method_counts = {
            'IQR': daily_data['anomaly_iqr'].sum(),
            'Z-Score': daily_data['anomaly_zscore'].sum(),
            'Isolation\nForest': daily_data['anomaly_isolation_forest'].sum(),
            'Pattern': daily_data['anomaly_pattern'].sum()
        }

        axes[1, 0].bar(method_counts.keys(), method_counts.values(),
                      color='coral', alpha=0.7)
        axes[1, 0].set_ylabel('Number of Anomalies Detected')
        axes[1, 0].set_title('Anomalies by Detection Method', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Consensus distribution
        consensus_counts = daily_data['anomaly_count'].value_counts().sort_index()
        axes[1, 1].bar(consensus_counts.index, consensus_counts.values,
                      color='steelblue', alpha=0.7)
        axes[1, 1].set_xlabel('Number of Methods Agreeing')
        axes[1, 1].set_ylabel('Number of Days')
        axes[1, 1].set_title('Anomaly Consensus Distribution', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_filename:
            plt.savefig(f"{self.save_dir}/{save_filename}", dpi=300, bbox_inches='tight')
            print(f"Saved: {save_filename}")

        plt.show()

    def plot_model_evaluation(self, y_true: pd.Series, y_pred: np.ndarray,
                             model_name: str = "Model",
                             save_filename: str = None):
        """
        Comprehensive model evaluation visualization.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model (for title)
            save_filename: Filename to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Residual plot
        residuals = y_true.values - y_pred
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted Values (kWh)')
        axes[0, 0].set_ylabel('Residuals (kWh)')
        axes[0, 0].set_title(f'{model_name} - Residual Plot', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Error distribution
        axes[0, 1].hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Prediction Error (kWh)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'{model_name} - Error Distribution', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # Add mean and std to error distribution
        mean_error = residuals.mean()
        std_error = residuals.std()
        axes[0, 1].text(0.05, 0.95, f'Mean: {mean_error:.3f}\nStd: {std_error:.3f}',
                       transform=axes[0, 1].transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 3. Actual vs Predicted over time
        axes[1, 0].plot(y_true.index, y_true.values, label='Actual',
                       color='steelblue', linewidth=1.5, alpha=0.7)
        axes[1, 0].plot(y_true.index, y_pred, label='Predicted',
                       color='orange', linewidth=1.5, alpha=0.7)
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Daily Consumption (kWh)')
        axes[1, 0].set_title(f'{model_name} - Predictions Over Time', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Error metrics summary
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true.values - y_pred) / y_true.values)) * 100

        # Calculate error percentage by bins
        error_pcts = np.abs((y_true.values - y_pred) / y_true.values) * 100
        bins = [0, 5, 10, 15, 20, 100]
        labels = ['0-5%', '5-10%', '10-15%', '15-20%', '>20%']
        error_categories = pd.cut(error_pcts, bins=bins, labels=labels)
        error_counts = error_categories.value_counts().sort_index()

        axes[1, 1].bar(range(len(error_counts)), error_counts.values,
                      color='steelblue', alpha=0.7)
        axes[1, 1].set_xticks(range(len(error_counts)))
        axes[1, 1].set_xticklabels(error_counts.index, rotation=45)
        axes[1, 1].set_ylabel('Number of Predictions')
        axes[1, 1].set_title(f'{model_name} - Error Distribution by %', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        # Add metrics text
        metrics_text = f'MAE: {mae:.3f} kWh\nRMSE: {rmse:.3f} kWh\nR²: {r2:.3f}\nMAPE: {mape:.2f}%'
        axes[1, 1].text(0.98, 0.98, metrics_text,
                       transform=axes[1, 1].transAxes,
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
                       fontsize=11)

        plt.tight_layout()

        if save_filename:
            plt.savefig(f"{self.save_dir}/{save_filename}", dpi=300, bbox_inches='tight')
            print(f"Saved: {save_filename}")

        plt.show()

    def plot_sarima_forecast(self, historical_data: pd.Series,
                            forecast: pd.Series,
                            confidence_intervals: tuple = None,
                            save_filename: str = None):
        """
        Plot SARIMA forecast with confidence intervals.

        Args:
            historical_data: Historical time series
            forecast: Forecasted values
            confidence_intervals: Tuple of (lower, upper) confidence bounds
            save_filename: Filename to save plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # 1. Forecast plot
        axes[0].plot(historical_data.index, historical_data.values,
                    label='Historical', color='steelblue', linewidth=2)
        axes[0].plot(forecast.index, forecast.values,
                    label='Forecast', color='orange', linewidth=2)

        if confidence_intervals:
            lower, upper = confidence_intervals
            axes[0].fill_between(forecast.index, lower, upper,
                                alpha=0.3, color='orange', label='95% CI')

        axes[0].axvline(x=historical_data.index[-1], color='red',
                       linestyle='--', linewidth=2, label='Forecast Start')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Daily Consumption (kWh)')
        axes[0].set_title('SARIMA Forecast', fontweight='bold', fontsize=14)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)

        # 2. Forecast zoom (last 60 days of history + forecast)
        last_60_hist = historical_data.iloc[-60:]
        axes[1].plot(last_60_hist.index, last_60_hist.values,
                    label='Recent History', color='steelblue', linewidth=2)
        axes[1].plot(forecast.index, forecast.values,
                    label='Forecast', color='orange', linewidth=2, marker='o', markersize=4)

        if confidence_intervals:
            lower, upper = confidence_intervals
            axes[1].fill_between(forecast.index, lower, upper,
                                alpha=0.3, color='orange')

        axes[1].axvline(x=historical_data.index[-1], color='red',
                       linestyle='--', linewidth=2, label='Forecast Start')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Daily Consumption (kWh)')
        axes[1].set_title('SARIMA Forecast (Zoomed)', fontweight='bold', fontsize=14)
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_filename:
            plt.savefig(f"{self.save_dir}/{save_filename}", dpi=300, bbox_inches='tight')
            print(f"Saved: {save_filename}")

        plt.show()

    def plot_model_comparison(self, rf_metrics: dict, sarima_metrics: dict = None,
                             save_filename: str = None):
        """
        Compare Random Forest and SARIMA model performance.

        Args:
            rf_metrics: Dictionary with Random Forest metrics
            sarima_metrics: Dictionary with SARIMA metrics (optional)
            save_filename: Filename to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Prepare data
        metrics_names = ['MAE\n(kWh)', 'RMSE\n(kWh)', 'MAPE\n(%)']
        rf_values = [rf_metrics['mae'], rf_metrics['rmse'], rf_metrics['mape']]

        if sarima_metrics:
            sarima_values = [sarima_metrics['mae'], sarima_metrics['rmse'],
                           sarima_metrics['mape']]

            # 1. Side-by-side comparison
            x = np.arange(len(metrics_names))
            width = 0.35

            axes[0].bar(x - width/2, rf_values, width, label='Random Forest',
                       color='steelblue', alpha=0.7)
            axes[0].bar(x + width/2, sarima_values, width, label='SARIMA',
                       color='orange', alpha=0.7)

            axes[0].set_ylabel('Error Magnitude')
            axes[0].set_title('Model Performance Comparison', fontweight='bold')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(metrics_names)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, axis='y')

            # 2. R² comparison (if available)
            if 'r2' in rf_metrics and 'r2' in sarima_metrics:
                models = ['Random Forest', 'SARIMA']
                r2_values = [rf_metrics['r2'], sarima_metrics.get('r2', 0)]

                axes[1].bar(models, r2_values, color=['steelblue', 'orange'], alpha=0.7)
                axes[1].set_ylabel('R² Score')
                axes[1].set_title('Model Accuracy (R²)', fontweight='bold')
                axes[1].set_ylim([0, 1.0])
                axes[1].grid(True, alpha=0.3, axis='y')

                # Add value labels on bars
                for i, v in enumerate(r2_values):
                    axes[1].text(i, v + 0.02, f'{v:.3f}',
                               ha='center', fontweight='bold')
            else:
                # Just show RF metrics breakdown
                axes[1].bar(metrics_names, rf_values, color='steelblue', alpha=0.7)
                axes[1].set_ylabel('Error Magnitude')
                axes[1].set_title('Random Forest Metrics Breakdown', fontweight='bold')
                axes[1].grid(True, alpha=0.3, axis='y')
        else:
            # Only Random Forest available
            axes[0].bar(metrics_names, rf_values, color='steelblue', alpha=0.7)
            axes[0].set_ylabel('Error Magnitude')
            axes[0].set_title('Random Forest Performance', fontweight='bold')
            axes[0].grid(True, alpha=0.3, axis='y')

            # R² if available
            if 'r2' in rf_metrics:
                axes[1].bar(['R² Score'], [rf_metrics['r2']], color='steelblue', alpha=0.7)
                axes[1].set_ylim([0, 1.0])
                axes[1].set_title('Random Forest Accuracy', fontweight='bold')
                axes[1].grid(True, alpha=0.3, axis='y')
                axes[1].text(0, rf_metrics['r2'] + 0.02, f"{rf_metrics['r2']:.3f}",
                           ha='center', fontweight='bold')
            else:
                axes[1].axis('off')

        plt.tight_layout()

        if save_filename:
            plt.savefig(f"{self.save_dir}/{save_filename}", dpi=300, bbox_inches='tight')
            print(f"Saved: {save_filename}")

        plt.show()


if __name__ == "__main__":
    print("Visualization module loaded successfully.")
