"""
Daily Profile Clustering Module
Groups similar daily consumption patterns to identify typical behaviors.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class DailyProfileClusterer:
    """
    Clusters daily consumption profiles to identify typical patterns.
    """

    def __init__(self, n_clusters: int = None):
        """
        Initialize profile clusterer.

        Args:
            n_clusters: Number of clusters (None = auto-determine)
        """
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        self.pca = None
        self.cluster_centers = None
        self.profile_features = [f'h{i}' for i in range(24)]

    def find_optimal_clusters(self, profiles: pd.DataFrame,
                             max_clusters: int = 10) -> int:
        """
        Find optimal number of clusters using elbow method and silhouette score.

        Args:
            profiles: DataFrame with 24-hour profiles
            max_clusters: Maximum number of clusters to try

        Returns:
            Optimal number of clusters
        """
        print("Finding optimal number of clusters...")

        X = profiles[self.profile_features].values
        X_scaled = self.scaler.fit_transform(X)

        inertias = []
        silhouette_scores = []
        K_range = range(2, min(max_clusters + 1, len(profiles) // 10))

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)

            inertias.append(kmeans.inertia_)

            # Calculate silhouette score
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)

        # Find elbow point (max second derivative)
        if len(inertias) >= 3:
            # Calculate rate of change
            deltas = np.diff(inertias)
            second_deltas = np.diff(deltas)
            elbow_idx = np.argmax(second_deltas) + 2  # +2 because of two diffs

            optimal_k_elbow = list(K_range)[elbow_idx] if elbow_idx < len(K_range) else K_range[len(K_range)//2]
        else:
            optimal_k_elbow = 3

        # Find best silhouette score
        optimal_k_silhouette = list(K_range)[np.argmax(silhouette_scores)]

        # Use average of both methods
        optimal_k = (optimal_k_elbow + optimal_k_silhouette) // 2

        print(f"Optimal clusters by elbow method: {optimal_k_elbow}")
        print(f"Optimal clusters by silhouette: {optimal_k_silhouette}")
        print(f"Selected: {optimal_k} clusters")

        return optimal_k

    def cluster_profiles(self, profiles: pd.DataFrame,
                        n_clusters: int = None) -> pd.DataFrame:
        """
        Cluster daily profiles using K-Means.

        Args:
            profiles: DataFrame with 24-hour profiles
            n_clusters: Number of clusters (None = use auto-determined value)

        Returns:
            profiles DataFrame with cluster assignments
        """
        if n_clusters is None:
            if self.n_clusters is None:
                n_clusters = self.find_optimal_clusters(profiles)
            else:
                n_clusters = self.n_clusters
        else:
            self.n_clusters = n_clusters

        print(f"\nClustering {len(profiles)} daily profiles into {n_clusters} clusters...")

        X = profiles[self.profile_features].values
        X_scaled = self.scaler.fit_transform(X)

        # Fit K-Means
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=20,
            max_iter=500
        )

        cluster_labels = self.kmeans.fit_predict(X_scaled)

        # Add cluster labels to profiles
        profiles_clustered = profiles.copy()
        profiles_clustered['cluster'] = cluster_labels

        # Get cluster centers in original space
        self.cluster_centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)

        print(f"Silhouette Score: {silhouette_avg:.3f}")
        print(f"\nCluster distribution:")
        print(profiles_clustered['cluster'].value_counts().sort_index())

        return profiles_clustered

    def get_cluster_characteristics(self, profiles_clustered: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze characteristics of each cluster.

        Args:
            profiles_clustered: DataFrame with cluster assignments

        Returns:
            DataFrame with cluster characteristics
        """
        print("\nAnalyzing cluster characteristics...")

        cluster_stats = []

        for cluster_id in sorted(profiles_clustered['cluster'].unique()):
            cluster_data = profiles_clustered[profiles_clustered['cluster'] == cluster_id]

            # Calculate statistics
            hourly_values = cluster_data[self.profile_features].values.flatten()

            stats = {
                'cluster': cluster_id,
                'n_days': len(cluster_data),
                'avg_consumption': hourly_values.mean(),
                'peak_hour': cluster_data[self.profile_features].mean().idxmax().replace('h', ''),
                'peak_value': cluster_data[self.profile_features].mean().max(),
                'min_hour': cluster_data[self.profile_features].mean().idxmin().replace('h', ''),
                'min_value': cluster_data[self.profile_features].mean().min(),
                'volatility': hourly_values.std(),
                'pct_weekdays': ((cluster_data['is_weekend'] == 0).sum() / len(cluster_data)) * 100,
                'pct_weekends': ((cluster_data['is_weekend'] == 1).sum() / len(cluster_data)) * 100,
                'dominant_season': cluster_data['season'].mode()[0] if len(cluster_data) > 0 else -1
            }

            cluster_stats.append(stats)

        cluster_summary = pd.DataFrame(cluster_stats)

        print("\nCluster Summary:")
        print(cluster_summary.to_string(index=False))

        return cluster_summary

    def get_typical_profiles(self, profiles_clustered: pd.DataFrame) -> Dict[int, np.ndarray]:
        """
        Get typical 24-hour profile for each cluster.

        Args:
            profiles_clustered: DataFrame with cluster assignments

        Returns:
            Dictionary mapping cluster_id -> 24-hour profile array
        """
        typical_profiles = {}

        for cluster_id in sorted(profiles_clustered['cluster'].unique()):
            cluster_data = profiles_clustered[profiles_clustered['cluster'] == cluster_id]

            # Average profile for this cluster
            typical_profile = cluster_data[self.profile_features].mean().values
            typical_profiles[cluster_id] = typical_profile

        return typical_profiles

    def predict_cluster(self, hourly_profile: np.ndarray) -> int:
        """
        Predict which cluster a new 24-hour profile belongs to.

        Args:
            hourly_profile: Array of 24 hourly consumption values

        Returns:
            Cluster ID
        """
        if self.kmeans is None:
            raise ValueError("Must run cluster_profiles() first")

        profile_scaled = self.scaler.transform(hourly_profile.reshape(1, -1))
        cluster_id = self.kmeans.predict(profile_scaled)[0]

        return cluster_id

    def get_similar_days(self, target_date: pd.Timestamp,
                        profiles_clustered: pd.DataFrame,
                        n_similar: int = 10) -> pd.DataFrame:
        """
        Find days with similar consumption patterns to target date.

        Args:
            target_date: Date to find similar days for
            profiles_clustered: DataFrame with cluster assignments
            n_similar: Number of similar days to return

        Returns:
            DataFrame of similar days sorted by similarity
        """
        if target_date not in profiles_clustered.index:
            raise ValueError(f"Date {target_date} not found in profiles")

        target_profile = profiles_clustered.loc[target_date, self.profile_features].values
        target_cluster = profiles_clustered.loc[target_date, 'cluster']

        # Filter to same cluster and same day of week
        target_dow = profiles_clustered.loc[target_date, 'day_of_week']

        similar_candidates = profiles_clustered[
            (profiles_clustered['cluster'] == target_cluster) &
            (profiles_clustered['day_of_week'] == target_dow) &
            (profiles_clustered.index != target_date)
        ].copy()

        if len(similar_candidates) == 0:
            # Relax day of week constraint
            similar_candidates = profiles_clustered[
                (profiles_clustered['cluster'] == target_cluster) &
                (profiles_clustered.index != target_date)
            ].copy()

        if len(similar_candidates) == 0:
            return pd.DataFrame()

        # Calculate Euclidean distance to target profile
        distances = []
        for idx, row in similar_candidates.iterrows():
            candidate_profile = row[self.profile_features].values
            distance = np.linalg.norm(target_profile - candidate_profile)
            distances.append(distance)

        similar_candidates['similarity_distance'] = distances
        similar_candidates = similar_candidates.sort_values('similarity_distance')

        return similar_candidates.head(n_similar)


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import EnergyDataPreprocessor

    print("Testing Profile Clustering Module...\n")

    preprocessor = EnergyDataPreprocessor("data/household_power_consumption.txt")
    hourly, daily, profiles = preprocessor.process_pipeline()

    clusterer = DailyProfileClusterer()
    profiles_clustered = clusterer.cluster_profiles(profiles)
    cluster_summary = clusterer.get_cluster_characteristics(profiles_clustered)
    typical_profiles = clusterer.get_typical_profiles(profiles_clustered)

    print("\nTypical profiles extracted for each cluster")
    print(f"Total clusters: {len(typical_profiles)}")
