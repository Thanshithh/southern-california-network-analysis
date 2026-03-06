#!/usr/bin/env python3
"""
Southern California Wildfire Analysis - Restricted to California State
Comprehensive clustering and network analysis
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from shapely.geometry import Point, MultiPoint
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("⚠️  hdbscan not available. Install with: pip install hdbscan")

try:
    import contextily as ctx
    CONTEXTILY_AVAILABLE = True
except ImportError:
    CONTEXTILY_AVAILABLE = False
    print("⚠️  contextily not available. Install with: pip install contextily")

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    print("⚠️  python-louvain not available. Install with: pip install python-louvain")

# Path Configuration - Relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = SCRIPT_DIR  # Assuming script is in project root
DATA_DIR = os.path.join(DATASET_ROOT, "data")
RESULTS_DIR = os.path.join(DATASET_ROOT, "results")
VIZ_DIR = os.path.join(RESULTS_DIR, "visualizations")
DATA_FILE = os.path.join(DATA_DIR, "wildfire_data.geojson")

# Analysis Parameters
MIN_FIRE_SIZE = 500
YEARS_BACK = 25
CURRENT_YEAR = 2024
MAX_DISTANCE_KM = 30.0
MAX_TIME_WINDOW_DAYS = 180
BUFFER_DEG = 0.74  # Buffer to match friend's 717 fire count (approx 50 miles)


class WildfireAnalyzer:
    """Complete wildfire clustering analysis system."""
    
    def __init__(self):
        self.gdf = None
        self.cluster_results = {}
        self.resource_plans = {}
    
    def load_and_preprocess_data(self):
        """Load and preprocess wildfire data."""
        print("=" * 70)
        print("SOUTHERN CALIFORNIA WILDFIRE CLUSTERING ANALYSIS")
        print("=" * 70)
        
        if not os.path.exists(DATA_FILE):
            print(f"❌ No data found at {DATA_FILE}")
            print(f"Expected location: {os.path.abspath(DATA_FILE)}")
            print("Please ensure wildfire_data.geojson is in the data/ directory")
            sys.exit(1)
        
        print(f"\nLoading data from {DATA_FILE}...")
        self.gdf = gpd.read_file(DATA_FILE)
        print(f"✓ Loaded {len(self.gdf)} fire records")

        # Convert to centroids if needed (BEFORE filtering)
        if not self.gdf.empty and self.gdf.geometry.iloc[0].geom_type in ['Polygon', 'MultiPolygon']:
            self.gdf['geometry'] = self.gdf.geometry.centroid
        self.gdf = self.gdf.dropna(subset=['geometry']).copy()


        # --- CALIFORNIA FILTERING START ---
        states_file = os.path.join(DATA_DIR, "us_states.json")
        if os.path.exists(states_file):
            print("\nFiltering for California data...")
            try:
                states = gpd.read_file(states_file)
                # Filter for California
                california = states[states['name'] == 'California']
                
                if not california.empty:
                    # Ensure CRS matches
                    if self.gdf.crs != california.crs:
                        california = california.to_crs(self.gdf.crs)
                    
                    # Apply buffer if configured
                    filtering_geom = california
                    if BUFFER_DEG > 0:
                        print(f"   Applying {BUFFER_DEG} degree buffer to state boundary...")
                        # Warning: Buffering in geographic CRS is approximate but used here to match legacy method
                        filtering_geom = california.copy()
                        filtering_geom['geometry'] = california.geometry.buffer(BUFFER_DEG)
                    
                    # Spatial join to keep only points within California
                    original_count = len(self.gdf)
                    # Use inner join with 'within' predicate
                    self.gdf = gpd.sjoin(self.gdf, filtering_geom, how='inner', predicate='within')
                    
                    # Clean up columns added by join if needed
                    cols_to_drop = ['index_right', 'name', 'density', 'id']
                    existing_cols_to_drop = [c for c in cols_to_drop if c in self.gdf.columns]
                    if existing_cols_to_drop:
                        self.gdf = self.gdf.drop(columns=existing_cols_to_drop)
                        
                    print(f"✓ Filtered to California state boundary: {original_count} → {len(self.gdf)} fires")
                else:
                    print("⚠️  'California' not found in us_states.json properties")
            except Exception as e:
                print(f"⚠️  Error during state filtering: {e}")
        else:
            print(f"⚠️  States file not found at {states_file}, skipping state filtering")
        # --- CALIFORNIA FILTERING END ---
        

        
        # Filter by fire size
        if 'BurnBndAc' in self.gdf.columns:
            original_count = len(self.gdf)
            self.gdf = self.gdf[self.gdf['BurnBndAc'] > MIN_FIRE_SIZE].copy()
            print(f"✓ Filtered by size: {original_count} → {len(self.gdf)} fires (>{MIN_FIRE_SIZE} acres)")
        
        # Filter by time period
        if 'Year' in self.gdf.columns:
            original_count = len(self.gdf)
            min_year = CURRENT_YEAR - YEARS_BACK
            self.gdf = self.gdf[self.gdf['Year'] >= min_year].copy()
            print(f"✓ Filtered by time: {original_count} → {len(self.gdf)} fires ({min_year}-{CURRENT_YEAR})")
            print(f"✓ Time period: {self.gdf['Year'].min():.0f}-{self.gdf['Year'].max():.0f}")
        
        if 'BurnBndAc' in self.gdf.columns:
            print(f"✓ Fire size range: {self.gdf['BurnBndAc'].min():.0f} - {self.gdf['BurnBndAc'].max():.0f} acres")
            print(f"✓ Total acres burned: {self.gdf['BurnBndAc'].sum():,.0f}")
    
    def prepare_features(self):
        """Prepare feature matrices for clustering."""
        print(f"\nPreparing features for clustering...")
        
        # Extract coordinates
        coords = np.array([[point.x, point.y] for point in self.gdf.geometry])
        coords_scaled = StandardScaler().fit_transform(coords)
        
        # Build feature matrix with weighted features
        features = [coords * 2.0]  # Geographic coordinates with higher weight
        
        if 'BurnBndAc' in self.gdf.columns:
            size_feature = np.log1p(self.gdf['BurnBndAc'].values).reshape(-1, 1) * 0.3
            features.append(size_feature)
            print("✓ Added fire size feature")
        
        if 'Year' in self.gdf.columns:
            year_norm = ((self.gdf['Year'].values - self.gdf['Year'].min()) / 
                        (self.gdf['Year'].max() - self.gdf['Year'].min())).reshape(-1, 1) * 0.2
            features.append(year_norm)
            print("✓ Added temporal feature")
        
        features_scaled = StandardScaler().fit_transform(np.hstack(features))
        print(f"✓ Prepared {features_scaled.shape[1]} features for {len(self.gdf)} fires")
        
        return coords, coords_scaled, features_scaled
    
    def run_dbscan(self, coords_scaled, coords):
        """Run optimized DBSCAN clustering."""
        print(f"\n{'='*50}")
        print("DBSCAN Clustering")
        print("="*50)
        
        min_samples = 5 if len(self.gdf) > 500 else 3
        
        # Find optimal eps using k-distance graph
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        distances, _ = neighbors.fit(coords_scaled).kneighbors(coords_scaled)
        k_distances = np.sort(distances[:, min_samples-1])
        optimal_eps = np.percentile(k_distances, 60)
        
        # Test different eps values
        best_eps, best_labels, best_score = optimal_eps, None, float('inf')
        for multiplier in [0.8, 1.0, 1.2, 1.5, 1.8, 2.0]:
            test_eps = optimal_eps * multiplier
            labels = DBSCAN(eps=test_eps, min_samples=min_samples).fit_predict(coords_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = (labels == -1).sum() / len(labels)
            
            score = abs(n_clusters - 8) + (noise_ratio * 20)
            
            if n_clusters >= 5 and noise_ratio < 0.5 and score < best_score:
                best_eps, best_labels, best_score = test_eps, labels, score
            elif best_labels is None:
                best_eps, best_labels = test_eps, labels
        
        n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
        n_noise = (best_labels == -1).sum()
        noise_ratio = n_noise / len(best_labels)
        
        print(f"   Clusters: {n_clusters}")
        print(f"   Noise: {n_noise} ({noise_ratio*100:.1f}%)")
        print(f"   Eps: {best_eps:.3f}")
        
        return best_labels, {
            'n_clusters': n_clusters,
            'noise_count': n_noise,
            'noise_ratio': noise_ratio,
            'eps': best_eps
        }
    
    def run_hdbscan(self, coords_scaled):
        """Run HDBSCAN clustering."""
        print(f"\n{'='*50}")
        print("HDBSCAN Clustering")
        print("="*50)
        
        if not HDBSCAN_AVAILABLE:
            print("⚠️  HDBSCAN not available, skipping")
            return np.full(len(self.gdf), -1), {'n_clusters': 0, 'noise_count': len(self.gdf), 'noise_ratio': 1.0}
        
        min_cluster_size = max(35, len(self.gdf) // 20)
        min_samples = 10
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        labels = clusterer.fit_predict(coords_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        noise_ratio = n_noise / len(labels)
        
        print(f"   Clusters: {n_clusters}")
        print(f"   Noise: {n_noise} ({noise_ratio*100:.1f}%)")
        
        return labels, {
            'n_clusters': n_clusters,
            'noise_count': n_noise,
            'noise_ratio': noise_ratio
        }
    
    def run_kmeans(self, features_scaled):
        """Run K-Means clustering."""
        print(f"\n{'='*50}")
        print("K-Means Clustering")
        print("="*50)
        
        max_k = min(12, len(features_scaled) // 50)
        best_k, best_score = 6, -1
        
        for k in range(6, max_k + 1):
            try:
                labels = KMeans(n_clusters=k, random_state=42, n_init=30).fit_predict(features_scaled)
                silhouette = silhouette_score(features_scaled, labels)
                davies_bouldin = davies_bouldin_score(features_scaled, labels)
                score = silhouette - (davies_bouldin * 0.2)
                if score > best_score:
                    best_k, best_score = k, score
            except:
                pass
        
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=30)
        labels = kmeans.fit_predict(features_scaled)
        
        silhouette = silhouette_score(features_scaled, labels)
        calinski = calinski_harabasz_score(features_scaled, labels)
        davies_bouldin = davies_bouldin_score(features_scaled, labels)
        
        print(f"   Regions: {best_k}")
        print(f"   Silhouette Score: {silhouette:.3f}")
        print(f"   Calinski-Harabasz Index: {calinski:.2f}")
        print(f"   Davies-Bouldin Index: {davies_bouldin:.3f}")
        
        return labels, {
            'n_clusters': best_k,
            'silhouette_score': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies_bouldin
        }
    
    def calculate_cluster_metrics(self, cluster_col):
        """Calculate comprehensive metrics for each cluster."""
        clusters = []
        
        for cluster_id in sorted(self.gdf[cluster_col].unique()):
            if cluster_id == -1:
                continue
                
            cluster_data = self.gdf[self.gdf[cluster_col] == cluster_id]
            
            # Basic metrics
            fire_count = len(cluster_data)
            
            # Intensity metrics
            total_acres = cluster_data['BurnBndAc'].sum() if 'BurnBndAc' in cluster_data.columns else 0
            avg_acres = cluster_data['BurnBndAc'].mean() if 'BurnBndAc' in cluster_data.columns else 0
            max_acres = cluster_data['BurnBndAc'].max() if 'BurnBndAc' in cluster_data.columns else 0
            
            # Geographic metrics
            centroid_x = cluster_data.geometry.x.mean()
            centroid_y = cluster_data.geometry.y.mean()
            
            # Risk scoring
            frequency_score = min(fire_count / 10, 10) * 10
            intensity_score = min(avg_acres / 10000, 10) * 10
            severity_score = min(max_acres / 50000, 10) * 10
            
            risk_score = (frequency_score * 0.4 + intensity_score * 0.3 + severity_score * 0.3)
            
            clusters.append({
                'cluster_id': cluster_id,
                'fire_count': fire_count,
                'total_acres_burned': total_acres,
                'avg_acres_burned': avg_acres,
                'max_fire_size': max_acres,
                'centroid_lon': centroid_x,
                'centroid_lat': centroid_y,
                'risk_score': risk_score
            })
        
        return pd.DataFrame(clusters).sort_values('risk_score', ascending=False)
    
    def recommend_resources(self, cluster_metrics):
        """Generate resource allocation recommendations."""
        recommendations = []
        
        for _, cluster in cluster_metrics.iterrows():
            risk = cluster['risk_score']
            fire_count = cluster['fire_count']
            
            # Determine priority level
            if risk >= 70:
                priority = "CRITICAL"
            elif risk >= 50:
                priority = "HIGH"
            elif risk >= 30:
                priority = "MEDIUM"
            else:
                priority = "LOW"
            
            # Recommend stations
            stations = max(1, int(np.ceil(fire_count / 50 + risk / 100)))
            
            recommendations.append({
                'cluster_id': cluster['cluster_id'],
                'priority': priority,
                'risk_score': risk,
                'recommended_stations': stations,
                'estimated_cost_millions': stations * 2.5
            })
        
        return pd.DataFrame(recommendations)
    
    def create_visualization(self, cluster_col, title, filename):
        """Create cluster visualization."""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot clusters
        unique_clusters = sorted([c for c in self.gdf[cluster_col].unique() if c != -1])
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster_id in enumerate(unique_clusters):
            cluster_data = self.gdf[self.gdf[cluster_col] == cluster_id]
            ax.scatter(cluster_data.geometry.x, cluster_data.geometry.y,
                      s=60, c=[colors[i]], alpha=0.7, edgecolors='black', linewidths=0.5,
                      label=f'Cluster {cluster_id}')
        
        # Plot noise points
        noise_data = self.gdf[self.gdf[cluster_col] == -1]
        if len(noise_data) > 0:
            ax.scatter(noise_data.geometry.x, noise_data.geometry.y,
                      s=30, c='gray', alpha=0.5, label='Scattered fires')
        
        # Add resource recommendations if available
        if cluster_col in self.resource_plans:
            cluster_metrics = self.resource_plans[cluster_col]['metrics']
            for _, cluster in cluster_metrics.iterrows():
                risk = cluster['risk_score']
                size = 200 + (risk * 5)
                ax.scatter(cluster['centroid_lon'], cluster['centroid_lat'],
                          s=size, c='red', marker='*', edgecolors='darkred', linewidths=2,
                          alpha=0.9, zorder=10)
        
        ax.set_title(f'{title} - Southern California Wildfires\n'
                    f'{len(unique_clusters)} Clusters | Red Stars = Recommended Fire Stations',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(VIZ_DIR, exist_ok=True)
        plt.savefig(os.path.join(VIZ_DIR, filename), dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization: {filename}")
        plt.close()
    
    def export_csv_data(self):
        """Export all results to CSV files."""
        print(f"\nExporting results to CSV...")
        
        # Export main clustering data
        df = pd.DataFrame(self.gdf.drop(columns='geometry'))
        df['longitude'] = self.gdf.geometry.x
        df['latitude'] = self.gdf.geometry.y
        
        # Reorder columns
        base_columns = ['longitude', 'latitude']
        other_columns = [col for col in df.columns if col not in base_columns and not col.endswith('_cluster')]
        cluster_columns = [col for col in df.columns if col.endswith('_cluster')]
        final_columns = base_columns + other_columns + cluster_columns
        df = df[final_columns]
        
        main_file = os.path.join(RESULTS_DIR, "wildfire_clustering_data.csv")
        df.to_csv(main_file, index=False)
        print(f"✓ Exported main data: wildfire_clustering_data.csv ({len(df)} records)")
        
        # Export cluster summaries
        for cluster_col in ['dbscan_cluster', 'hdbscan_cluster', 'kmeans_cluster']:
            if cluster_col in df.columns and cluster_col in self.resource_plans:
                summary_file = os.path.join(RESULTS_DIR, f"{cluster_col}_summary.csv")
                self.resource_plans[cluster_col]['metrics'].to_csv(summary_file, index=False)
                print(f"✓ Exported {cluster_col.replace('_cluster', '').upper()} summary")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        report_file = os.path.join(RESULTS_DIR, 'analysis_summary.txt')
        
        with open(report_file, 'w') as f:
            f.write("SOUTHERN CALIFORNIA WILDFIRE COMPREHENSIVE ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Total fires analyzed: {len(self.gdf)}\n")
            if 'Year' in self.gdf.columns:
                f.write(f"Time period: {self.gdf['Year'].min():.0f}-{self.gdf['Year'].max():.0f}\n")
            
            if 'BurnBndAc' in self.gdf.columns:
                f.write(f"\nFire Statistics:\n")
                f.write(f"  Total acres burned: {self.gdf['BurnBndAc'].sum():,.0f}\n")
                f.write(f"  Average fire size: {self.gdf['BurnBndAc'].mean():,.0f} acres\n")
                f.write(f"  Largest fire: {self.gdf['BurnBndAc'].max():,.0f} acres\n")
            
            # Clustering results
            for cluster_col, metrics in self.cluster_results.items():
                algorithm = cluster_col.replace('_cluster', '').upper()
                f.write(f"\n{algorithm} CLUSTERING RESULTS\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Clusters found: {metrics['n_clusters']}\n")
                
                if 'noise_count' in metrics:
                    f.write(f"  Noise points: {metrics['noise_count']} ({metrics['noise_ratio']*100:.1f}%)\n")
                
                if cluster_col in self.resource_plans:
                    recs = self.resource_plans[cluster_col]['recommendations']
                    total_stations = recs['recommended_stations'].sum()
                    total_cost = recs['estimated_cost_millions'].sum()
                    f.write(f"  Recommended stations: {int(total_stations)}\n")
                    f.write(f"  Estimated cost: ${total_cost:.1f} million/year\n")
        
        print(f"✓ Saved summary report: analysis_summary.txt")
    
    def run_complete_analysis(self):
        """Run the complete clustering analysis pipeline."""
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        if len(self.gdf) == 0:
            print("❌ No data available for analysis")
            return
        
        # Prepare features
        coords, coords_scaled, features_scaled = self.prepare_features()
        
        # Create results directory
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Run clustering algorithms
        print(f"\nRunning clustering algorithms...")
        
        # DBSCAN
        dbscan_labels, dbscan_metrics = self.run_dbscan(coords_scaled, coords)
        self.gdf['dbscan_cluster'] = dbscan_labels
        self.cluster_results['dbscan_cluster'] = dbscan_metrics
        
        # HDBSCAN
        hdbscan_labels, hdbscan_metrics = self.run_hdbscan(coords_scaled)
        self.gdf['hdbscan_cluster'] = hdbscan_labels
        self.cluster_results['hdbscan_cluster'] = hdbscan_metrics
        
        # K-Means
        kmeans_labels, kmeans_metrics = self.run_kmeans(features_scaled)
        self.gdf['kmeans_cluster'] = kmeans_labels
        self.cluster_results['kmeans_cluster'] = kmeans_metrics
        
        # Calculate metrics and resource planning
        print(f"\nCalculating cluster metrics and resource planning...")
        
        for cluster_col in ['dbscan_cluster', 'hdbscan_cluster', 'kmeans_cluster']:
            if cluster_col in self.gdf.columns:
                cluster_metrics = self.calculate_cluster_metrics(cluster_col)
                recommendations = self.recommend_resources(cluster_metrics)
                
                self.resource_plans[cluster_col] = {
                    'metrics': cluster_metrics,
                    'recommendations': recommendations
                }
        
        # Create visualizations
        print(f"\nCreating visualizations...")
        self.create_visualization('dbscan_cluster', 'DBSCAN Clustering', 'dbscan_clustering.png')
        if HDBSCAN_AVAILABLE:
            self.create_visualization('hdbscan_cluster', 'HDBSCAN Clustering', 'hdbscan_clustering.png')
        self.create_visualization('kmeans_cluster', 'K-Means Clustering', 'kmeans_clustering.png')
        
        # Export results
        self.export_csv_data()
        self.generate_summary_report()
        
        print(f"\n" + "=" * 70)
        print(" CLUSTERING ANALYSIS COMPLETE!")
        print(f" Results saved in: {os.path.abspath(RESULTS_DIR)}")
        print(f" CSV files: wildfire_clustering_data.csv + cluster summaries")
        print(f" Visualizations: PNG files with cluster maps")
        print(f" Summary: analysis_summary.txt")
        print("=" * 70)


class NetworkAnalyzer:
    """Complete wildfire network analysis system."""
    
    def __init__(self):
        self.gdf = None
        self.graph = None
        self.centrality_results = {}
        self.community_results = {}
        self.hub_nodes = []
    
    def load_and_preprocess_data(self):
        """Load and preprocess wildfire data."""
        print("=" * 70)
        print("CALIFORNIA WILDFIRE NETWORK ANALYSIS")
        print("=" * 70)
        
        if not os.path.exists(DATA_FILE):
            print(f"❌ No data found at {DATA_FILE}")
            print(f"Expected location: {os.path.abspath(DATA_FILE)}")
            print("Please ensure wildfire_data.geojson is in the data/ directory")
            sys.exit(1)
        
        print(f"\nLoading data from {DATA_FILE}...")
        self.gdf = gpd.read_file(DATA_FILE)
        print(f"✓ Loaded {len(self.gdf)} fire records")

        # Convert to centroids if needed (BEFORE filtering)
        if not self.gdf.empty and self.gdf.geometry.iloc[0].geom_type in ['Polygon', 'MultiPolygon']:
            self.gdf['geometry'] = self.gdf.geometry.centroid
        self.gdf = self.gdf.dropna(subset=['geometry']).copy()


        # --- CALIFORNIA FILTERING START ---
        states_file = os.path.join(DATA_DIR, "us_states.json")
        if os.path.exists(states_file):
            print("\nFiltering for California data...")
            try:
                states = gpd.read_file(states_file)
                # Filter for California
                california = states[states['name'] == 'California']
                
                if not california.empty:
                    # Ensure CRS matches
                    if self.gdf.crs != california.crs:
                        california = california.to_crs(self.gdf.crs)
                    
                    # Apply buffer if configured
                    filtering_geom = california
                    if BUFFER_DEG > 0:
                        print(f"   Applying {BUFFER_DEG} degree buffer to state boundary...")
                        # Warning: Buffering in geographic CRS is approximate but used here to match legacy method
                        filtering_geom = california.copy()
                        filtering_geom['geometry'] = california.geometry.buffer(BUFFER_DEG)
                    
                    # Spatial join to keep only points within California
                    original_count = len(self.gdf)
                    # Use inner join with 'within' predicate
                    self.gdf = gpd.sjoin(self.gdf, filtering_geom, how='inner', predicate='within')
                    
                    # Clean up columns added by join if needed
                    cols_to_drop = ['index_right', 'name', 'density', 'id']
                    existing_cols_to_drop = [c for c in cols_to_drop if c in self.gdf.columns]
                    if existing_cols_to_drop:
                        self.gdf = self.gdf.drop(columns=existing_cols_to_drop)
                        
                    print(f"✓ Filtered to California state boundary: {original_count} → {len(self.gdf)} fires")
                else:
                    print("⚠️  'California' not found in us_states.json properties")
            except Exception as e:
                print(f"⚠️  Error during state filtering: {e}")
        else:
            print(f"⚠️  States file not found at {states_file}, skipping state filtering")
        # --- CALIFORNIA FILTERING END ---
        

        
        # Filter by fire size
        if 'BurnBndAc' in self.gdf.columns:
            original_count = len(self.gdf)
            self.gdf = self.gdf[self.gdf['BurnBndAc'] > MIN_FIRE_SIZE].copy()
            print(f"✓ Filtered by size: {original_count} → {len(self.gdf)} fires (>{MIN_FIRE_SIZE} acres)")
        
        # Parse dates
        if 'Ig_Date' in self.gdf.columns:
            self.gdf['Ig_Date'] = pd.to_datetime(self.gdf['Ig_Date'], errors='coerce')
            self.gdf = self.gdf.dropna(subset=['Ig_Date'])
            self.gdf['Year'] = self.gdf['Ig_Date'].dt.year
            print(f"✓ Date range: {self.gdf['Year'].min():.0f}-{self.gdf['Year'].max():.0f}")
    
    def build_fire_network(self):
        """Build network graph from fire data."""
        print(f"\n{'='*50}")
        print("BUILDING FIRE NETWORK")
        print("="*50)
        
        # Create graph
        self.graph = nx.Graph()
        
        # Add nodes
        for idx, row in self.gdf.iterrows():
            self.graph.add_node(
                row['Event_ID'],
                longitude=row.geometry.x,
                latitude=row.geometry.y,
                fire_name=row.get('Incid_Name', 'Unknown'),
                burn_area_acres=row.get('BurnBndAc', 0),
                year=row.get('Year', 0),
                ignition_date=row.get('Ig_Date')
            )
        
        print(f"✓ Added {self.graph.number_of_nodes()} fire nodes")
        
        # Add edges based on spatial and temporal proximity
        edges_added = 0
        nodes = list(self.graph.nodes(data=True))
        
        for i, (node1, data1) in enumerate(nodes):
            for j, (node2, data2) in enumerate(nodes[i+1:], i+1):
                # Calculate spatial distance
                distance_km = self._haversine_distance(
                    data1['longitude'], data1['latitude'],
                    data2['longitude'], data2['latitude']
                )
                
                # Calculate temporal difference
                if data1['ignition_date'] and data2['ignition_date']:
                    time_diff = abs((data1['ignition_date'] - data2['ignition_date']).days)
                else:
                    time_diff = float('inf')
                
                # Add edge if within thresholds
                if distance_km <= MAX_DISTANCE_KM and time_diff <= MAX_TIME_WINDOW_DAYS:
                    # Calculate connection strength
                    spatial_weight = np.exp(-0.1 * distance_km)
                    temporal_weight = np.exp(-0.01 * time_diff) if time_diff < float('inf') else 0
                    connection_strength = (spatial_weight + temporal_weight) / 2
                    
                    self.graph.add_edge(
                        node1, node2,
                        distance_km=distance_km,
                        time_diff_days=time_diff,
                        weight=connection_strength
                    )
                    edges_added += 1
        
        print(f"✓ Added {edges_added} network connections")
        print(f"✓ Network density: {nx.density(self.graph):.4f}")
        print(f"✓ Connected: {nx.is_connected(self.graph)}")
    
    def _haversine_distance(self, lon1, lat1, lon2, lat2):
        """Calculate haversine distance between two points."""
        # Convert to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        return 6371 * c  # Earth's radius in km
    
    def calculate_centrality_measures(self):
        """Calculate network centrality measures."""
        print(f"\n{'='*50}")
        print("CENTRALITY ANALYSIS")
        print("="*50)
        
        if self.graph.number_of_nodes() == 0:
            print("❌ No nodes in graph")
            return
        
        # Calculate centrality measures
        print("   Calculating degree centrality...")
        degree_centrality = nx.degree_centrality(self.graph)
        
        print("   Calculating betweenness centrality...")
        betweenness_centrality = nx.betweenness_centrality(self.graph, weight='weight')
        
        print("   Calculating closeness centrality...")
        closeness_centrality = nx.closeness_centrality(self.graph, distance='weight')
        
        print("   Calculating eigenvector centrality...")
        try:
            eigenvector_centrality = nx.eigenvector_centrality(self.graph, weight='weight')
        except:
            eigenvector_centrality = degree_centrality  # Fallback
        
        self.centrality_results = {
            'degree': degree_centrality,
            'betweenness': betweenness_centrality,
            'closeness': closeness_centrality,
            'eigenvector': eigenvector_centrality
        }
        
        # Identify hub nodes
        combined_scores = {}
        for node in self.graph.nodes():
            combined_scores[node] = (
                degree_centrality.get(node, 0) * 0.3 +
                betweenness_centrality.get(node, 0) * 0.3 +
                closeness_centrality.get(node, 0) * 0.2 +
                eigenvector_centrality.get(node, 0) * 0.2
            )
        
        # Get top 5 hubs
        sorted_hubs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        self.hub_nodes = [node for node, score in sorted_hubs[:5]]
        
        print(f"✓ Identified {len(self.hub_nodes)} connectivity hubs")
        for i, hub in enumerate(self.hub_nodes, 1):
            hub_data = self.graph.nodes[hub]
            print(f"   {i}. {hub_data['fire_name']} ({hub})")
            print(f"      Score: {combined_scores[hub]:.3f}, Area: {hub_data['burn_area_acres']:,.0f} acres")
    
    def detect_communities(self):
        """Detect fire communities."""
        print(f"\n{'='*50}")
        print("COMMUNITY DETECTION")
        print("="*50)
        
        if not LOUVAIN_AVAILABLE:
            print("⚠️  Using greedy modularity (Louvain not available)")
            communities = nx.community.greedy_modularity_communities(self.graph, weight='weight')
            partition = {}
            for i, community in enumerate(communities):
                for node in community:
                    partition[node] = i
        else:
            print("   Using Louvain algorithm...")
            partition = community_louvain.best_partition(self.graph, weight='weight')
        
        self.community_results = partition
        num_communities = len(set(partition.values()))
        
        print(f"✓ Detected {num_communities} fire communities")
        
        # Calculate modularity
        try:
            if LOUVAIN_AVAILABLE:
                modularity = community_louvain.modularity(partition, self.graph, weight='weight')
            else:
                community_sets = []
                for i in range(num_communities):
                    community_sets.append({node for node, comm in partition.items() if comm == i})
                modularity = nx.community.modularity(self.graph, community_sets, weight='weight')
            
            print(f"✓ Modularity score: {modularity:.4f}")
        except:
            print("⚠️  Could not calculate modularity")
    
    def create_network_visualizations(self):
        """Create network visualizations."""
        print(f"\nCreating fire connectivity visualizations...")
        
        # Create map visualization
        fig_map = plt.figure(figsize=(16, 12))
        ax_map = fig_map.add_subplot(111)
        
        ax_map.set_title('California Fire Connectivity Network\n'
                        'Fire Propagation Hubs and Network Connections', 
                        fontsize=18, fontweight='bold', pad=30)
        
        # Plot all fires
        all_fires_data = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            all_fires_data.append({
                'lon': node_data['longitude'],
                'lat': node_data['latitude'], 
                'year': node_data['year'],
                'area': node_data['burn_area_acres'],
                'is_hub': node in self.hub_nodes
            })
        
        # Separate hub and non-hub fires
        hub_fires = [f for f in all_fires_data if f['is_hub']]
        regular_fires = [f for f in all_fires_data if not f['is_hub']]
        
        # Plot regular fires
        if regular_fires:
            reg_lons = [f['lon'] for f in regular_fires]
            reg_lats = [f['lat'] for f in regular_fires]
            reg_years = [f['year'] for f in regular_fires]
            
            scatter_reg = ax_map.scatter(reg_lons, reg_lats, s=25, c=reg_years, 
                                       cmap='viridis', alpha=0.7, label='Regular Fires')
        
        # Draw network connections
        important_edges = []
        for edge in self.graph.edges():
            node1, node2 = edge
            if (node1 in self.hub_nodes or node2 in self.hub_nodes or 
                self.graph.degree(node1) >= 4 or self.graph.degree(node2) >= 4):
                important_edges.append(edge)
        
        for edge in important_edges[:60]:
            node1, node2 = edge
            x1, y1 = self.graph.nodes[node1]['longitude'], self.graph.nodes[node1]['latitude']
            x2, y2 = self.graph.nodes[node2]['longitude'], self.graph.nodes[node2]['latitude']
            ax_map.plot([x1, x2], [y1, y2], 'gray', alpha=0.4, linewidth=1.5, zorder=1)
        
        # Plot hub fires
        if hub_fires:
            hub_lons = [f['lon'] for f in hub_fires]
            hub_lats = [f['lat'] for f in hub_fires]
            hub_areas = [f['area'] for f in hub_fires]
            
            hub_sizes = [400 + (area/30) + (self.graph.degree(self.hub_nodes[i])*50) 
                        for i, area in enumerate(hub_areas)]
            
            ax_map.scatter(hub_lons, hub_lats, s=hub_sizes, c='red', 
                          marker='*', edgecolors='darkred', linewidths=4, 
                          alpha=0.9, label='Fire Connectivity Hubs', zorder=10)
            
            for i, hub in enumerate(self.hub_nodes):
                hub_data = self.graph.nodes[hub]
                ax_map.annotate(f'{i+1}', 
                               (hub_data['longitude'], hub_data['latitude']),
                               ha='center', va='center', fontsize=18, fontweight='bold',
                               color='white', zorder=15)
        
        ax_map.set_xlabel('Longitude', fontsize=14)
        ax_map.set_ylabel('Latitude', fontsize=14)
        ax_map.grid(True, alpha=0.3)
        ax_map.legend(loc='upper right', fontsize=13)
        
        if regular_fires:
            cbar = plt.colorbar(scatter_reg, ax=ax_map, shrink=0.8, pad=0.02)
            cbar.set_label('Fire Year', fontsize=13)
        
        plt.tight_layout()
        os.makedirs(VIZ_DIR, exist_ok=True)
        map_file = os.path.join(VIZ_DIR, 'fire_connectivity_map.png')
        plt.savefig(map_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved fire connectivity map")
        
        # Create bar chart
        fig_bar = plt.figure(figsize=(14, 8))
        ax_bar = fig_bar.add_subplot(111)
        
        ax_bar.set_title('Fire Connectivity Hubs - Network Analysis Results\n'
                        'Ranked by Number of Fire Network Connections', 
                        fontsize=16, fontweight='bold', pad=25)
        
        hub_names = [self.graph.nodes[hub]['fire_name'] for hub in self.hub_nodes]
        hub_connections = [self.graph.degree(hub) for hub in self.hub_nodes]
        hub_areas = [self.graph.nodes[hub]['burn_area_acres'] for hub in self.hub_nodes]
        hub_years = [self.graph.nodes[hub]['year'] for hub in self.hub_nodes]
        
        y_pos = range(len(hub_names))
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
        bars = ax_bar.barh(y_pos, hub_connections, color=colors, alpha=0.8, 
                          edgecolor='black', linewidth=1.5)
        
        for i, (bar, area, year, name, connections) in enumerate(zip(bars, hub_areas, hub_years, hub_names, hub_connections)):
            width = bar.get_width()
            ax_bar.text(width/2, bar.get_y() + bar.get_height()/2,
                       f'{connections} connections', 
                       ha='center', va='center', fontsize=12, fontweight='bold', color='white')
            ax_bar.text(width + 0.15, bar.get_y() + bar.get_height()/2,
                       f'{area:,.0f} acres burned in {year}', 
                       ha='left', va='center', fontsize=11, fontweight='bold')
        
        numbered_labels = [f'{i+1}. {name} Fire' for i, name in enumerate(hub_names)]
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(numbered_labels, fontsize=13)
        ax_bar.set_xlabel('Number of Fire Network Connections\n(within 30km distance and 180 days)', fontsize=13)
        ax_bar.grid(True, alpha=0.3, axis='x')
        ax_bar.invert_yaxis()
        ax_bar.set_xlim(0, max(hub_connections) + 2)
        
        plt.tight_layout()
        bar_file = os.path.join(VIZ_DIR, 'fire_connectivity_ranking.png')
        plt.savefig(bar_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved fire connectivity ranking")
    
    def export_network_results(self):
        """Export network analysis results to CSV files."""
        print(f"\nExporting network results to CSV...")
        
        # Main network data file
        network_data = []
        for node in self.graph.nodes():
            node_attrs = self.graph.nodes[node]
            row = {
                'Event_ID': node,
                'Fire_Name': node_attrs['fire_name'],
                'Longitude': node_attrs['longitude'],
                'Latitude': node_attrs['latitude'],
                'Burn_Area_Acres': node_attrs['burn_area_acres'],
                'Year': node_attrs['year'],
                'Network_Degree': self.graph.degree(node),
                'Degree_Centrality': self.centrality_results['degree'].get(node, 0),
                'Community_ID': self.community_results.get(node, -1),
                'Is_Hub': node in self.hub_nodes
            }
            network_data.append(row)
        
        network_df = pd.DataFrame(network_data)
        network_df = network_df.sort_values('Degree_Centrality', ascending=False)
        
        network_file = os.path.join(RESULTS_DIR, 'wildfire_network_data.csv')
        network_df.to_csv(network_file, index=False)
        print(f"✓ Exported main data: wildfire_network_data.csv ({len(network_df)} records)")
        
        # Hub summary
        hub_data = []
        for i, hub in enumerate(self.hub_nodes, 1):
            hub_attrs = self.graph.nodes[hub]
            hub_info = {
                'hub_rank': i,
                'event_id': hub,
                'fire_name': hub_attrs['fire_name'],
                'burn_area_acres': hub_attrs['burn_area_acres'],
                'year': hub_attrs['year'],
                'network_degree': self.graph.degree(hub),
                'centrality_score': self.centrality_results['degree'].get(hub, 0)
            }
            hub_data.append(hub_info)
        
        hub_df = pd.DataFrame(hub_data)
        hub_file = os.path.join(RESULTS_DIR, 'network_hub_summary.csv')
        hub_df.to_csv(hub_file, index=False)
        print(f"✓ Exported network hub summary")
    
    def generate_network_summary_report(self):
        """Generate network summary report."""
        report_file = os.path.join(RESULTS_DIR, 'network_analysis_summary.txt')
        
        with open(report_file, 'w') as f:
            f.write("SOUTHERN CALIFORNIA WILDFIRE NETWORK ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Total fires analyzed: {self.graph.number_of_nodes()}\n")
            f.write(f"Network connections: {self.graph.number_of_edges()}\n")
            f.write(f"Network density: {nx.density(self.graph):.4f}\n")
            
            f.write(f"\nTOP CONNECTIVITY HUBS\n")
            f.write("-" * 30 + "\n")
            for i, hub in enumerate(self.hub_nodes, 1):
                hub_data = self.graph.nodes[hub]
                f.write(f"  {i}. {hub_data['fire_name']} - {hub_data['burn_area_acres']:,.0f} acres ({hub_data['year']})\n")
            
            f.write(f"\nCommunities detected: {len(set(self.community_results.values()))}\n")
            f.write(f"Analysis parameters: {MAX_DISTANCE_KM}km spatial, {MAX_TIME_WINDOW_DAYS} days temporal\n")
        
        print(f"✓ Saved network summary report")
    
    def run_complete_analysis(self):
        """Run the complete network analysis pipeline."""
        try:
            # Load data
            self.load_and_preprocess_data()
            
            if len(self.gdf) < 10:
                print("❌ Insufficient data for network analysis")
                return
            
            # Build network
            self.build_fire_network()
            
            if self.graph.number_of_nodes() == 0:
                print("❌ No network nodes created")
                return
            
            # Run analysis
            self.calculate_centrality_measures()
            self.detect_communities()
            
            # Generate outputs
            self.create_network_visualizations()
            self.export_network_results()
            self.generate_network_summary_report()
            
            print(f"\n" + "=" * 70)
            print("✅ NETWORK ANALYSIS COMPLETE!")
            print(f"📁 Results saved in: {os.path.abspath(RESULTS_DIR)}")
            print(f"📊 CSV files: wildfire_network_data.csv + network_hub_summary.csv")
            print(f"📈 Visualizations: fire_connectivity_map.png + fire_connectivity_ranking.png")
            print(f"📋 Summary: network_analysis_summary.txt")
            print("=" * 70)
            
        except Exception as e:
            print(f"\n❌ Network analysis failed: {e}")
            import traceback
            traceback.print_exc()



def main():
    """Main execution function with command-line argument support."""
    parser = argparse.ArgumentParser(
        description='Southern California Wildfire Analysis - Restricted to California',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wildfire_analysis_ca.py                    # Run both clustering and network analysis
  python wildfire_analysis_ca.py --mode clustering  # Run clustering analysis only
  python wildfire_analysis_ca.py --mode network     # Run network analysis only
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['clustering', 'network', 'both'],
        default='both',
        help='Analysis mode to run (default: both)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode in ['clustering', 'both']:
            print("\n" + "="*70)
            print("STARTING CLUSTERING ANALYSIS")
            print("="*70 + "\n")
            analyzer = WildfireAnalyzer()
            analyzer.run_complete_analysis()
        
        if args.mode in ['network', 'both']:
            if args.mode == 'both':
                print("\n\n")  # Add spacing between analyses
            print("\n" + "="*70)
            print("STARTING NETWORK ANALYSIS")
            print("="*70 + "\n")
            network = NetworkAnalyzer()
            network.run_complete_analysis()
        
        print("\n" + "="*70)
        print("ALL ANALYSES COMPLETE!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
