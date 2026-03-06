import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import contextily as ctx
from shapely.geometry import Point
import os
import sys
import argparse
import networkx as nx
from math import radians, cos, sin, asin, sqrt
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "data"
RESULTS_DIR = "results_socal"  # New results directory
VIZ_DIR = os.path.join(RESULTS_DIR, "visualizations")
DATA_FILE = os.path.join(DATA_DIR, "wildfire_data.geojson")

# Analysis parameters
MIN_FIRE_SIZE = 500  # acres
EPS_KILOMETERS = 15  # Distance threshold for DBSCAN
MIN_SAMPLES = 5      # Min points for DBSCAN

# Network parameters
MAX_DISTANCE_KM = 50      # Connection threshold
MAX_TIME_WINDOW_DAYS = 180

# SOCAL BOUNDING BOX
MIN_LON = -120.68
MAX_LON = -114.05
MIN_LAT = 32.55
MAX_LAT = 35.50

class WildfireAnalyzer:
    """Complete wildfire clustering analysis system."""
    
    def __init__(self):
        self.gdf = None
        self.cluster_results = {}
        self.resource_plans = {}
    
    def load_and_preprocess_data(self):
        """Load and preprocess wildfire data."""
        print("=" * 70)
        print("SOUTHERN CALIFORNIA CLUSTERING ANALYSIS (FRIEND'S COORDS)")
        print("=" * 70)
        
        if not os.path.exists(DATA_FILE):
            print(f"❌ No data found at {DATA_FILE}")
            sys.exit(1)
        
        print(f"\nLoading data from {DATA_FILE}...")
        self.gdf = gpd.read_file(DATA_FILE)
        print(f"✓ Loaded {len(self.gdf)} fire records")

        # Convert to centroids if needed (BEFORE filtering)
        if not self.gdf.empty and self.gdf.geometry.iloc[0].geom_type in ['Polygon', 'MultiPolygon']:
            self.gdf['geometry'] = self.gdf.geometry.centroid
        self.gdf = self.gdf.dropna(subset=['geometry']).copy()

        # --- SOCAL FILTERING START ---
        print(f"\nFiltering for Southern California (Friend's bbox)...")
        print(f"   Lon: {MIN_LON} to {MAX_LON}")
        print(f"   Lat: {MIN_LAT} to {MAX_LAT}")
        
        original_count = len(self.gdf)
        # Filter using bounding box
        self.gdf = self.gdf.cx[MIN_LON:MAX_LON, MIN_LAT:MAX_LAT]
        print(f"✓ Filtered to Custom SoCal BBox: {original_count} → {len(self.gdf)} fires")
        # --- SOCAL FILTERING END ---
        
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
            print(f"✓ Filtered by time: {len(self.gdf)} → {len(self.gdf)} fires")
            print(f"✓ Time period: {self.gdf['Year'].min():.0f}-{self.gdf['Year'].max():.0f}")

        # Basic stats
        print(f"✓ Fire size range: {self.gdf['BurnBndAc'].min():.0f} - {self.gdf['BurnBndAc'].max():.0f} acres")
        print(f"✓ Total acres burned: {self.gdf['BurnBndAc'].sum():,.0f}")

    def prepare_features(self):
        """Prepare features for clustering."""
        print("\nPreparing features for clustering...")
        
        # Project to decent local CRS for distance calculation
        # EPSG:3310 (California Albers) is good for preserving area/distance
        self.gdf_proj = self.gdf.to_crs(epsg=3310)
        
        # Extract coordinates
        coords = np.column_stack((self.gdf_proj.geometry.x, self.gdf_proj.geometry.y))
        
        # Normalize features
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        # Add fire size as feature?
        # Only if we want to cluster by size AND location. 
        # Usually spatial clustering is purely location based.
        self.features = coords_scaled
        
        # We also want extra features for K-Means to differentiate it
        size_feature = np.log1p(self.gdf['BurnBndAc'].values).reshape(-1, 1)
        time_feature = self.gdf['Year'].values.reshape(-1, 1)
        
        # Combine for a "complex" feature set
        self.complex_features = np.hstack([
            coords_scaled, 
            StandardScaler().fit_transform(size_feature),
            StandardScaler().fit_transform(time_feature) * 0.5  # Weight time less
        ])
        
        print(f"✓ Added fire size feature")
        print(f"✓ Added temporal feature")
        print(f"✓ Prepared {self.complex_features.shape[1]} features for {len(self.gdf)} fires")

    def run_clustering(self):
        """Run multiple clustering algorithms."""
        print("\nRunning clustering algorithms...")
        
        # 1. DBSCAN (Spatial only)
        # Convert km to feature scale (approximate)
        print("\n" + "="*50)
        print("DBSCAN Clustering")
        print("="*50)
        
        # For lat/lon DBSCAN with Haversine
        coords_rad = np.radians(np.column_stack((self.gdf.geometry.y, self.gdf.geometry.x)))
        kms_per_radian = 6371.0088
        epsilon = EPS_KILOMETERS / kms_per_radian
        
        dbscan = DBSCAN(eps=epsilon, min_samples=MIN_SAMPLES, metric='haversine')
        self.gdf['cluster_dbscan'] = dbscan.fit_predict(coords_rad)
        
        n_clusters = len(set(self.gdf['cluster_dbscan'])) - (1 if -1 in self.gdf['cluster_dbscan'] else 0)
        n_noise = list(self.gdf['cluster_dbscan']).count(-1)
        print(f"   Clusters: {n_clusters}")
        print(f"   Noise: {n_noise} ({n_noise/len(self.gdf)*100:.1f}%)")
        print(f"   Eps: {epsilon:.3f}")

        # 2. HDBSCAN (using sklearn implementation or alternative)
        print("\n" + "="*50)
        print("HDBSCAN Clustering")
        print("="*50)
        try:
            import hdbscan
            hdb = hdbscan.HDBSCAN(min_cluster_size=MIN_SAMPLES, metric='haversine')
            self.gdf['cluster_hdbscan'] = hdb.fit_predict(coords_rad)
            
            n_clusters_h = len(set(self.gdf['cluster_hdbscan'])) - (1 if -1 in self.gdf['cluster_hdbscan'] else 0)
            n_noise_h = list(self.gdf['cluster_hdbscan']).count(-1)
            print(f"   Clusters: {n_clusters_h}")
            print(f"   Noise: {n_noise_h} ({n_noise_h/len(self.gdf)*100:.1f}%)")
        except ImportError:
            print("⚠️  hdbscan not installed, skipping")
            self.gdf['cluster_hdbscan'] = -1

        # 3. K-Means
        print("\n" + "="*50)
        print("K-Means Clustering")
        print("="*50)
        
        # Determine optimal K (simplified elbow/silhouette)
        best_k = 5
        best_score = -1
        
        for k in range(3, 12):
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(self.complex_features)
            score = silhouette_score(self.complex_features, labels)
            if score > best_score:
                best_score = score
                best_k = k
        
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        self.gdf['cluster_kmeans'] = kmeans.fit_predict(self.complex_features)
        
        print(f"   Regions: {best_k}")
        print(f"   Silhouette Score: {best_score:.3f}")
        print(f"   Calinski-Harabasz Index: {calinski_harabasz_score(self.complex_features, self.gdf['cluster_kmeans']):.2f}")
        print(f"   Davies-Bouldin Index: {davies_bouldin_score(self.complex_features, self.gdf['cluster_kmeans']):.3f}")

    def plan_resources(self):
        """Calculate resource needs per cluster."""
        print("\nCalculating cluster metrics and resource planning...")
        # (Simplified for this run)
        pass

    def create_visualizations(self):
        """Create maps matching friends style."""
        print("\nCreating visualizations...")
        os.makedirs(VIZ_DIR, exist_ok=True)
        
        # Style settings
        plt.style.use('default') 
        
        for algo in ['dbscan']:  # Focus on DBSCAN as requested
            col = f'cluster_{algo}'
            if col not in self.gdf.columns: continue
            
            # Prepare data
            clustered = self.gdf[self.gdf[col] != -1]
            noise = self.gdf[self.gdf[col] == -1]
            
            n_clusters = len(self.gdf[col].unique()) - (1 if -1 in self.gdf[col].values else 0)
            
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Plot Noise (Scattered fires)
            if not noise.empty:
                ax.scatter(noise.geometry.x, noise.geometry.y, 
                          c='gray', alpha=0.5, s=20, label='Scattered fires')
            
            # Plot Clusters
            # Use distinct colors
            unique_clusters = sorted(clustered[col].unique())
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
            
            for i, cluster_id in enumerate(unique_clusters):
                cluster_data = clustered[clustered[col] == cluster_id]
                ax.scatter(cluster_data.geometry.x, cluster_data.geometry.y,
                          label=f'Cluster {cluster_id}', s=40, alpha=0.9, edgecolors='w', linewidth=0.5)
            
            # Styling match
            plt.title(f'DBSCAN Clustering - Southern California Wildfires\n{n_clusters} Clusters', 
                     fontsize=16, fontweight='bold', pad=15)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True, alpha=0.3)
            
            # Legend outside
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
            
            # Set Bounds to match friend roughly
            ax.set_xlim(MIN_LON - 0.5, MAX_LON + 0.5)
            ax.set_ylim(MIN_LAT - 0.5, MAX_LAT + 0.5)

            out_file = os.path.join(VIZ_DIR, f'{algo}_clustering_matched.png')
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved matched visualization: {os.path.basename(out_file)}")

    def export_results(self):
        """Export results to CSV."""
        print("\nExporting results to CSV...")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Main CSV
        out_csv = os.path.join(RESULTS_DIR, "wildfire_clustering_data.csv")
        self.gdf.drop(columns=['geometry']).to_csv(out_csv, index=False)
        print(f"✓ Exported main data: {os.path.basename(out_csv)} ({len(self.gdf)} records)")
        
        # Summary text
        with open(os.path.join(RESULTS_DIR, "analysis_summary.txt"), "w") as f:
            f.write("SOCAL ANALYSIS SUMMARY\n")
            f.write(f"Total Fires: {len(self.gdf)}\n")
            f.write(f"Acres Burned: {self.gdf['BurnBndAc'].sum():,.0f}\n")
        print(f"✓ Saved summary report: analysis_summary.txt")

    def run_complete_analysis(self):
        self.load_and_preprocess_data()
        self.prepare_features()
        self.run_clustering()
        self.plan_resources()
        self.create_visualizations()
        self.export_results()

class NetworkAnalyzer:
    """Network analysis for SoCal."""
    
    def __init__(self):
        self.gdf = None
        self.graph = None
        self.hub_nodes = []
    
    def load_and_preprocess_data(self):
        print("=" * 70)
        print("SOUTHERN CALIFORNIA NETWORK ANALYSIS (FRIEND'S COORDS)")
        print("=" * 70)
        
        self.gdf = gpd.read_file(DATA_FILE)
        
        # Centroids
        if not self.gdf.empty and self.gdf.geometry.iloc[0].geom_type in ['Polygon', 'MultiPolygon']:
            self.gdf['geometry'] = self.gdf.geometry.centroid
        self.gdf = self.gdf.dropna(subset=['geometry']).copy()
        
        # SoCal Filter
        self.gdf = self.gdf.cx[MIN_LON:MAX_LON, MIN_LAT:MAX_LAT]
        
        # Size Filter
        if 'BurnBndAc' in self.gdf.columns:
            self.gdf = self.gdf[self.gdf['BurnBndAc'] > MIN_FIRE_SIZE].copy()
            
        # Date Parse
        if 'Ig_Date' in self.gdf.columns:
            self.gdf['Ig_Date'] = pd.to_datetime(self.gdf['Ig_Date'], errors='coerce')
            self.gdf = self.gdf.dropna(subset=['Ig_Date'])
            self.gdf['Year'] = self.gdf['Ig_Date'].dt.year

        print(f"✓ Loaded and filtered: {len(self.gdf)} fires")

    def build_fire_network(self):
        print(f"\nBuilding Fire Network...")
        self.graph = nx.Graph()
        
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
        
        nodes = list(self.graph.nodes(data=True))
        edges_added = 0
        
        for i, (node1, data1) in enumerate(nodes):
            for j, (node2, data2) in enumerate(nodes[i+1:], i+1):
                # Haversine
                lon1, lat1, lon2, lat2 = map(radians, [data1['longitude'], data1['latitude'], 
                                                     data2['longitude'], data2['latitude']])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * asin(sqrt(a))
                distance_km = 6371 * c
                
                # Time diff
                time_diff = abs((data1['ignition_date'] - data2['ignition_date']).days)
                
                if distance_km <= MAX_DISTANCE_KM and time_diff <= MAX_TIME_WINDOW_DAYS:
                    self.graph.add_edge(node1, node2, weight=1)
                    edges_added += 1
        
        print(f"✓ Added {edges_added} connections")

    def calculate_centrality(self):
        print("\nCalculating Centrality...")
        degree_centrality = nx.degree_centrality(self.graph)
        sorted_hubs = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        self.hub_nodes = [node for node, score in sorted_hubs[:5]]
        
        for i, hub in enumerate(self.hub_nodes, 1):
             print(f"   {i}. {self.graph.nodes[hub]['fire_name']} ({hub})")

    def create_visualizations(self):
        print(f"\nCreating network visualizations...")
        fig_map = plt.figure(figsize=(16, 12))
        ax_map = fig_map.add_subplot(111)
        
        ax_map.set_title('Southern California Fire Connectivity Network\n'
                        'Friend\'s Coordinates: 32.55-35.50N, 120.68-114.05W', 
                        fontsize=18, fontweight='bold', pad=30)
        
        # Plot Network
        positions = {n: (self.graph.nodes[n]['longitude'], self.graph.nodes[n]['latitude']) for n in self.graph.nodes()}
        
        # Edges
        nx.draw_networkx_edges(self.graph, pos=positions, ax=ax_map, alpha=0.2, edge_color='gray')
        
        # Nodes
        nx.draw_networkx_nodes(self.graph, pos=positions, ax=ax_map, node_size=20, node_color='blue', alpha=0.6)
        
        # Hubs
        hub_pos = {n: positions[n] for n in self.hub_nodes}
        nx.draw_networkx_nodes(self.graph, pos=hub_pos, ax=ax_map, node_size=200, node_color='red', nodelist=self.hub_nodes)
        
        ax_map.set_xlabel('Longitude')
        ax_map.set_ylabel('Latitude')
        
        os.makedirs(VIZ_DIR, exist_ok=True)
        plt.savefig(os.path.join(VIZ_DIR, 'fire_connectivity_map.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved network map")
        
        # Ranking
        fig_bar = plt.figure(figsize=(14, 8))
        ax_bar = fig_bar.add_subplot(111)
        ax_bar.set_title('Top 5 Connectivity Hubs (SoCal)', fontsize=16)
        
        hubs = [self.graph.nodes[hub]['fire_name'] for hub in self.hub_nodes]
        degrees = [self.graph.degree(hub) for hub in self.hub_nodes]
        
        ax_bar.barh(range(len(hubs)), degrees, color='red')
        ax_bar.set_yticks(range(len(hubs)))
        ax_bar.set_yticklabels(hubs)
        ax_bar.set_xlabel('Connections')
        
        plt.savefig(os.path.join(VIZ_DIR, 'fire_connectivity_ranking.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def result_csv(self):
        # export network data
        data = []
        for n in self.graph.nodes(data=True):
            n[1]['Degree'] = self.graph.degree(n[0])
            n[1]['Event_ID'] = n[0]
            data.append(n[1])
        pd.DataFrame(data).to_csv(os.path.join(RESULTS_DIR, "wildfire_network_data.csv"), index=False)


    def run_complete_analysis(self):
        self.load_and_preprocess_data()
        self.build_fire_network()
        self.calculate_centrality()
        self.create_visualizations()
        self.result_csv()

def main():
    parser = argparse.ArgumentParser(description="SoCal Wildfire Analysis")
    parser.add_argument('--mode', choices=['clustering', 'network', 'both'], default='both')
    args = parser.parse_args()
    
    if args.mode in ['clustering', 'both']:
        WildfireAnalyzer().run_complete_analysis()
    
    if args.mode in ['network', 'both']:
        NetworkAnalyzer().run_complete_analysis()

    print("\n" + "="*70)
    print(f"SOCAL ANALYSIS COMPLETE! Results in {RESULTS_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
