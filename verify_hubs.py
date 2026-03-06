import pandas as pd
import networkx as nx
import numpy as np
from math import radians, cos, sin, asin, sqrt

# Load data
df = pd.read_csv('data/california_wildfire_data.csv')
print(f"Loaded {len(df)} fires")

# Constants from analysis script
MAX_DISTANCE_KM = 30.0
MAX_TIME_WINDOW_DAYS = 180

def haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

# Build Graph
G = nx.Graph()
for idx, row in df.iterrows():
    G.add_node(row['Event_ID'], year=row['Year'], name=row['Incid_Name'], 
               lon=row['longitude'], lat=row['latitude'], 
               date=pd.to_datetime(row['Ig_Date']))

nodes = list(G.nodes(data=True))
print("Building edges...")
for i, (node1, data1) in enumerate(nodes):
    for j, (node2, data2) in enumerate(nodes[i+1:], i+1):
        dist = haversine_distance(data1['lon'], data1['lat'], data2['lon'], data2['lat'])
        time_diff = abs((data1['date'] - data2['date']).days)
        
        if dist <= MAX_DISTANCE_KM and time_diff <= MAX_TIME_WINDOW_DAYS:
            G.add_edge(node1, node2)

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Calculate Centrality
print("Calculating centrality...")
degree = nx.degree_centrality(G)
# using unweighted for speed/simplicity as we just want to check year dominance
# The user likely saw high degree/centrality nodes.

top_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:20]

print("\nTOP 20 HUBS (Degree Centrality):")
print(f"{'Rank':<5} {'Year':<6} {'Name':<20} {'Score':<10}")
print("-" * 50)
for i, (node, score) in enumerate(top_nodes, 1):
    year = G.nodes[node]['year']
    name = G.nodes[node]['name']
    print(f"{i:<5} {year:<6} {name:<20} {score:.4f}")

# Check Year Distribution of top 50
top_50 = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:50]
years = [G.nodes[n]['year'] for n, s in top_50]
print("\nYear distribution in Top 50 Hubs:")
print(pd.Series(years).value_counts())
