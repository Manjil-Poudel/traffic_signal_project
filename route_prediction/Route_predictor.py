import osmnx as ox
import networkx as nx
import random
import folium

#Loading OSM file
osm_file = "map.osm"  # place your .osm file here

# Create graph from OSM
G = ox.graph_from_xml(
    osm_file,
    simplify=True,
    retain_all=True,
    bidirectional=False
)

print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")

# Keep only the largest connected component
# Keep only the largest weakly-connected component (best for routing)
largest_cc = max(nx.weakly_connected_components(G), key=len)
G = G.subgraph(largest_cc).copy()
print(f"Largest component: {len(G.nodes)} nodes, {len(G.edges)} edges")

#Assign dummy travel time
for u, v, k, data in G.edges(keys=True, data=True):
    length = data.get("length", 1)  # meters
    base_speed = 11.1  # ~40 km/h in m/s

    # Dummy congestion factor: simulate heavier traffic randomly
    congestion = 1.0
    if random.random() < 0.3:  # 30% edges congested
        congestion = random.uniform(1.5, 3.0)

    # Travel time in seconds
    travel_time = length / base_speed * congestion
    data["travel_time"] = travel_time

# Step 3: Define start & end points (lat, lon)
Pulchowk_chowk = (27.68313, 85.31890)
Maitighar_chowk= (27.69469, 85.32044)
Thapathali_chowk = (27.69061, 85.31779)
Tripureshwor_chowk = (27.69384, 85.31414)

start_latlon = Pulchowk_chowk
end_latlon = Tripureshwor_chowk

# Get nearest nodes in graph
orig_node = ox.distance.nearest_nodes(G, X=start_latlon[1], Y=start_latlon[0])
dest_node = ox.distance.nearest_nodes(G, X=end_latlon[1], Y=end_latlon[0])

print(f"Start node: {orig_node}, End node: {dest_node}")

# -------------------------------
# Step 4: Compute shortest-time route
# -------------------------------
route = nx.shortest_path(
    G,
    orig_node,
    dest_node,
    weight="travel_time"
)
print(f"Route computed: {len(route)} nodes")

# Get center of map
center_lat = sum(nx.get_node_attributes(G, 'y').values()) / len(G.nodes)
center_lon = sum(nx.get_node_attributes(G, 'x').values()) / len(G.nodes)

# Create interactive map
m = folium.Map(location=[center_lat, center_lon], zoom_start=16)

# Convert route node list to lat-lon coordinates
route_coordinates = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]

# Add route polyline
folium.PolyLine(route_coordinates, color="red", weight=5).add_to(m)

# Add start and end markers
folium.Marker(route_coordinates[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
folium.Marker(route_coordinates[-1], popup="End", icon=folium.Icon(color="blue")).add_to(m)

# Save to file
m.save("route_map.html")
print("Saved map to route_map.html")





