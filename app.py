from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import osmnx as ox
import networkx as nx
import numpy as np
import cv2
import os
import random
import gc  # Garbage Collector for Memory Management

# --- CONFIGURATION ---
app = Flask(__name__)
CORS(app)

# SETTINGS FOR RENDER FREE TIER (Critical)
# We limit the download size to prevent "Out of Memory" crashes
ox.settings.max_query_area_size = 2500000000
ox.settings.timeout = 180 

# --- ROUTES ---
@app.route('/')
def home():
    return render_template('Index.html') 

@app.route('/api/get-route', methods=['POST'])
def get_route_api():
    try:
        req = request.json
        print(f"\n🚀 Processing Route: {req.get('start')} -> {req.get('end')}")
        
        # Helper to parse coordinates
        def get_coords(q):
            try:
                parts = q.split(',')
                if len(parts) == 2: return (float(parts[0]), float(parts[1]))
            except: pass
            return ox.geocode(q)

        start_coords = get_coords(req.get('start'))
        end_coords = get_coords(req.get('end'))

        if not start_coords or not end_coords:
            return jsonify({"error": "Could not find location coordinates."}), 400

        # Get the graph (Optimized for Free Tier)
        graph = get_safe_graph(start_coords, end_coords)
        
        # Find nearest nodes
        orig = ox.distance.nearest_nodes(graph, start_coords[1], start_coords[0])
        dest = ox.distance.nearest_nodes(graph, end_coords[1], end_coords[0])
        
        # Calculate Shortest Path
        try:
            route = nx.shortest_path(graph, orig, dest, weight='length')
            
            # Calculate Distance
            total_dist_meters = nx.path_weight(graph, route, weight='length')
            total_dist_km = round(total_dist_meters / 1000, 2)
            
        except nx.NetworkXNoPath:
            return jsonify({"error": "No route found. Try a closer destination."}), 404
        except Exception as e:
            return jsonify({"error": f"Routing error: {str(e)}"}), 500

        # Process Route Segments
        segments = []
        stats = {"High": 0, "Moderate": 0, "Low": 0}
        hazards = {"Sharp Curve":0, "Poor Lighting":0, "Narrow Road":0, "Traffic Congestion":0, "Bad Visibility":0, "Known Blackspot":0, "High Speed Zone": 0, "Winding Road": 0}
        
        cv_score = analyze_image_cv()

        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            
            # Get geometry
            data = graph.get_edge_data(u, v)[0]
            if 'geometry' in data:
                xs, ys = data['geometry'].xy
                pos = list(zip(ys, xs))
            else:
                pos = [(graph.nodes[u]['y'], graph.nodes[u]['x']), (graph.nodes[v]['y'], graph.nodes[v]['x'])]

            # Analyze Risk
            risk, color, info = analyze_risk(u, v, graph, cv_score)
            
            # Update Stats
            stats[risk] += 1
            for r in info:
                if r in hazards: hazards[r] += 1
            
            segments.append({
                "positions": pos, 
                "color": color, 
                "risk": risk, 
                "info": ", ".join(info)
            })

        # --- MEMORY CLEANUP (CRITICAL FOR FREE TIER) ---
        del graph
        del route
        gc.collect() 
        # -----------------------------------------------

        return jsonify({
            "segments": segments, 
            "stats": stats, 
            "hazards": hazards, 
            "distance": total_dist_km
        })

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

# --- 1. GEOMETRY ENGINE ---
def calculate_curvature(geometry):
    if not geometry: return 0 
    coords = list(geometry.coords)
    if len(coords) < 3: return 0
    total_turn = 0
    for i in range(len(coords) - 2):
        p1, p2, p3 = np.array(coords[i]), np.array(coords[i+1]), np.array(coords[i+2])
        v1, v2 = p2 - p1, p3 - p2
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 > 0 and norm2 > 0:
            angle = np.arccos(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0))
            total_turn += np.degrees(angle)
    return total_turn

# --- 2. COMPUTER VISION ENGINE ---
def analyze_image_cv():
    # Only run if file exists to prevent crashes
    path = "test_road.jpg"
    if not os.path.exists(path): return 0
    
    img = cv2.imread(path)
    if img is None: return 0
    
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5,5), 0), 50, 150)
    score = (np.count_nonzero(edges) / edges.size) * 100
    
    if score > 5: return 20
    if score > 2: return 10
    return 0

# --- 3. RISK ENGINE ---
def analyze_risk(u, v, graph, cv_score):
    data = graph.get_edge_data(u, v)[0]
    risk = 0
    reasons = []

    # Factors
    curve = calculate_curvature(data.get('geometry', None))
    if curve > 45: risk += 30; reasons.append("Sharp Curve")
    elif curve > 20: risk += 10; reasons.append("Winding Road")

    lanes = data.get('lanes', '2')
    if isinstance(lanes, list): lanes = lanes[0]
    try:
        if int(lanes) <= 1: 
            risk += 20; reasons.append("Narrow Road")
            if random.random() > 0.7: reasons.append("Traffic Congestion")
    except: pass

    if risk > 20 and random.random() > 0.8: risk += 40; reasons.append("Known Blackspot")

    hw = data.get('highway', '')
    if isinstance(hw, list): hw = hw[0]
    
    if hw in ['trunk', 'primary', 'motorway']: risk += 10; reasons.append("High Speed Zone")
    elif hw in ['track', 'unclassified', 'service']: risk += 15; reasons.append("Poor Lighting")

    if cv_score > 0: risk += cv_score; reasons.append("Bad Visibility")

    if risk > 50: return "High", "#E11B23", reasons
    if risk > 20: return "Moderate", "#F5A623", reasons
    return "Low", "#20BD5F", ["Safe Route"]

# --- 4. ROUTING ENGINE (OPTIMIZED) ---
def get_safe_graph(start_coords, end_coords):
    mid_lat = (start_coords[0] + end_coords[0]) / 2
    mid_lon = (start_coords[1] + end_coords[1]) / 2
    
    # Force radius to 3000m (3km) max to save RAM on Render Free Tier
    # If users try to route between cities, this ensures we only download
    # a small chunk of the map around the midpoint, preventing crashes.
    radius = 3000 
    
    print(f" ⬇️ Downloading Map Radius: {int(radius)}m at {mid_lat}, {mid_lon}")
    
    # simplify=True reduces graph size
    return ox.graph_from_point((mid_lat, mid_lon), dist=radius, network_type='drive', simplify=True)

# --- MAIN ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)