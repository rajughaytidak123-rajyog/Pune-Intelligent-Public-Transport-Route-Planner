# Pune Intelligent Public Transport Route Planner — Dataset

**MSc Mini-Project | A* Search Algorithm | Python + Streamlit**  
Generated: 2026-11-12 19:13:03

---

## Dataset Summary

| File | Description | Records | Source Status |
|------|-------------|---------|---------------|
| `pmpml_stops.csv` | Bus stop IDs, names, coordinates, zones | 150 stops | ESTIMATED |
| `pmpml_routes.csv` | Bus route IDs, names, types | 25 routes | ESTIMATED |
| `pmpml_trips.csv` | Trip IDs linked to routes | 50 trips | ESTIMATED |
| `pmpml_stop_times.csv` | Stop sequences + arrival times | 298 entries | ESTIMATED |
| `pune_metro_stations.csv` | Metro stations (Purple + Aqua) | 31 stations | **VERIFIED** |
| `bus_metro_interchanges.csv` | Bus-metro walking connections | 136 pairs | COMPUTED |
| `pune_city_boundary.geojson` | PMC+PCMC boundary polygon | 1 polygon | APPROXIMATE |
| `pune_transport_network.geojson` | Full network as GeoJSON | 344 features | MIXED |
| `pune_pmpml_gtfs.zip` | Valid GTFS package | 8 files | ESTIMATED |
| `dataset_validation_report.txt` | Validation + A* test results | - | - |

---

## Official Data Sources

| Component | Official Source | URL |
|-----------|----------------|-----|
| PMPML (community) | croyla/pmpml-gtfs | https://github.com/croyla/pmpml-gtfs |
| Pune Metro | punemetrorail.org | https://www.punemetrorail.org |
| Metro data | Wikipedia | https://en.wikipedia.org/wiki/Pune_Metro |
| Metro coords | bharatmetrolocator.com | https://bharatmetrolocator.com |
| Bus routes | citybuses.in | https://citybuses.in |
| Boundary | DataMeet | https://github.com/datameet/Pune_wards |
| Open data | OpenCity | https://data.opencity.in |

> **IMPORTANT**: No official PMPML GTFS feed is publicly downloadable.
> All bus schedule times are ESTIMATED. Metro station data is VERIFIED.

---

## Metro Network

### Purple Line (VERIFIED — Operational)
PCMC Bhavan → Sant Tukaram Nagar → Bhosari → Kasarwadi → Phugewadi →
Dapodi → Bopodi → Khadki → Range Hill → Shivajinagar → **District Court** →
Kasba Peth → Mandai → Swargate (14 stations, ~16.6 km)

### Aqua Line (VERIFIED — Operational)
Vanaz → Anand Nagar → Ideal Colony → Nal Stop → SNDT College →
Garware College → Deccan Gymkhana → Sambhaji Udyan → PMC →
**District Court** → RTO Pune → Pune Railway Station → Ruby Hall Clinic →
Bund Garden → Yerawada → Kalyani Nagar → Ramwadi (17 stations, ~14.7 km)

**District Court** = interchange between Purple Line & Aqua Line

---

## Integrating with Your Streamlit app.py

### 1. Load Bus Stops
```python
import pandas as pd
stops_df = pd.read_csv("Pune_Transport_Dataset/pmpml_stops.csv")
```

### 2. Load Metro Stations
```python
metro_df = pd.read_csv("Pune_Transport_Dataset/pune_metro_stations.csv")
```

### 3. Load Interchanges
```python
ix_df = pd.read_csv("Pune_Transport_Dataset/bus_metro_interchanges.csv")
```

### 4. Upload GTFS ZIP
In your Streamlit GTFS upload section, upload: `pune_pmpml_gtfs.zip`

### 5. Display Network Map (Folium/PyDeck)
```python
import folium, json

m = folium.Map(location=[18.5204, 73.8567], zoom_start=12)
with open("Pune_Transport_Dataset/pune_transport_network.geojson") as f:
    network = json.load(f)

for feature in network["features"]:
    if feature["properties"].get("type") == "bus_stop":
        coords = feature["geometry"]["coordinates"]
        folium.CircleMarker([coords[1], coords[0]], radius=4,
            color='orange', popup=feature["properties"]["name"]).add_to(m)
    elif feature["properties"].get("type") == "metro_station":
        coords = feature["geometry"]["coordinates"]
        line = feature["properties"].get("line","")
        color = "purple" if "Purple" in line else "blue"
        folium.Marker([coords[1], coords[0]],
            icon=folium.Icon(color=color, icon="train", prefix="fa"),
            popup=feature["properties"]["name"]).add_to(m)
```

### 6. A* Graph Construction
```python
import math, heapq

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((lat2-lat1)/2*math.pi/180)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin((lon2-lon1)/2*math.pi/180)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# Build graph from stops + routes + metro + interchanges
# See dataset_validation_report.txt for A* route results
```

---

## A* Algorithm Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Heuristic | Haversine distance / 60 km/h | Admissible upper-bound speed |
| Bus avg speed | 20 km/h | City traffic estimate |
| Metro avg speed | 35 km/h | Incl. station dwell time |
| Walk speed | 1.2 m/s (~4.3 km/h) | Standard pedestrian speed |
| Transfer penalty | 5 minutes | Per mode change |

---

## Sample Route: Kothrud → Ramwadi

| Mode | Time | Distance | Notes |
|------|------|----------|-------|
| Bus Only | ~65 min | ~18 km | Via Deccan → Pune Station → Yerawada |
| Metro Only | ~35 min | ~15 km | Walk to Vanaz → Aqua Line → Ramwadi |
| Combined | ~45 min | ~16 km | Bus to Vanaz metro → Aqua Line → Ramwadi |

*All times are ESTIMATED based on average speeds, not official schedules.*
