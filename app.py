import streamlit as st
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import math

# ============================================================
# PUNE INTELLIGENT PUBLIC TRANSPORT ROUTE PLANNER
# BUS + METRO INTEGRATION USING A* ALGORITHM
# ============================================================

st.set_page_config(
    page_title="Pune Transit AI",
    page_icon="🚇",
    layout="wide"
)

# ============================================================
# DASHBOARD DESIGN
# ============================================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #111c31;
}
div[data-testid="stMetric"] {
    background-color: #172741;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #294466;
}
</style>
""", unsafe_allow_html=True)

st.title("🚇 Pune Intelligent Public Transport Route Planner")
st.caption(
    "PMPML Bus + Pune Metro | A* Search Algorithm | Multimodal Dashboard"
)

# ============================================================
# FILE UPLOAD
# ============================================================

st.sidebar.header("📂 Upload Dataset")

uploaded_files = st.sidebar.file_uploader(
    "Upload all six CSV files",
    type=["csv"],
    accept_multiple_files=True
)

required_files = {
    "pmpml_stops.csv",
    "pmpml_routes.csv",
    "pmpml_trips.csv",
    "pmpml_stop_times.csv",
    "pune_metro_stations.csv",
    "bus_metro_interchanges.csv"
}

if not uploaded_files:
    st.info(
        "Upload the six CSV dataset files from the sidebar to start the dashboard."
    )
    st.stop()

data = {
    file.name.lower(): pd.read_csv(file)
    for file in uploaded_files
}

missing_files = required_files - set(data.keys())

if missing_files:
    st.error(
        "Missing files: " + ", ".join(sorted(missing_files))
    )
    st.stop()

stops = data["pmpml_stops.csv"].copy()
routes = data["pmpml_routes.csv"].copy()
trips = data["pmpml_trips.csv"].copy()
stop_times = data["pmpml_stop_times.csv"].copy()
metro = data["pune_metro_stations.csv"].copy()
interchanges = data["bus_metro_interchanges.csv"].copy()

# ============================================================
# DATA CLEANING
# ============================================================

for col in ["stop_lat", "stop_lon"]:
    stops[col] = pd.to_numeric(
        stops[col], errors="coerce"
    )

for col in [
    "latitude",
    "longitude",
    "sequence_number",
    "estimated_travel_time_minutes"
]:
    metro[col] = pd.to_numeric(
        metro[col], errors="coerce"
    )

stop_times["stop_sequence"] = pd.to_numeric(
    stop_times["stop_sequence"], errors="coerce"
)

for col in [
    "walking_distance_m",
    "estimated_walking_time_minutes"
]:
    interchanges[col] = pd.to_numeric(
        interchanges[col], errors="coerce"
    )

stops = stops.dropna(
    subset=["stop_lat", "stop_lon"]
)

metro = metro.dropna(
    subset=["latitude", "longitude", "sequence_number"]
)

stop_times = stop_times.dropna(
    subset=["stop_sequence"]
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate geographical distance in kilometers.
    """

    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 6371 * 2 * math.asin(
        min(1, math.sqrt(a))
    )


def time_to_seconds(value):
    """
    Convert HH:MM:SS to seconds.
    """

    try:
        h, m, s = map(int, str(value).split(":"))
        return h * 3600 + m * 60 + s

    except (ValueError, TypeError):
        return None


def add_connection(
    graph,
    node_a,
    node_b,
    minutes,
    distance,
    mode
):
    """
    Add a connection to the graph.
    """

    if (
        pd.isna(minutes)
        or float(minutes) <= 0
        or not math.isfinite(float(minutes))
    ):
        return

    graph.add_edge(
        node_a,
        node_b,
        weight=float(minutes),
        distance=max(0.0, float(distance)),
        mode=mode
    )


# ============================================================
# BUILD CONNECTED BUS + METRO GRAPH
# ============================================================

def build_transport_graph(
    stops,
    stop_times,
    trips,
    metro,
    interchanges,
    transport
):

    G = nx.DiGraph()

    bus_nodes = {}
    metro_nodes = {}

    # --------------------------------------------------------
    # 1. ADD BUS STOPS
    # --------------------------------------------------------

    if transport in ["Bus Only", "Bus + Metro"]:

        for _, row in stops.iterrows():

            stop_id = str(row["stop_id"])

            node = ("BUS", stop_id)

            bus_nodes[stop_id] = node

            G.add_node(
                node,
                name=str(row["stop_name"]),
                lat=float(row["stop_lat"]),
                lon=float(row["stop_lon"]),
                mode="Bus"
            )

    # --------------------------------------------------------
    # 2. CONNECT BUS STOPS USING BUS TRIPS
    # --------------------------------------------------------

    if transport in ["Bus Only", "Bus + Metro"]:

        valid_trip_ids = set(
            trips["trip_id"].astype(str)
        )

        bus_times = stop_times[
            stop_times["trip_id"].astype(str).isin(
                valid_trip_ids
            )
        ].copy()

        bus_times["trip_id"] = bus_times["trip_id"].astype(str)
        bus_times["stop_id"] = bus_times["stop_id"].astype(str)

        bus_times = bus_times[
            bus_times["stop_id"].isin(bus_nodes)
        ]

        for trip_id, trip_df in bus_times.groupby(
            "trip_id",
            sort=False
        ):

            trip_df = trip_df.sort_values(
                "stop_sequence"
            )

            records = trip_df.to_dict("records")

            for first, second in zip(
                records[:-1],
                records[1:]
            ):

                id_a = str(first["stop_id"])
                id_b = str(second["stop_id"])

                if id_a == id_b:
                    continue

                node_a = bus_nodes[id_a]
                node_b = bus_nodes[id_b]

                lat1 = G.nodes[node_a]["lat"]
                lon1 = G.nodes[node_a]["lon"]

                lat2 = G.nodes[node_b]["lat"]
                lon2 = G.nodes[node_b]["lon"]

                distance = haversine(
                    lat1, lon1,
                    lat2, lon2
                )

                departure = time_to_seconds(
                    first.get("departure_time")
                )

                arrival = time_to_seconds(
                    second.get("arrival_time")
                )

                if (
                    departure is not None
                    and arrival is not None
                    and arrival > departure
                ):

                    travel_time = (
                        arrival - departure
                    ) / 60

                else:

                    # Estimated bus speed: 17 km/h
                    travel_time = max(
                        2,
                        (distance / 17) * 60
                    )

                add_connection(
                    G,
                    node_a,
                    node_b,
                    travel_time,
                    distance,
                    "Bus"
                )

    # --------------------------------------------------------
    # 3. ADD METRO STATIONS
    # --------------------------------------------------------

    if transport in ["Metro Only", "Bus + Metro"]:

        for _, row in metro.iterrows():

            station_id = str(row["station_id"])

            node = ("METRO", station_id)

            metro_nodes[station_id] = node

            G.add_node(
                node,
                name=str(row["station_name"]),
                lat=float(row["latitude"]),
                lon=float(row["longitude"]),
                mode="Metro",
                line=str(row["line_name"])
            )

    # --------------------------------------------------------
    # 4. CONNECT METRO STATIONS IN BOTH DIRECTIONS
    # --------------------------------------------------------

    if transport in ["Metro Only", "Bus + Metro"]:

        for line_name, line_df in metro.groupby(
            "line_name",
            sort=False
        ):

            line_df = line_df.sort_values(
                "sequence_number"
            )

            records = line_df.to_dict("records")

            for first, second in zip(
                records[:-1],
                records[1:]
            ):

                id_a = str(first["station_id"])
                id_b = str(second["station_id"])

                node_a = metro_nodes[id_a]
                node_b = metro_nodes[id_b]

                distance = haversine(
                    first["latitude"],
                    first["longitude"],
                    second["latitude"],
                    second["longitude"]
                )

                travel_time = first.get(
                    "estimated_travel_time_minutes",
                    4
                )

                if (
                    pd.isna(travel_time)
                    or float(travel_time) <= 0
                ):

                    # Estimated metro speed: 30 km/h
                    travel_time = max(
                        2,
                        (distance / 30) * 60
                    )

                # Forward direction
                add_connection(
                    G,
                    node_a,
                    node_b,
                    travel_time,
                    distance,
                    "Metro"
                )

                # Reverse direction
                add_connection(
                    G,
                    node_b,
                    node_a,
                    travel_time,
                    distance,
                    "Metro"
                )

    # --------------------------------------------------------
    # 5. METRO INTERCHANGE STATIONS
    # --------------------------------------------------------

    if transport in ["Metro Only", "Bus + Metro"]:

        stations_by_name = {}

        for node, attrs in G.nodes(data=True):

            if node[0] == "METRO":

                station_name = attrs["name"].strip().lower()

                stations_by_name.setdefault(
                    station_name,
                    []
                ).append(node)

        for station_list in stations_by_name.values():

            if len(station_list) > 1:

                for node_a in station_list:

                    for node_b in station_list:

                        if node_a != node_b:

                            add_connection(
                                G,
                                node_a,
                                node_b,
                                2,
                                0,
                                "Metro Transfer"
                            )

    # --------------------------------------------------------
    # 6. BUS-METRO TRANSFERS USING PROVIDED INTERCHANGES
    # --------------------------------------------------------

    if transport == "Bus + Metro":

        for _, row in interchanges.iterrows():

            bus_id = str(row["bus_stop_id"])
            metro_id = str(row["metro_station_id"])

            if (
                bus_id not in bus_nodes
                or metro_id not in metro_nodes
            ):
                continue

            bus_node = bus_nodes[bus_id]
            metro_node = metro_nodes[metro_id]

            distance = row.get(
                "walking_distance_m",
                500
            )

            if pd.isna(distance):
                distance = 500

            distance = float(distance) / 1000

            walk_time = row.get(
                "estimated_walking_time_minutes",
                5
            )

            if pd.isna(walk_time):
                walk_time = (distance / 4.5) * 60

            walk_time = float(walk_time) + 2

            # Bus stop to metro station
            add_connection(
                G,
                bus_node,
                metro_node,
                walk_time,
                distance,
                "Walk / Transfer"
            )

            # Metro station to bus stop
            add_connection(
                G,
                metro_node,
                bus_node,
                walk_time,
                distance,
                "Walk / Transfer"
            )

    # --------------------------------------------------------
    # 7. AUTOMATIC NEAREST METRO CONNECTION FOR EACH BUS STOP
    # --------------------------------------------------------

    if transport == "Bus + Metro":

        metro_station_list = list(
            metro_nodes.values()
        )

        for bus_node in bus_nodes.values():

            bus_lat = G.nodes[bus_node]["lat"]
            bus_lon = G.nodes[bus_node]["lon"]

            nearest_station = None
            shortest_distance = float("inf")

            for metro_node in metro_station_list:

                metro_lat = G.nodes[metro_node]["lat"]
                metro_lon = G.nodes[metro_node]["lon"]

                distance = haversine(
                    bus_lat,
                    bus_lon,
                    metro_lat,
                    metro_lon
                )

                if distance < shortest_distance:

                    shortest_distance = distance
                    nearest_station = metro_node

            if nearest_station is not None:

                # Walking speed: 4.5 km/h
                walking_time = (
                    shortest_distance / 4.5
                ) * 60

                # Add a small transfer buffer
                walking_time += 3

                # Automatic bus to nearest metro
                add_connection(
                    G,
                    bus_node,
                    nearest_station,
                    walking_time,
                    shortest_distance,
                    "Walk / Nearest Metro Transfer"
                )

                # Automatic metro to nearest bus stop
                add_connection(
                    G,
                    nearest_station,
                    bus_node,
                    walking_time,
                    shortest_distance,
                    "Walk / Nearest Metro Transfer"
                )

    return G


# ============================================================
# A* SEARCH
# ============================================================

def find_astar_route(G, start_node, end_node):

    def heuristic(node, target):

        lat1 = G.nodes[node]["lat"]
        lon1 = G.nodes[node]["lon"]

        lat2 = G.nodes[target]["lat"]
        lon2 = G.nodes[target]["lon"]

        distance = haversine(
            lat1, lon1,
            lat2, lon2
        )

        # Lower-bound heuristic in minutes.
        # Assumes maximum speed of 60 km/h.
        return distance

    try:

        path = nx.astar_path(
            G,
            start_node,
            end_node,
            heuristic=heuristic,
            weight="weight"
        )

        total_time = sum(
            G[a][b]["weight"]
            for a, b in zip(
                path[:-1],
                path[1:]
            )
        )

        total_distance = sum(
            G[a][b]["distance"]
            for a, b in zip(
                path[:-1],
                path[1:]
            )
        )

        return path, total_time, total_distance

    except nx.NetworkXNoPath:

        return None, 0, 0

    except nx.NodeNotFound:

        return None, 0, 0


# ============================================================
# ROUTE TABLE
# ============================================================

def create_route_table(G, path):

    rows = []

    elapsed_time = 0

    for i, node in enumerate(path):

        if i < len(path) - 1:

            next_node = path[i + 1]

            edge = G[node][next_node]

            travel_time = edge["weight"]
            distance = edge["distance"]
            transport_mode = edge["mode"]

            elapsed_time += travel_time

        else:

            travel_time = 0
            distance = 0
            transport_mode = "Destination"

        rows.append({

            "Step": i + 1,

            "Location": G.nodes[node]["name"],

            "Transport": G.nodes[node]["mode"],

            "Next Connection": transport_mode,

            "Travel Time (min)": round(
                travel_time,
                2
            ),

            "Distance (km)": round(
                distance,
                2
            ),

            "Elapsed Time (min)": round(
                elapsed_time,
                2
            )

        })

    return pd.DataFrame(rows)


# ============================================================
# ROUTE MAP
# ============================================================

def create_route_map(G, path=None):

    fig = go.Figure()

    # BUS STOP LOCATIONS
    bus_nodes = [
        node for node in G.nodes
        if node[0] == "BUS"
    ]

    if bus_nodes:

        fig.add_trace(
            go.Scattermap(
                lat=[
                    G.nodes[node]["lat"]
                    for node in bus_nodes
                ],

                lon=[
                    G.nodes[node]["lon"]
                    for node in bus_nodes
                ],

                mode="markers",

                marker=dict(
                    size=6,
                    color="#00BFFF"
                ),

                text=[
                    G.nodes[node]["name"]
                    for node in bus_nodes
                ],

                name="Bus Stops"
            )
        )

    # METRO STATIONS
    metro_nodes = [
        node for node in G.nodes
        if node[0] == "METRO"
    ]

    if metro_nodes:

        fig.add_trace(
            go.Scattermap(
                lat=[
                    G.nodes[node]["lat"]
                    for node in metro_nodes
                ],

                lon=[
                    G.nodes[node]["lon"]
                    for node in metro_nodes
                ],

                mode="markers",

                marker=dict(
                    size=10,
                    color="#A855F7"
                ),

                text=[
                    G.nodes[node]["name"]
                    for node in metro_nodes
                ],

                name="Metro Stations"
            )
        )

    # SELECTED ROUTE
    if path:

        route_lat = [
            G.nodes[node]["lat"]
            for node in path
        ]

        route_lon = [
            G.nodes[node]["lon"]
            for node in path
        ]

        route_text = [
            G.nodes[node]["name"]
            for node in path
        ]

        fig.add_trace(
            go.Scattermap(

                lat=route_lat,

                lon=route_lon,

                mode="lines+markers",

                marker=dict(
                    size=11,
                    color="#FF9F43"
                ),

                line=dict(
                    width=5,
                    color="#FF5733"
                ),

                text=route_text,

                name="Selected Journey"

            )
        )

        # START
        fig.add_trace(
            go.Scattermap(

                lat=[route_lat[0]],

                lon=[route_lon[0]],

                mode="markers+text",

                marker=dict(
                    size=15,
                    color="green"
                ),

                text=["START"],

                name="Start"

            )
        )

        # DESTINATION
        fig.add_trace(
            go.Scattermap(

                lat=[route_lat[-1]],

                lon=[route_lon[-1]],

                mode="markers+text",

                marker=dict(
                    size=15,
                    color="red"
                ),

                text=["DESTINATION"],

                name="Destination"

            )
        )

        center_lat = sum(route_lat) / len(route_lat)
        center_lon = sum(route_lon) / len(route_lon)

    else:

        center_lat = 18.52
        center_lon = 73.85

    fig.update_layout(

        map_style="open-street-map",

        map_zoom=10,

        map_center={
            "lat": center_lat,
            "lon": center_lon
        },

        height=550,

        template="plotly_dark",

        margin={
            "l": 0,
            "r": 0,
            "t": 10,
            "b": 0
        }

    )

    return fig


# ============================================================
# DASHBOARD METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "🚌 Bus Stops",
    len(stops)
)

c2.metric(
    "🛣️ Bus Routes",
    routes["route_id"].nunique()
)

c3.metric(
    "🚍 Bus Trips",
    trips["trip_id"].nunique()
)

c4.metric(
    "🚇 Metro Stations",
    len(metro)
)

st.divider()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

page = st.sidebar.radio(

    "Dashboard Navigation",

    [
        "Dashboard Overview",
        "Plan Journey",
        "Network Analytics",
        "Dataset Explorer",
        "About Project"
    ]

)

# ============================================================
# DASHBOARD OVERVIEW
# ============================================================

if page == "Dashboard Overview":

    st.header("📊 Pune Transport Network Overview")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Metro Stations by Line")

        line_data = metro[
            "line_name"
        ].value_counts().reset_index()

        line_data.columns = [
            "Metro Line",
            "Stations"
        ]

        fig = px.bar(
            line_data,
            x="Metro Line",
            y="Stations",
            color="Metro Line"
        )

        fig.update_layout(
            template="plotly_dark",
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("Bus Stops by Zone")

        if "zone_id" in stops.columns:

            zone_data = stops[
                "zone_id"
            ].value_counts().reset_index()

            zone_data.columns = [
                "Zone",
                "Bus Stops"
            ]

            fig = px.pie(
                zone_data,
                names="Zone",
                values="Bus Stops",
                hole=0.45
            )

            fig.update_layout(
                template="plotly_dark",
                height=400
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # Metro map
    st.subheader("🗺️ Pune Metro Map")

    fig = px.scatter_map(
        metro,
        lat="latitude",
        lon="longitude",
        color="line_name",
        hover_name="station_name",
        zoom=10,
        height=500
    )

    fig.update_layout(
        map_style="open-street-map",
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Bus map
    st.subheader("🚌 Pune PMPML Bus Stops Map")

    fig = px.scatter_map(
        stops,
        lat="stop_lat",
        lon="stop_lon",
        hover_name="stop_name",
        zoom=10,
        height=500
    )

    fig.update_layout(
        map_style="open-street-map",
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# PLAN JOURNEY PAGE
# ============================================================

elif page == "Plan Journey":

    st.header("🧭 Intelligent Bus + Metro Journey Planner")

    transport = st.selectbox(

        "Select Transport Mode",

        [
            "Bus + Metro",
            "Bus Only",
            "Metro Only"
        ],

        index=0

    )

    st.info(
        "Bus + Metro mode automatically connects every bus stop "
        "to its nearest metro station using a walking-transfer edge. "
        "A* can choose bus, metro, or both based on estimated travel time."
    )

    with st.spinner(
        "Building connected bus and metro network..."
    ):

        G = build_transport_graph(

            stops,

            stop_times,

            trips,

            metro,

            interchanges,

            transport

        )

    st.success(
        f"Network ready: {G.number_of_nodes():,} locations and "
        f"{G.number_of_edges():,} connections."
    )

    node_labels = {}

    for node, attrs in G.nodes(data=True):

        label = (
            attrs["name"]
            + " | "
            + attrs["mode"]
            + " | "
            + str(node[1])
        )

        node_labels[label] = node

    location_names = list(
        node_labels.keys()
    )

    col1, col2 = st.columns(2)

    with col1:

        start = st.selectbox(
            "📍 Starting Location",
            location_names,
            key="origin"
        )

    with col2:

        destination = st.selectbox(
            "🏁 Destination",
            location_names,
            key="destination"
        )

    if st.button(
        "🔍 FIND BEST JOURNEY",
        use_container_width=True
    ):

        start_node = node_labels[start]

        end_node = node_labels[destination]

        if start_node == end_node:

            st.warning(
                "Select two different locations."
            )

        else:

            with st.spinner(
                "A* is finding the best multimodal route..."
            ):

                path, total_time, total_distance = find_astar_route(
                    G,
                    start_node,
                    end_node
                )

            if path is None:

                st.error(
                    "No connected route found. "
                    "Try a different starting point or destination."
                )

            else:

                st.session_state["route_path"] = path

                st.session_state["route_graph"] = G

                st.session_state["route_time"] = total_time

                st.session_state["route_distance"] = total_distance

    # --------------------------------------------------------
    # DISPLAY ROUTE
    # --------------------------------------------------------

    if "route_path" in st.session_state:

        path = st.session_state["route_path"]

        route_graph = st.session_state["route_graph"]

        total_time = st.session_state["route_time"]

        total_distance = st.session_state["route_distance"]

        route_table = create_route_table(
            route_graph,
            path
        )

        edges = [
            route_graph[a][b]
            for a, b in zip(
                path[:-1],
                path[1:]
            )
        ]

        bus_count = sum(
            edge["mode"] == "Bus"
            for edge in edges
        )

        metro_count = sum(
            edge["mode"] == "Metro"
            for edge in edges
        )

        transfer_count = sum(
            "Transfer" in edge["mode"]
            for edge in edges
        )

        st.success(
            "🎉 Best connected route found!"
        )

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "⏱️ Estimated Travel Time",
            f"{total_time:.1f} min"
        )

        m2.metric(
            "📏 Estimated Distance",
            f"{total_distance:.2f} km"
        )

        m3.metric(
            "🚌 Bus Connections",
            bus_count
        )

        m4.metric(
            "🚇 Metro Connections",
            metro_count
        )

        st.subheader(
            "🗺️ Complete Journey Map"
        )

        st.caption(
            "Blue = Bus Stops | Purple = Metro Stations | "
            "Orange = Selected Journey"
        )

        fig = create_route_map(
            route_graph,
            path
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ROUTE DETAILS
        col1, col2 = st.columns([1.4, 1])

        with col1:

            st.subheader(
                "🛣️ Journey Steps"
            )

            st.dataframe(
                route_table,
                use_container_width=True,
                hide_index=True
            )

        with col2:

            st.subheader(
                "📊 Transport Breakdown"
            )

            mode_counts = pd.Series(
                [
                    edge["mode"]
                    for edge in edges
                ]
            ).value_counts()

            if not mode_counts.empty:

                fig = px.pie(
                    values=mode_counts.values,
                    names=mode_counts.index,
                    hole=0.5
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=400
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # A* EXPLANATION
        st.subheader(
            "🧠 A* Algorithm Explanation"
        )

        st.latex(
            r"f(n)=g(n)+h(n)"
        )

        st.write(
            "**g(n):** Estimated travel time already accumulated "
            "from the starting location."
        )

        st.write(
            "**h(n):** Straight-line distance to the destination "
            "used as a conservative estimated time in minutes."
        )

        st.write(
            "**f(n):** Combined cost used by A* to decide "
            "which location to explore next."
        )

        st.write(
            "**Transfer behavior:** When a bus stop is close to a metro "
            "station, the graph contains a walking-transfer edge. "
            "A* can use this connection to switch from bus to metro."
        )

        st.download_button(
            "⬇️ Download Journey Report",
            route_table.to_csv(index=False),
            file_name="pune_multimodal_route.csv",
            mime="text/csv"
        )

        st.warning(
            "Travel times are estimated from the provided data. "
            "The planner does not use live traffic, live vehicle locations, "
            "or actual timetable waiting times."
        )


# ============================================================
# NETWORK ANALYTICS
# ============================================================

elif page == "Network Analytics":

    st.header("📈 Network Analytics")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Metro Line Distribution")

        metro_count = metro[
            "line_name"
        ].value_counts().reset_index()

        metro_count.columns = [
            "Line",
            "Station Count"
        ]

        fig = px.bar(
            metro_count,
            x="Line",
            y="Station Count",
            color="Line"
        )

        fig.update_layout(
            template="plotly_dark",
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("Bus Routes Distribution")

        if "route_short_name" in routes.columns:

            route_count = routes[
                "route_short_name"
            ].value_counts().reset_index()

            route_count.columns = [
                "Route",
                "Count"
            ]

            fig = px.bar(
                route_count,
                x="Route",
                y="Count",
                color="Route"
            )

            fig.update_layout(
                template="plotly_dark",
                height=400,
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.subheader("Bus Stop Zone Distribution")

    if "zone_id" in stops.columns:

        zone_count = stops[
            "zone_id"
        ].value_counts().reset_index()

        zone_count.columns = [
            "Zone",
            "Bus Stops"
        ]

        fig = px.bar(
            zone_count,
            x="Zone",
            y="Bus Stops",
            color="Zone"
        )

        fig.update_layout(
            template="plotly_dark",
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# DATASET EXPLORER
# ============================================================

elif page == "Dataset Explorer":

    st.header("📂 Dataset Explorer")

    selected_dataset = st.selectbox(
        "Select Dataset",
        list(data.keys())
    )

    df = data[selected_dataset]

    st.metric(
        "Total Rows",
        len(df)
    )

    st.metric(
        "Total Columns",
        len(df.columns)
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Statistical Summary"
    )

    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "About Project":

    st.header("📚 About the Project")

    st.markdown("""
    ## Pune Intelligent Public Transport Route Planner

    The project integrates PMPML bus and Pune Metro networks to
    identify efficient public transportation routes using A* search.

    ### Objectives

    1. Visualize Pune public transport infrastructure.
    2. Connect PMPML bus stops and Metro stations.
    3. Automatically link bus stops to their nearest metro station.
    4. Implement multimodal route planning using A*.
    5. Estimate journey time and distance.
    6. Visualize the journey using an interactive map and charts.

    ### Technologies

    - Python
    - Streamlit
    - Pandas
    - NetworkX
    - Plotly
    - A* Search Algorithm

    ### How the multimodal routing works

    A bus stop is connected to its nearest metro station by a walking edge.
    A* explores the bus and metro connections and chooses a route with
    the lowest modeled travel-time cost.

    **Note:** This is an academic prototype. The bus data and travel times
    may be estimated and do not represent live public transportation data.
    """)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Pune Transit AI | Bus + Metro Integration | A* Search | MSc Mini Project"
)