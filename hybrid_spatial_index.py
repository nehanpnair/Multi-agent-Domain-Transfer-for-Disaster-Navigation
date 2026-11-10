"""
hybrid_spatial_index.py
------------------------
Hybrid spatial indexing for disaster-aware navigation.

Architecture
------------
  Step 1  R-Tree  — spatial filtering
          Index all edges by bounding box. Given a disaster epicentre +
          radius, retrieve only affected edges in O(log n) time.

  Step 2  KD-Tree — nearest-neighbour snapping
          Built over road nodes. For each hospital / fire_station /
          building, find nearest road nodes, then walk their adjacency
          lists to find the geometrically closest edge.

  Step 3  Graph construction
          Merge the original graph with new facility_connector edges.
          Weights = haversine distance. Disaster-disabled edges get
          weight = infinity.

  Step 4  Disaster integration
          R-Tree re-queried to mark affected edges as disabled or
          high-cost in O(log n).

  Step 5  Routing
          Dijkstra (distance) and A* (travel-time) on the hybrid graph.
          Fallback staging-zone logic when no path exists.

Drop-in replacement for dataset_connector.py — same public API plus
hybrid index methods.

Usage
-----
    from hybrid_spatial_index import HybridTokyoGraph

    graph = HybridTokyoGraph("tokyo_graph.json")
    graph.build_index()          # builds R-tree + KD-tree + snaps facilities

    disaster = graph.apply_disaster("earthquake", (35.68, 139.76), 2000, 0.8)
    path = graph.shortest_path_astar(src_id, dst_id)
    staging = graph.find_staging_zones((35.68, 139.76), 2000)
"""

import json
import math
import heapq
import time
import numpy as np
from collections import defaultdict
from typing import Optional
from scipy.spatial import KDTree
from rtree import index as rtree_index


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def _point_to_segment_m(plat, plon, lat1, lon1, lat2, lon2) -> float:
    """Perpendicular distance from point to line segment (metres)."""
    dx = lon2 - lon1
    dy = lat2 - lat1
    if dx == 0 and dy == 0:
        return _haversine_m(plat, plon, lat1, lon1)
    t = ((plon - lon1) * dx + (plat - lat1) * dy) / (dx*dx + dy*dy)
    t = max(0.0, min(1.0, t))
    return _haversine_m(plat, plon, lat1 + t*dy, lon1 + t*dx)


# ---------------------------------------------------------------------------
# Road speed table (from RTree.ipynb)
# ---------------------------------------------------------------------------

_ROAD_CLASS = {
    "motorway": "highway", "motorway_link": "highway",
    "trunk": "highway",    "trunk_link":    "highway",
    "primary": "major",    "primary_link":  "major",
    "secondary": "major",  "secondary_link":"major",
    "tertiary": "minor",   "tertiary_link": "minor",
    "residential": "local","unclassified":  "local",
    "service": "local",    "facility_connector": "local",
    "connector_node": "local", "connector_edge": "local",
}
_SPEED_MS = {"highway": 25/3.6, "major": 16/3.6, "minor": 11/3.6, "local": 7/3.6}


def _travel_time(length_m: float, road_type: str) -> float:
    cls = _ROAD_CLASS.get(road_type, "local")
    return length_m / _SPEED_MS[cls]


# ---------------------------------------------------------------------------
# HybridTokyoGraph
# ---------------------------------------------------------------------------

class HybridTokyoGraph:
    """
    Tokyo urban graph with hybrid R-tree + KD-tree spatial indexing.

    Public API (same as dataset_connector.TokyoGraph plus extras)
    -------------------------------------------------------------
    build_index()                     — build R-tree + KD-tree, snap facilities
    apply_disaster(type, centre, r, s)— mark edges/nodes damaged via R-tree
    reset_disaster()
    shortest_path(from_id, to_id)     — Dijkstra (distance)
    shortest_path_astar(from_id, to_id) — A* (travel time, faster)
    find_staging_zones(centre, r)     — fallback when no path exists
    nearest_facility(type, lat, lon)
    nearest_road_node(lat, lon)
    get_nodes_in_radius(lat, lon, r)
    get_population_in_radius(lat, lon, r)
    snapshot_local(lat, lon, r)
    get_hospitals() / get_fire_stations() / get_buildings()
    is_edge_passable(u, v)
    get_node_damage(node_id)
    path_distance_m(path)
    """

    INF_WEIGHT = 1e9   # weight for disaster-disabled edges

    def __init__(self, json_path: str):
        print(f"Loading {json_path} ...")
        t0 = time.time()
        with open(json_path) as f:
            raw = json.load(f)

        self.city: str = raw.get("city", "Tokyo")
        self._nodes: dict[int, dict] = {}
        self._adj: dict[int, list] = defaultdict(list)
        self._disabled_edges: set = set()
        self._node_damage: dict[int, float] = {}

        # Spatial index structures (built lazily by build_index())
        self._rtree: Optional[rtree_index.Index] = None
        self._rtree_edges: list = []          # edge list parallel to R-tree
        self._kdtree: Optional[KDTree] = None
        self._kd_road_nodes: list = []        # road nodes parallel to KD-tree
        self._index_built = False

        for node in raw["nodes"]:
            self._nodes[node["id"]] = node

        for edge in raw["edges"]:
            self._add_edge(edge)

        # Spatial bucket for fast radius queries (0.01° ≈ 1 km buckets)
        self._bucket: dict = defaultdict(list)
        for nid, node in self._nodes.items():
            bkey = (round(node["lat"], 2), round(node["lon"], 2))
            self._bucket[bkey].append(nid)

        print(f"  Loaded {len(self._nodes):,} nodes, "
              f"{sum(len(v) for v in self._adj.values())//2:,} edges "
              f"in {time.time()-t0:.2f}s")

    def _add_edge(self, edge: dict):
        frm, to = edge["from"], edge["to"]
        # Enrich with travel_time if missing
        if "travel_time" not in edge:
            edge = dict(edge)
            edge["travel_time"] = _travel_time(
                edge.get("length", 1.0),
                str(edge.get("road_type", "unclassified"))
            )
        self._adj[frm].append((to, edge))
        self._adj[to].append((frm, edge))

    # -----------------------------------------------------------------------
    # Step 1 + 2: Build R-tree and KD-tree
    # -----------------------------------------------------------------------

    def build_index(self):
        """
        Build the hybrid spatial index and snap all facilities to the road
        network. Call once after loading; run.py does this automatically.

        R-tree  — indexes every edge by its lon/lat bounding box.
        KD-tree — indexes all road nodes by (lat, lon) for fast NN lookup.

        Facility snapping uses both:
          KD-tree → K nearest road nodes  →  walk adjacency for candidate edges
          R-tree  → select candidate edges in bbox  →  pick closest by
                    perpendicular distance  →  add facility_connector edges
        """
        if self._index_built:
            return

        t0 = time.time()

        # --- R-tree over edges ---
        prop = rtree_index.Property()
        prop.dimension = 2
        self._rtree = rtree_index.Index(properties=prop)
        self._rtree_edges = []

        for i, (nid, neighbours) in enumerate(self._adj.items()):
            node = self._nodes.get(nid)
            if not node:
                continue
            for nb_id, edge in neighbours:
                nb = self._nodes.get(nb_id)
                if not nb:
                    continue
                # Only insert each edge once (from < to)
                if nid < nb_id:
                    ei = len(self._rtree_edges)
                    minx = min(node["lon"], nb["lon"])
                    miny = min(node["lat"], nb["lat"])
                    maxx = max(node["lon"], nb["lon"])
                    maxy = max(node["lat"], nb["lat"])
                    # Pad bbox slightly so point queries intersect
                    pad = 1e-5
                    self._rtree.insert(ei, (minx-pad, miny-pad, maxx+pad, maxy+pad))
                    self._rtree_edges.append(edge)

        print(f"  R-tree: {len(self._rtree_edges):,} edges indexed "
              f"in {time.time()-t0:.2f}s")

        # --- KD-tree over road nodes ---
        t1 = time.time()
        self._kd_road_nodes = [
            n for n in self._nodes.values()
            if n.get("type", "road_node") == "road_node"
        ]
        coords = np.array([[n["lat"], n["lon"]] for n in self._kd_road_nodes])
        self._kdtree = KDTree(coords)
        print(f"  KD-tree: {len(self._kd_road_nodes):,} road nodes indexed "
              f"in {time.time()-t1:.2f}s")

        # --- Snap facilities to road network ---
        t2 = time.time()
        new_edges = self._snap_facilities()
        for e in new_edges:
            self._add_edge(e)
            # Also insert into R-tree
            n1 = self._nodes.get(e["from"])
            n2 = self._nodes.get(e["to"])
            if n1 and n2:
                ei = len(self._rtree_edges)
                minx = min(n1["lon"], n2["lon"])
                miny = min(n1["lat"], n2["lat"])
                maxx = max(n1["lon"], n2["lon"])
                maxy = max(n1["lat"], n2["lat"])
                pad = 1e-5
                self._rtree.insert(ei, (minx-pad, miny-pad, maxx+pad, maxy+pad))
                self._rtree_edges.append(e)

        print(f"  Snapped {len(new_edges):,} facility connector edges "
              f"in {time.time()-t2:.2f}s")

        self._index_built = True
        print(f"  Index ready. Total build time: {time.time()-t0:.2f}s")

    # -----------------------------------------------------------------------
    # Step 2: KD-tree facility snapping (edge snapping method)
    # -----------------------------------------------------------------------

    def _snap_facilities(self, k_neighbours: int = 5) -> list[dict]:
        """
        For each hospital and fire_station, find the geometrically nearest
        road edge using KD-tree + adjacency walk, then connect the facility
        to both endpoints of that edge.

        This is more accurate than node-only snapping because it handles
        facilities that sit mid-block (between two intersections).
        """
        facilities = [
            n for n in self._nodes.values()
            if n.get("type") in ("hospital", "fire_station")
        ]

        # Build adjacency lookup for edge-snapping
        adj_edges: dict[int, list[dict]] = defaultdict(list)
        for nid, neighbours in self._adj.items():
            for nb_id, edge in neighbours:
                adj_edges[nid].append(edge)

        new_edges = []
        already_connected = set()

        for fac in facilities:
            if fac["id"] in already_connected:
                continue

            # KD-tree: find K nearest road nodes
            _, idxs = self._kdtree.query(
                [fac["lat"], fac["lon"]], k=min(k_neighbours, len(self._kd_road_nodes))
            )
            if not hasattr(idxs, "__len__"):
                idxs = [idxs]

            # Collect candidate edges from those nodes' adjacency lists
            candidate_edges: dict[tuple, dict] = {}
            for idx in idxs:
                rn = self._kd_road_nodes[idx]
                for edge in adj_edges[rn["id"]]:
                    key = (min(edge["from"], edge["to"]),
                           max(edge["from"], edge["to"]))
                    candidate_edges[key] = edge

            if not candidate_edges:
                # Fallback: connect directly to nearest road node
                rn = self._kd_road_nodes[idxs[0]]
                d = _haversine_m(fac["lat"], fac["lon"], rn["lat"], rn["lon"])
                new_edges.append({
                    "from": fac["id"], "to": rn["id"],
                    "length": round(d, 2), "road_type": "facility_connector"
                })
                continue

            # Find geometrically closest edge via perpendicular distance
            best_dist = float("inf")
            best_edge = None
            for edge in candidate_edges.values():
                n1 = self._nodes.get(edge["from"])
                n2 = self._nodes.get(edge["to"])
                if not n1 or not n2:
                    continue
                d = _point_to_segment_m(
                    fac["lat"], fac["lon"],
                    n1["lat"], n1["lon"],
                    n2["lat"], n2["lon"]
                )
                if d < best_dist:
                    best_dist = d
                    best_edge = edge

            if best_edge:
                # Connect facility to BOTH endpoints (bidirectional coverage)
                for endpoint_id in (best_edge["from"], best_edge["to"]):
                    ep = self._nodes.get(endpoint_id)
                    if not ep:
                        continue
                    ep_dist = _haversine_m(
                        fac["lat"], fac["lon"], ep["lat"], ep["lon"]
                    )
                    new_edges.append({
                        "from": fac["id"],
                        "to":   endpoint_id,
                        "length": round(ep_dist, 2),
                        "road_type": "facility_connector",
                    })
                already_connected.add(fac["id"])

        return new_edges

    # -----------------------------------------------------------------------
    # Step 4: Disaster simulation using R-tree
    # -----------------------------------------------------------------------

    def apply_disaster(
        self,
        disaster_type: str,
        centre: tuple[float, float],
        radius_m: float,
        severity: float = 0.5,
    ) -> dict:
        """
        Use the R-tree to instantly retrieve all edges within the disaster
        radius, then mark them disabled or high-cost based on severity.
        Falls back to bucket-based node search if index not built.
        """
        lat, lon = centre
        deg = radius_m / 111_000

        # --- Node damage (bucket index) ---
        affected_nodes = self.get_nodes_in_radius(lat, lon, radius_m)
        affected_ids = {n["id"] for n in affected_nodes}
        for node in affected_nodes:
            dist = _haversine_m(lat, lon, node["lat"], node["lon"])
            attenuation = 1 - (dist / radius_m) * 0.5
            self._node_damage[node["id"]] = max(
                self._node_damage.get(node["id"], 0.0),
                min(severity * attenuation, 1.0),
            )

        # --- Edge disabling: R-tree query (Step 4) ---
        disabled_count = 0
        if self._rtree is not None:
            candidate_ids = list(self._rtree.intersection(
                (lon - deg, lat - deg, lon + deg, lat + deg)
            ))
            _ROAD_PRIORITY = {
                "motorway": 1, "trunk": 2, "primary": 3,
                "secondary": 4, "tertiary": 5,
                "residential": 6, "unclassified": 7,
            }
            for ei in candidate_ids:
                edge = self._rtree_edges[ei]
                frm, to = edge["from"], edge["to"]
                if frm not in affected_ids or to not in affected_ids:
                    continue
                priority = _ROAD_PRIORITY.get(
                    str(edge.get("road_type", "")), 7
                )
                threshold = 0.3 + (priority / 10) * 0.4
                if severity >= threshold:
                    self._disabled_edges.add(frozenset({frm, to}))
                    disabled_count += 1
        else:
            # Fallback without R-tree
            for nid in affected_ids:
                for nb_id, edge in self._adj[nid]:
                    if nb_id in affected_ids:
                        self._disabled_edges.add(frozenset({nid, nb_id}))
                        disabled_count += 1

        return {
            "disaster_type": disaster_type,
            "centre": {"lat": lat, "lon": lon},
            "radius_m": radius_m,
            "severity": severity,
            "nodes_affected": len(affected_nodes),
            "edges_disabled": disabled_count,
            "population_at_risk": self.get_population_in_radius(lat, lon, radius_m),
            "hospitals_affected": len(self.facilities_in_radius("hospital", lat, lon, radius_m)),
            "fire_stations_affected": len(self.facilities_in_radius("fire_station", lat, lon, radius_m)),
        }

    def reset_disaster(self):
        self._disabled_edges.clear()
        self._node_damage.clear()

    # -----------------------------------------------------------------------
    # Run isolation: all per-scenario state is _node_damage + _disabled_edges.
    # fork() gives each run its own copy of those two and shares everything
    # else (nodes, adjacency, R-tree, KD-tree), so no reload or re-index.
    # -----------------------------------------------------------------------

    def fork(self) -> "HybridTokyoGraph":
        """
        Independent scenario view of this graph (microseconds, vs ~17 s to
        reload and re-index). Damage applied to a fork never touches the parent
        or other forks. Call after build_index(); do not mutate the static
        network (nodes/edges/indexes) through a fork.
        """
        import copy
        if not self._index_built:
            raise RuntimeError("fork() requires build_index() to have been called")
        g = copy.copy(self)                      # shallow: shares static structures
        g._node_damage    = dict(self._node_damage)
        g._disabled_edges = set(self._disabled_edges)
        return g

    def damage_fingerprint(self) -> tuple:
        """Cheap summary of damage state, for leakage tests and logging."""
        return (len(self._node_damage), len(self._disabled_edges),
                round(sum(self._node_damage.values()), 6))

    def is_edge_passable(self, u: int, v: int) -> bool:
        return frozenset({u, v}) not in self._disabled_edges

    def get_node_damage(self, node_id: int) -> float:
        return self._node_damage.get(node_id, 0.0)

    # -----------------------------------------------------------------------
    # Step 5a: Dijkstra (distance-weighted)
    # -----------------------------------------------------------------------

    def shortest_path(
        self, from_id: int, to_id: int, avoid_damaged: bool = True
    ) -> Optional[list[int]]:
        """Dijkstra shortest path by distance (metres)."""
        if from_id not in self._nodes or to_id not in self._nodes:
            return None

        dist = {from_id: 0.0}
        prev: dict[int, int] = {}
        heap = [(0.0, from_id)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")):
                continue
            if u == to_id:
                break
            for v, edge in self._adj[u]:
                if avoid_damaged and not self.is_edge_passable(u, v):
                    continue
                w = edge["length"] * (1 + self._node_damage.get(v, 0.0) * 2)
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

        return self._reconstruct(prev, from_id, to_id)

    # -----------------------------------------------------------------------
    # Step 5b: A* (travel-time weighted, faster for long routes)
    # -----------------------------------------------------------------------

    def shortest_path_astar(
        self, from_id: int, to_id: int, avoid_damaged: bool = True
    ) -> Optional[list[int]]:
        """
        A* shortest path weighted by travel time (seconds).
        Heuristic = haversine distance / max speed (25 m/s).
        Faster than Dijkstra for long cross-city routes.
        """
        if from_id not in self._nodes or to_id not in self._nodes:
            return None

        goal = self._nodes[to_id]

        def h(nid: int) -> float:
            n = self._nodes.get(nid)
            if not n:
                return 0.0
            return _haversine_m(n["lat"], n["lon"], goal["lat"], goal["lon"]) / 25.0

        g: dict[int, float] = {from_id: 0.0}
        prev: dict[int, int] = {}
        heap = [(h(from_id), 0.0, from_id)]

        while heap:
            f, gc, u = heapq.heappop(heap)
            if gc > g.get(u, float("inf")):
                continue
            if u == to_id:
                break
            for v, edge in self._adj[u]:
                if avoid_damaged and not self.is_edge_passable(u, v):
                    continue
                damage_pen = 1 + self._node_damage.get(v, 0.0) * 3
                tt = edge.get("travel_time",
                              _travel_time(edge["length"],
                                           str(edge.get("road_type", "")))) * damage_pen
                ng = gc + tt
                if ng < g.get(v, float("inf")):
                    g[v] = ng
                    prev[v] = u
                    heapq.heappush(heap, (ng + h(v), ng, v))

        return self._reconstruct(prev, from_id, to_id)

    def _reconstruct(self, prev, from_id, to_id) -> Optional[list[int]]:
        if to_id not in prev and from_id != to_id:
            return None
        path, cur = [], to_id
        while cur != from_id:
            path.append(cur)
            cur = prev[cur]
        path.append(from_id)
        return list(reversed(path))

    # -----------------------------------------------------------------------
    # Fallback: staging zones (when no path exists)
    # -----------------------------------------------------------------------

    def find_staging_zones(
        self,
        disaster_centre: tuple[float, float],
        disaster_radius_m: float,
        n_zones: int = 3,
    ) -> list[dict]:
        """
        When routing fails, find the best staging zones just outside the
        disaster perimeter: fire stations with intact road access.
        Uses R-tree to quickly exclude facilities inside the affected zone.
        """
        lat, lon = disaster_centre
        outer_radius = disaster_radius_m * 1.5
        inner_radius = disaster_radius_m

        candidates = []
        for fs in self.get_fire_stations():
            d = _haversine_m(lat, lon, fs["lat"], fs["lon"])
            if inner_radius < d <= outer_radius:
                # Check it has at least one passable outgoing road
                passable_roads = sum(
                    1 for nb_id, _ in self._adj.get(fs["id"], [])
                    if self.is_edge_passable(fs["id"], nb_id)
                )
                if passable_roads > 0:
                    candidates.append({
                        "id":   fs["id"],
                        "lat":  fs["lat"],
                        "lon":  fs["lon"],
                        "type": "fire_station",
                        "dist_to_epicentre_m": round(d),
                        "passable_roads": passable_roads,
                    })

        # Sort: closest to perimeter first, then most roads
        candidates.sort(key=lambda x: (x["dist_to_epicentre_m"], -x["passable_roads"]))
        return candidates[:n_zones]

    # -----------------------------------------------------------------------
    # Spatial queries (bucket index — same as dataset_connector.py)
    # -----------------------------------------------------------------------

    def get_nodes_in_radius(
        self, lat: float, lon: float, radius_m: float,
        node_type: Optional[str] = None
    ) -> list[dict]:
        deg_spread = (radius_m / 111_000) + 0.02
        candidates: set = set()
        lat0, lon0 = round(lat, 2), round(lon, 2)
        steps = math.ceil(deg_spread / 0.01) + 1
        for dlat in range(-steps, steps+1):
            for dlon in range(-steps, steps+1):
                bkey = (round(lat0+dlat*0.01, 2), round(lon0+dlon*0.01, 2))
                candidates.update(self._bucket.get(bkey, []))
        results = []
        for nid in candidates:
            node = self._nodes[nid]
            if node_type and node.get("type", "road_node") != node_type:
                continue
            if _haversine_m(lat, lon, node["lat"], node["lon"]) <= radius_m:
                results.append(node)
        return results

    def get_population_in_radius(self, lat, lon, radius_m) -> int:
        return sum(
            b.get("population", 0)
            for b in self.get_nodes_in_radius(lat, lon, radius_m, "building")
        )

    def nearest_road_node(self, lat: float, lon: float) -> Optional[dict]:
        if self._kdtree is not None:
            _, idx = self._kdtree.query([lat, lon], k=1)
            return self._kd_road_nodes[idx]
        for r in [500, 2000, 10_000]:
            c = [n for n in self.get_nodes_in_radius(lat, lon, r, "road_node")
                 if n["id"] in self._adj]
            if c:
                return min(c, key=lambda n: _haversine_m(lat, lon, n["lat"], n["lon"]))
        return None

    def nearest_facility(self, facility_type: str, lat: float, lon: float) -> Optional[dict]:
        facs = self.get_hospitals() if facility_type == "hospital" else self.get_fire_stations()
        return min(facs, key=lambda n: _haversine_m(lat, lon, n["lat"], n["lon"])) if facs else None

    def facilities_in_radius(self, facility_type, lat, lon, radius_m) -> list[dict]:
        return self.get_nodes_in_radius(lat, lon, radius_m, node_type=facility_type)

    def get_hospitals(self)     -> list[dict]:
        return [n for n in self._nodes.values() if n.get("type") == "hospital"]

    def get_fire_stations(self) -> list[dict]:
        return [n for n in self._nodes.values() if n.get("type") == "fire_station"]

    def get_buildings(self)     -> list[dict]:
        return [n for n in self._nodes.values() if n.get("type") == "building"]

    def path_distance_m(self, path: list[int]) -> float:
        total = 0.0
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            for nb, edge in self._adj[u]:
                if nb == v:
                    total += edge["length"]
                    break
        return total

    def path_travel_time_s(self, path: list[int]) -> float:
        total = 0.0
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            for nb, edge in self._adj[u]:
                if nb == v:
                    total += edge.get("travel_time",
                                      _travel_time(edge["length"],
                                                   str(edge.get("road_type", ""))))
                    break
        return total

    def snapshot_local(self, lat: float, lon: float, radius_m: float) -> dict:
        nodes = self.get_nodes_in_radius(lat, lon, radius_m)
        return {
            "centre": {"lat": lat, "lon": lon},
            "radius_m": radius_m,
            "road_nodes":   [n for n in nodes if n.get("type", "road_node") == "road_node"],
            "buildings":    [n for n in nodes if n.get("type") == "building"],
            "hospitals":    [n for n in nodes if n.get("type") == "hospital"],
            "fire_stations":[n for n in nodes if n.get("type") == "fire_station"],
            "population":   sum(n.get("population", 0) for n in nodes
                                if n.get("type") == "building"),
            "damaged_nodes": [nid for nid in self._node_damage
                              if any(n["id"] == nid for n in nodes)],
        }

    def __repr__(self):
        idx_status = "indexed" if self._index_built else "not indexed"
        return (f"<HybridTokyoGraph city={self.city!r} "
                f"nodes={len(self._nodes):,} "
                f"edges={sum(len(v) for v in self._adj.values())//2:,} "
                f"[{idx_status}]>")


# ---------------------------------------------------------------------------
# Cached loader — parse + index each graph file once per process
# ---------------------------------------------------------------------------

_GRAPH_CACHE: dict = {}

def load_cached(json_path: str) -> "HybridTokyoGraph":
    """
    Return a fresh, undamaged fork of a graph that is loaded and indexed only
    once per process. Use one call per simulation run:

        graph = load_cached("tokyo_graph.json")     # first call ~17 s, then instant
    """
    import os
    key = os.path.abspath(json_path)
    if key not in _GRAPH_CACHE:
        base = HybridTokyoGraph(key)
        base.build_index()
        base.reset_disaster()
        _GRAPH_CACHE[key] = base                 # pristine; never handed out directly
    return _GRAPH_CACHE[key].fork()
