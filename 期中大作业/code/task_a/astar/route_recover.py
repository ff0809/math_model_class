import argparse
import heapq
import importlib
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


REF_LON = 108.946518
REF_LAT = 34.244678
DEFAULT_ROAD_SHP = (
    "task_a/shaanxi-260507-free.shp/gis_osm_roads_free_1.shp"
)


DRIVABLE_FCLASSES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "road",
}


@dataclass(frozen=True)
class Segment:
    idx: int
    u: tuple
    v: tuple
    p0: tuple
    p1: tuple
    length: float
    fclass: str


@dataclass(frozen=True)
class Candidate:
    seg_idx: int
    dist: float
    ratio: float
    snap: tuple


class RoadMatcher:
    def __init__(
        self,
        road_shp,
        bbox,
        bbox_margin=0.05,
        candidate_radius=60.0,
        topk=3,
        max_speed_kph=150.0,
    ):
        self.road_shp = road_shp
        self.bbox = bbox
        self.bbox_margin = bbox_margin
        self.candidate_radius = candidate_radius
        self.topk = topk
        self.max_speed_mps = max_speed_kph / 3.6
        self.to_xy = Transformer.from_crs(
            "EPSG:4326", self._proj4(), always_xy=True
        )
        self.to_lonlat = Transformer.from_crs(
            self._proj4(), "EPSG:4326", always_xy=True
        )
        self.graph = nx.Graph()
        self.segments = []
        self.segment_lines = []
        self.segment_tree = None
        self._build()

    def _proj4(self):
        return (
            f"+proj=tmerc +lat_0={REF_LAT} +lon_0={REF_LON} +k=1 "
            "+x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
        )

    def lonlat_to_xy(self, lon, lat):
        return self.to_xy.transform(lon, lat)

    def xy_to_lonlat(self, x, y):
        return self.to_lonlat.transform(x, y)

    def _build(self):
        min_lon, min_lat, max_lon, max_lat = self.bbox
        margin = self.bbox_margin
        read_bbox = (
            min_lon - margin,
            min_lat - margin,
            max_lon + margin,
            max_lat + margin,
        )
        print(f"Loading roads in bbox={read_bbox}")
        gdf = gpd.read_file(self.road_shp, bbox=read_bbox)
        if "fclass" in gdf.columns:
            gdf = gdf[gdf["fclass"].isin(DRIVABLE_FCLASSES)].copy()
        print(f"Road records after filter: {len(gdf)}")

        node_xy = {}

        def node_key(x, y):
            return (round(float(x), 2), round(float(y), 2))

        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            fclass = str(row.get("fclass", ""))
            lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
            for line in lines:
                coords = list(line.coords)
                if len(coords) < 2:
                    continue
                lons = [p[0] for p in coords]
                lats = [p[1] for p in coords]
                xs, ys = self.lonlat_to_xy(lons, lats)
                xy = list(zip(xs, ys))
                for p0, p1 in zip(xy[:-1], xy[1:]):
                    length = dist_xy(p0, p1)
                    if length < 1.0:
                        continue
                    u = node_key(*p0)
                    v = node_key(*p1)
                    node_xy[u] = u
                    node_xy[v] = v
                    self.graph.add_node(u, xy=u)
                    self.graph.add_node(v, xy=v)
                    # Undirected is intentional for the fast prototype: geometry
                    # matters more than legal one-way semantics for recovery error.
                    old = self.graph.get_edge_data(u, v, default={}).get("weight")
                    if old is None or length < old:
                        self.graph.add_edge(u, v, weight=length)
                    seg_idx = len(self.segments)
                    self.segments.append(
                        Segment(seg_idx, u, v, u, v, length, fclass)
                    )
                    self.segment_lines.append(LineString([u, v]))

        self.segment_tree = STRtree(self.segment_lines)
        print(
            f"Graph nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()}, segments={len(self.segments)}"
        )

    def candidates(self, lon, lat):
        x, y = self.lonlat_to_xy(float(lon), float(lat))
        point = Point(x, y)
        radius = self.candidate_radius
        idxs = []
        for _ in range(4):
            idxs = list(self.segment_tree.query(point.buffer(radius)))
            if idxs:
                break
            radius *= 2.0
        candidates = []
        for raw_idx in idxs:
            seg_idx = int(raw_idx)
            line = self.segment_lines[seg_idx]
            proj = line.project(point)
            snap_pt = line.interpolate(proj)
            dist = point.distance(snap_pt)
            ratio = proj / max(line.length, 1e-6)
            candidates.append(
                Candidate(seg_idx, float(dist), float(ratio), (snap_pt.x, snap_pt.y))
            )
        candidates.sort(key=lambda c: c.dist)
        return candidates[: self.topk]

    def best_gap_route(self, cand_a, cand_b, dt_seconds, straight_m):
        if not cand_a or not cand_b:
            return None
        cutoff = max(
            self.max_speed_mps * max(dt_seconds, 1.0) * 1.35,
            straight_m * 2.5,
            500.0,
        )
        best = None
        for ca in cand_a:
            seg_a = self.segments[ca.seg_idx]
            for cb in cand_b:
                seg_b = self.segments[cb.seg_idx]
                if ca.seg_idx == cb.seg_idx:
                    total_len = dist_xy(ca.snap, cb.snap)
                    score = 6.0 * (ca.dist + cb.dist) + total_len
                    polyline = [ca.snap, cb.snap]
                    if best is None or score < best[0]:
                        best = (score, total_len, polyline)
                    continue
                for start_node in (seg_a.u, seg_a.v):
                    head = dist_xy(ca.snap, start_node)
                    for end_node in (seg_b.u, seg_b.v):
                        tail = dist_xy(end_node, cb.snap)
                        try:
                            path_nodes = nx.astar_path(
                                self.graph,
                                start_node,
                                end_node,
                                heuristic=lambda u, v: dist_xy(u, v),
                                weight="weight",
                            )
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            continue
                        core_len = path_length(self.graph, path_nodes)
                        total_len = head + core_len + tail
                        if total_len > cutoff:
                            continue
                        # Candidate GPS distance should dominate local road choice;
                        # route length keeps us from choosing silly detours.
                        score = 6.0 * (ca.dist + cb.dist) + total_len
                        polyline = [ca.snap]
                        polyline.extend([self.graph.nodes[n]["xy"] for n in path_nodes])
                        polyline.append(cb.snap)
                        if best is None or score < best[0]:
                            best = (score, total_len, polyline)
        return best


def dist_xy(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def path_length(graph, path_nodes):
    total = 0.0
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        total += graph[u][v]["weight"]
    return total


def haversine(lon1, lat1, lon2, lat2):
    radius = 6371000.0
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def load_pickle(path):
    for alias in ("numpy._core", "numpy._core._multiarray_umath"):
        if alias not in sys.modules:
            try:
                sys.modules[alias] = importlib.import_module(
                    alias.replace("numpy._core", "numpy.core")
                )
            except Exception:
                pass
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f)


def input_bbox(trajs):
    coords = []
    for traj in trajs:
        arr = np.asarray(traj["coords"], dtype=float)
        mask = np.asarray(traj["mask"], dtype=bool)
        finite = mask & np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
        if finite.any():
            coords.append(arr[finite])
    all_coords = np.vstack(coords)
    return (
        float(all_coords[:, 0].min()),
        float(all_coords[:, 1].min()),
        float(all_coords[:, 0].max()),
        float(all_coords[:, 1].max()),
    )


def interpolate_polyline(polyline, target_dist):
    if len(polyline) == 0:
        return None
    if len(polyline) == 1:
        return polyline[0]
    remaining = target_dist
    prev = polyline[0]
    for cur in polyline[1:]:
        seg_len = dist_xy(prev, cur)
        if seg_len >= remaining:
            ratio = 0.0 if seg_len <= 1e-9 else remaining / seg_len
            return (
                prev[0] + ratio * (cur[0] - prev[0]),
                prev[1] + ratio * (cur[1] - prev[1]),
            )
        remaining -= seg_len
        prev = cur
    return polyline[-1]


def local_linear_fill(coords, timestamps, mask, to_xy, to_lonlat):
    arr = np.asarray(coords, dtype=float).copy()
    x, y = to_xy.transform(arr[:, 0], arr[:, 1])
    known = np.where(mask)[0]
    xs = np.interp(np.arange(len(arr)), known, np.asarray(x)[known])
    ys = np.interp(np.arange(len(arr)), known, np.asarray(y)[known])
    lon, lat = to_lonlat.transform(xs, ys)
    arr[~mask, 0] = lon[~mask]
    arr[~mask, 1] = lat[~mask]
    return arr


def recover_trajectory(traj, matcher):
    raw_coords = np.asarray(traj["coords"], dtype=float)
    timestamps = np.asarray(traj["timestamps"], dtype=float)
    mask = np.asarray(traj["mask"], dtype=bool)
    final = local_linear_fill(
        raw_coords, timestamps, mask, matcher.to_xy, matcher.to_lonlat
    )

    known = np.where(mask)[0]
    candidate_cache = {}
    for idx in known:
        lon, lat = raw_coords[idx]
        candidate_cache[idx] = matcher.candidates(lon, lat)

    for left, right in zip(known[:-1], known[1:]):
        if right <= left + 1:
            continue
        dt = float(timestamps[right] - timestamps[left])
        if dt <= 0:
            continue
        straight = float(
            haversine(
                raw_coords[left, 0],
                raw_coords[left, 1],
                raw_coords[right, 0],
                raw_coords[right, 1],
            )
        )
        best = matcher.best_gap_route(
            candidate_cache[left], candidate_cache[right], dt, straight
        )
        if best is None:
            continue
        _, total_len, polyline = best
        if total_len < 1e-6:
            continue
        for idx in range(left + 1, right):
            alpha = float(timestamps[idx] - timestamps[left]) / dt
            xy = interpolate_polyline(polyline, alpha * total_len)
            if xy is None:
                continue
            lon, lat = matcher.xy_to_lonlat(*xy)
            final[idx] = [lon, lat]

    final[mask] = raw_coords[mask]
    return final


def evaluate(preds, inputs, gt):
    all_dist = []
    by_traj = []
    for pred, inp, truth in zip(preds, inputs, gt):
        pred_coords = np.asarray(pred["coords"], dtype=float)
        gt_coords = np.asarray(truth["coords"], dtype=float)
        mask_pred = ~np.asarray(inp["mask"], dtype=bool)
        d = haversine(
            pred_coords[mask_pred, 0],
            pred_coords[mask_pred, 1],
            gt_coords[mask_pred, 0],
            gt_coords[mask_pred, 1],
        )
        if len(d):
            all_dist.append(d)
            by_traj.append(float(np.mean(d)))
    if not all_dist:
        return 0.0, 0.0, []
    all_dist = np.concatenate(all_dist)
    return (
        float(np.mean(all_dist)),
        float(np.sqrt(np.mean(all_dist**2))),
        list(np.percentile(all_dist, [50, 75, 90, 95, 99])),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gt")
    parser.add_argument("--road-shp", default=DEFAULT_ROAD_SHP)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--candidate-radius", type=float, default=60.0)
    parser.add_argument("--bbox-margin", type=float, default=0.05)
    parser.add_argument("--max-speed-kph", type=float, default=150.0)
    return parser.parse_args()


def main():
    args = parse_args()
    inputs = load_pickle(args.input)
    if args.limit:
        inputs = inputs[: args.limit]
    bbox = input_bbox(inputs)
    matcher = RoadMatcher(
        args.road_shp,
        bbox=bbox,
        bbox_margin=args.bbox_margin,
        candidate_radius=args.candidate_radius,
        topk=args.topk,
        max_speed_kph=args.max_speed_kph,
    )

    results = []
    for i, traj in enumerate(inputs, 1):
        coords = recover_trajectory(traj, matcher)
        results.append({"traj_id": traj["traj_id"], "coords": coords.tolist()})
        if i % 50 == 0:
            print(f"Recovered {i}/{len(inputs)}")
    save_pickle(args.output, results)
    print(f"Saved {args.output}")

    if args.gt:
        gt = load_pickle(args.gt)
        if args.limit:
            gt = gt[: args.limit]
        mae, rmse, pct = evaluate(results, inputs, gt)
        print(f"MAE={mae:.2f}m RMSE={rmse:.2f}m percentiles={pct}")


if __name__ == "__main__":
    main()
