import numpy as np
from functools import lru_cache

try:
    from pyproj import Transformer
except Exception:
    Transformer = None

def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def lonlat_to_local_xy(lons, lats, ref_lon, ref_lat):
    if Transformer is not None:
        transformer = _get_transformer(ref_lon, ref_lat, inverse=False)
        return transformer.transform(lons, lats)
    x = (lons - ref_lon) * 111320 * np.cos(np.radians(ref_lat))
    y = (lats - ref_lat) * 110540
    return x, y

def local_xy_to_lonlat(x, y, ref_lon, ref_lat):
    if Transformer is not None:
        transformer = _get_transformer(ref_lon, ref_lat, inverse=True)
        return transformer.transform(x, y)
    lon = x / (111320 * np.cos(np.radians(ref_lat))) + ref_lon
    lat = y / 110540 + ref_lat
    return lon, lat

@lru_cache(maxsize=8)
def _get_transformer(ref_lon, ref_lat, inverse=False):
    proj_str = f"+proj=tmerc +lat_0={ref_lat} +lon_0={ref_lon} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
    if inverse:
        return Transformer.from_crs(proj_str, "EPSG:4326", always_xy=True)
    return Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)

def compute_metrics(pred_coords, gt_coords, mask_pred):
    diffs = []
    for i in range(len(pred_coords)):
        if mask_pred[i]:
            d = haversine_distance(pred_coords[i][0], pred_coords[i][1],
                                   gt_coords[i][0], gt_coords[i][1])
            diffs.append(d)
    diffs = np.array(diffs)
    mae = np.mean(diffs) if len(diffs) > 0 else 0
    rmse = np.sqrt(np.mean(diffs**2)) if len(diffs) > 0 else 0
    return mae, rmse
