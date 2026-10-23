import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree as KDTree
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def lonlat_to_local_xy_ref(lons, lats, ref_lon, ref_lat):
    """与 dataset.py 完全一致的投影"""
    x = (lons - ref_lon) * 111320 * np.cos(np.radians(ref_lat))
    y = (lats - ref_lat) * 110540
    return x, y

class RoadNetwork:
    def __init__(self, data_path, ref_lon, ref_lat):
        self.ref_lon = ref_lon
        self.ref_lat = ref_lat
        self.road_positions = None
        self.kdtree = None

        shp_file_path = data_path
        if os.path.isdir(data_path):
            for f in ["gis_osm_roads_free_1.shp", "gis_osm_pois_free_1.shp"]:
                path = os.path.join(data_path, f)
                if os.path.exists(path):
                    shp_file_path = path
                    break

        if not os.path.exists(shp_file_path):
            raise FileNotFoundError(f"找不到路网文件: {shp_file_path}")

        try:
            try:
                gdf = gpd.read_file(shp_file_path, encoding='utf-8')
            except:
                gdf = gpd.read_file(shp_file_path, encoding='gbk')

            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True)
            elif gdf.crs.to_string() != "EPSG:4326":
                gdf = gdf.to_crs(epsg=4326)

            # 提取所有节点，转为局部坐标
            nodes = []
            for geom in gdf.geometry:
                if geom.is_empty:
                    continue
                if geom.geom_type == 'LineString':
                    nodes.extend(geom.coords)
                elif geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        nodes.extend(line.coords)

            if not nodes:
                logging.warning("未提取到路网节点")
                return

            nodes = np.array(nodes)
            xs, ys = lonlat_to_local_xy_ref(nodes[:,0], nodes[:,1], ref_lon, ref_lat)
            self.road_positions = np.column_stack([xs, ys]).astype(np.float32)
            self.kdtree = KDTree(self.road_positions)
            logging.info(f"路网构建成功，节点数：{len(self.road_positions)}")

        except Exception as e:
            logging.error(f"加载路网失败: {e}")
            self.road_positions = np.empty((0,2))

    def query_nearest(self, x, y):
        """返回最近节点坐标和距离"""
        if self.kdtree is None or len(self.road_positions) == 0:
            return x, y, 0.0
        dist, idx = self.kdtree.query([x, y])
        node = self.road_positions[idx]
        return node[0], node[1], dist


_global_road_network = None
def get_road_network(data_path, ref_lon, ref_lat):
    global _global_road_network
    if _global_road_network is None:
        _global_road_network = RoadNetwork(data_path, ref_lon, ref_lat)
    return _global_road_network