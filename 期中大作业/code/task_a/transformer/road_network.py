# road_network.py
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree as KDTree
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RoadNetwork:
    def __init__(self, data_path, ref_lon, ref_lat, densify_step_m=15.0):
        """
        初始化路网处理器 (针对 Shapefile 格式优化)
        :param data_path: 文件夹路径 或 .shp 文件路径
        :param ref_lon: 投影参考点经度
        :param ref_lat: 投影参考点纬度
        """
        self.ref_lon = ref_lon
        self.ref_lat = ref_lat
        self.densify_step_m = densify_step_m
        self.kdtree = None
        self.road_positions = None

        logging.info(f"正在加载路网数据: {data_path}")

        # 1. 确定要读取的文件路径
        shp_file_path = data_path

        # 如果传入的是文件夹，自动寻找 roads 文件
        if os.path.isdir(data_path):
            potential_files = [
                os.path.join(data_path, "gis_osm_roads_free_1.shp"),
                os.path.join(data_path, "gis_osm_pois_free_1.shp")  # 备选
            ]
            for f in potential_files:
                if os.path.exists(f):
                    shp_file_path = f
                    break

        if not os.path.exists(shp_file_path):
            logging.error(f"找不到 Shapefile 文件: {shp_file_path}")
            raise FileNotFoundError(f"Shapefile not found: {shp_file_path}")

        try:
            # 2. 读取 Shapefile
            try:
                gdf = gpd.read_file(shp_file_path, encoding='utf-8')
            except UnicodeDecodeError:
                gdf = gpd.read_file(shp_file_path, encoding='gbk')

            logging.info(f"原始数据加载完成，共 {len(gdf)} 条几何记录。")

            # 3. 坐标系转换
            if gdf.crs is None:
                logging.warning("Shapefile 缺失 CRS 信息，假设为 WGS84 (EPSG:4326)")
                gdf.set_crs(epsg=4326, inplace=True)

            # 定义目标投影：以参考点为中心的横轴墨卡托投影
            proj_str = f"+proj=tmerc +lat_0={ref_lat} +lon_0={ref_lon} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"

            if gdf.crs.to_string() != "EPSG:4326":
                logging.info("正在将数据重投影到 WGS84 (EPSG:4326)...")
                gdf = gdf.to_crs(epsg=4326)

            min_lon = ref_lon - 0.14
            max_lon = ref_lon + 0.14
            min_lat = ref_lat - 0.12
            max_lat = ref_lat + 0.12
            gdf_local = gdf.cx[min_lon:max_lon, min_lat:max_lat]
            if len(gdf_local) > 0:
                logging.info(f"按轨迹活动范围裁剪路网：{len(gdf)} -> {len(gdf_local)} 条记录。")
                gdf = gdf_local

            logging.info("正在将数据投影到局部平面坐标系...")
            gdf_projected = gdf.to_crs(proj_str)

            # 4. 解析几何图形，并按固定间距加密道路折线。
            # 只用 OSM 原始顶点会把点吸到路口/折点，长直路段上会产生错误偏移。
            nodes = []
            logging.info("正在解析几何图形并构建索引...")
            for geom in gdf_projected.geometry:
                if geom.is_empty:
                    continue
                if geom.geom_type == 'LineString':
                    nodes.extend(self._densify_coords(list(geom.coords)))
                elif geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        nodes.extend(self._densify_coords(list(line.coords)))

            if len(nodes) == 0:
                logging.warning("加载路网失败：未从 Shapefile 中提取到任何坐标点。")
                self.road_positions = np.empty((0, 2), dtype=np.float32)
                return

            self.road_positions = np.array(nodes, dtype=np.float32)
            self.kdtree = KDTree(self.road_positions)

            logging.info(f"路网构建成功！包含 {len(self.road_positions)} 个路网节点。")

        except Exception as e:
            logging.error(f"加载路网时发生严重错误: {e}")
            self.road_positions = np.empty((0, 2), dtype=np.float32)

    def get_nearest_node(self, lon, lat):
        """
        根据投影坐标 (x, y) 找到最近的路网节点坐标。
        注意：输入应为局部投影坐标，而非原始经纬度。
        """
        if self.kdtree is None or len(self.road_positions) == 0:
            return None
        dist, idx = self.kdtree.query([lon, lat])
        return self.road_positions[idx]

    def query_nearest(self, x, y):
        """
        查询局部坐标 (x, y) 的最近路网节点。
        返回 (节点x, 节点y, 距离)。
        若路网不可用，返回 (x, y, 0.0)。
        """
        if self.kdtree is None or len(self.road_positions) == 0:
            return x, y, 0.0
        dist, idx = self.kdtree.query([x, y])
        node = self.road_positions[idx]
        return node[0], node[1], dist

    def _densify_coords(self, coords):
        if len(coords) < 2:
            return coords

        dense = []
        for p0, p1 in zip(coords[:-1], coords[1:]):
            x0, y0 = p0
            x1, y1 = p1
            dx = x1 - x0
            dy = y1 - y0
            length = float(np.hypot(dx, dy))
            if length <= 1e-6:
                continue
            steps = max(1, int(np.ceil(length / self.densify_step_m)))
            for j in range(steps):
                t = j / steps
                dense.append((x0 + t * dx, y0 + t * dy))
        dense.append(coords[-1])
        return dense


_global_road_network = None

def get_road_network(data_path, ref_lon, ref_lat, densify_step_m=15.0):
    """
    全局单例获取路网，避免重复加载。
    """
    global _global_road_network
    if _global_road_network is None:
        _global_road_network = RoadNetwork(data_path, ref_lon, ref_lat, densify_step_m)
    return _global_road_network
