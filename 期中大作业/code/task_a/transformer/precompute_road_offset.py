import pickle
import numpy as np
import os
import sys
from config import Config
from utils import lonlat_to_local_xy
from road_network import get_road_network

def fill_nan_with_linear(coords, mask):
    """
    基于已知点对缺失点做线性插值，返回完整坐标 (N,2)。
    若已知点不足2个，返回全零。
    """
    N = len(coords)
    interp = np.zeros_like(coords)
    known_idx = np.where(mask)[0]
    if len(known_idx) < 2:
        return interp
    for dim in range(2):
        interp[:, dim] = np.interp(np.arange(N), known_idx, coords[known_idx, dim])
    return interp

def precompute_road_offset(input_path, output_path, ref_lon, ref_lat, xy_std):
    print(f"处理文件: {input_path}")
    if not os.path.exists(input_path):
        print(f"警告：源文件不存在，跳过: {input_path}")
        return

    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    rn = get_road_network(Config.ROAD_PATH, ref_lon, ref_lat)

    new_data = []
    for traj in data:
        coords = np.array(traj['coords'], dtype=np.float32)
        timestamps = np.array(traj['timestamps'], dtype=np.float32)
        # 如果没有 mask 字段，则视为全部已知（插值后就是自身）
        mask = np.array(traj.get('mask', np.ones(len(coords), dtype=bool)))

        # 经纬度转局部坐标
        x, y = lonlat_to_local_xy(coords[:, 0], coords[:, 1], ref_lon, ref_lat)
        local_coords = np.stack([x, y], axis=-1)

        # 插值填充缺失点
        interp_coords = fill_nan_with_linear(local_coords, mask)

        # 计算路网偏移
        road_offset_x = np.zeros(len(coords), dtype=np.float32)
        road_offset_y = np.zeros(len(coords), dtype=np.float32)

        for i in range(len(coords)):
            px, py = interp_coords[i, 0], interp_coords[i, 1]
            try:
                nx, ny, _ = rn.query_nearest(px, py)
                road_offset_x[i] = (nx - px) / xy_std
                road_offset_y[i] = (ny - py) / xy_std
            except Exception:
                continue

        new_traj = dict(traj)
        new_traj['road_offset_x'] = road_offset_x
        new_traj['road_offset_y'] = road_offset_y
        new_data.append(new_traj)

    os.makedirs(os.path.dirname(os.fspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(new_data, f)
    print(f"已保存至: {output_path}")

if __name__ == "__main__":
    # 注意：这里要读的是**原始文件**，输出文件才是 Config 中配置的路径
    tasks = [
        # 原始推理输入 -> 带路网偏移的推理输入
        (Config.RAW_VAL_INPUT_8, Config.VAL_INPUT_8),
        (Config.RAW_VAL_INPUT_16, Config.VAL_INPUT_16),
    ]

    for src, dst in tasks:
        precompute_road_offset(src, dst, Config.REF_LON, Config.REF_LAT, Config.XY_STD)
