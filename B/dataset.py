# dataset.py
import pickle
import importlib
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config
from road_network import get_road_network

class TrajectoryDataset(Dataset):
    def __init__(self, pkl_path, augment=True):
        # Pickle 兼容性代码（保留原有逻辑）
        try:
            for alias in ('numpy._core', 'numpy._core._multiarray_umath'):
                if alias not in sys.modules:
                    try:
                        target = alias.replace('numpy._core', 'numpy.core')
                        sys.modules[alias] = importlib.import_module(target)
                    except Exception:
                        try:
                            sys.modules[alias] = importlib.import_module('numpy')
                        except Exception:
                            pass
            with open(pkl_path, 'rb') as f:
                try:
                    self.data = pickle.load(f)
                except ModuleNotFoundError as e:
                    name = getattr(e, 'name', '')
                    if name and name.startswith('numpy'):
                        for alias in (name, 'numpy._core', 'numpy._core._multiarray_umath'):
                            if alias not in sys.modules:
                                try:
                                    target = alias.replace('numpy._core', 'numpy.core')
                                    sys.modules[alias] = importlib.import_module(target)
                                except Exception:
                                    try:
                                        sys.modules[alias] = importlib.import_module('numpy')
                                    except Exception:
                                        pass
                        f.seek(0)
                        self.data = pickle.load(f)
                    else:
                        raise
        except Exception as e:
            print(f"加载数据失败: {e}")
            self.data = []
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj = self.data[idx]
        coords = np.array(traj['coords'], dtype=np.float32)
        timestamps = np.array(traj['timestamps'], dtype=np.float32)
        N = len(coords)  # 提前定义 N

        if self.augment:
            keep_ratio = np.random.uniform(0.06, 0.15)
            num_keep = max(2, int(N * keep_ratio))
            keep_indices = np.sort(np.random.choice(N, num_keep, replace=False))
            keep_indices = np.unique(np.concatenate(([0], keep_indices, [N-1])))
            mask_known = np.zeros(N, dtype=bool)
            mask_known[keep_indices] = True
        else:
            # 使用文件中已有的 mask，若没有则全为已知（不推荐用于验证）
            mask_known = np.array(traj.get('mask', np.ones(N, dtype=bool)))
        return coords, timestamps, mask_known


def _local_xy(lon, lat, ref_lon, ref_lat):
    x = (lon - ref_lon) * 111320 * np.cos(np.radians(ref_lat))
    y = (lat - ref_lat) * 110540
    return x, y


def pad_collate(batch):
    coords_list, times_list, mask_list = zip(*batch)
    lengths = [len(c) for c in coords_list]
    max_len = max(lengths)
    B = len(batch)

    # 坐标准备
    batch_xy = np.zeros((B, max_len, 2), dtype=np.float32)
    batch_interp = np.zeros((B, max_len, 2), dtype=np.float32)
    batch_mask = np.zeros((B, max_len), dtype=bool)
    batch_pad  = np.ones((B, max_len), dtype=bool)

    for i in range(B):
        L = lengths[i]
        c = coords_list[i]
        m = mask_list[i]
        x, y = _local_xy(c[:, 0], c[:, 1], Config.REF_LON, Config.REF_LAT)
        batch_xy[i, :L, 0] = x
        batch_xy[i, :L, 1] = y
        batch_mask[i, :L] = m
        batch_pad[i, :L] = False

        known_idx = np.where(m)[0]
        if len(known_idx) >= 2:
            batch_interp[i, :L, 0] = np.interp(np.arange(L), known_idx, x[m])
            batch_interp[i, :L, 1] = np.interp(np.arange(L), known_idx, y[m])
        else:
            batch_interp[i, :L, :] = 0.0

    batch_interp = np.nan_to_num(batch_interp, nan=0.0, posinf=0.0, neginf=0.0)

    # 路网偏移（在线查询，使用放大系数 ROAD_STD）
    road_ox = np.zeros((B, max_len), dtype=np.float32)
    road_oy = np.zeros((B, max_len), dtype=np.float32)
    try:
        rn = get_road_network(Config.ROAD_PATH, Config.REF_LON, Config.REF_LAT)
    except Exception:
        rn = None

    if rn is not None and rn.kdtree is not None:
        for i in range(B):
            L = lengths[i]
            if L == 0:
                continue
            pts = batch_interp[i, :L, :]   # (L, 2)
            dists, idxs = rn.kdtree.query(pts)
            nearest = rn.road_positions[idxs]   # (L, 2)
            road_ox[i, :L] = (nearest[:, 0] - pts[:, 0]) / Config.ROAD_STD
            road_oy[i, :L] = (nearest[:, 1] - pts[:, 1]) / Config.ROAD_STD

    # 时间差
    delta_t = np.zeros((B, max_len), dtype=np.float32)
    for i in range(B):
        L = lengths[i]
        if L > 1:
            delta_t[i, 1:L] = times_list[i][1:] - times_list[i][:-1]

    # 拼接 6 维特征
    features = np.concatenate([
        batch_interp / Config.XY_STD,                     # 2
        batch_mask.astype(np.float32)[..., None],         # 1
        (np.clip(delta_t, 0, 3600) / 60.0)[..., None],   # 1
        road_ox[..., None],                               # 1
        road_oy[..., None]                                # 1
    ], axis=-1)

    # 目标残差（真实坐标 - 插值坐标）/ XY_STD
    target_res = (batch_xy - batch_interp) / Config.XY_STD

    # 安全清理
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    target_res = np.nan_to_num(target_res, nan=0.0, posinf=0.0, neginf=0.0)

    features = torch.from_numpy(features)
    target_res = torch.from_numpy(target_res)
    batch_mask_t = torch.from_numpy(batch_mask)
    batch_pad_t  = torch.from_numpy(batch_pad)

    return features, target_res, batch_mask_t, batch_pad_t, lengths


def get_dataloader(pkl_path, batch_size, shuffle=True, augment=True, num_workers=0):
    ds = TrajectoryDataset(pkl_path, augment=augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=pad_collate, num_workers=num_workers)