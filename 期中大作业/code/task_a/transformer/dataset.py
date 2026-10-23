# dataset.py
import pickle
import importlib
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config
from road_network import get_road_network
from utils import lonlat_to_local_xy
import logging

class TrajectoryDataset(Dataset):
    def __init__(self, pkl_path, augment=True):
        # 原有兼容性映射代码
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

        if self.augment:
            keep_ratio = np.random.uniform(0.06, 0.15)
            N = len(coords)
            num_keep = max(2, int(N * keep_ratio))
            keep_indices = np.sort(np.random.choice(N, num_keep, replace=False))
            keep_indices = np.unique(np.concatenate(([0], keep_indices, [N-1])))
            mask_known = np.zeros(N, dtype=bool)
            mask_known[keep_indices] = True
        else:
            mask_known = np.array(traj.get('mask', np.ones(len(coords), dtype=bool)))

        # 若存在预计算的路网偏移，一并返回（验证/推理用）
        road_x = traj.get('road_offset_x', None)
        road_y = traj.get('road_offset_y', None)

        return coords, timestamps, mask_known, road_x, road_y


def pad_collate(batch):
    coords_list, times_list, mask_list, road_x_list, road_y_list = zip(*batch)
    lengths = [len(c) for c in coords_list]
    max_len = max(lengths)
    B = len(batch)

    # 1. 坐标转换与线性插值
    batch_xy = np.zeros((B, max_len, 2), dtype=np.float32)
    batch_interp_xy = np.zeros((B, max_len, 2), dtype=np.float32)
    batch_mask = np.zeros((B, max_len), dtype=bool)
    batch_padding = np.ones((B, max_len), dtype=bool)

    for i in range(B):
        L = lengths[i]
        c = coords_list[i]
        m = mask_list[i]

        x, y = lonlat_to_local_xy(c[:, 0], c[:, 1], Config.REF_LON, Config.REF_LAT)
        batch_xy[i, :L, 0] = x
        batch_xy[i, :L, 1] = y
        batch_mask[i, :L] = m
        batch_padding[i, :L] = False

        known_idx = np.where(m)[0]
        if len(known_idx) >= 2:
            batch_interp_xy[i, :L, 0] = np.interp(np.arange(L), known_idx, x[m])
            batch_interp_xy[i, :L, 1] = np.interp(np.arange(L), known_idx, y[m])
        else:
            batch_interp_xy[i, :L, :] = 0.0

    # 2. 路网偏移：优先使用预计算值；否则批量在线查询（大幅加速）
    road_offset_x = np.zeros((B, max_len), dtype=np.float32)
    road_offset_y = np.zeros((B, max_len), dtype=np.float32)

    # 检查是否全部轨迹都提供了预计算字段（验证/推理）
    all_precomputed = all(rx is not None for rx in road_x_list)

    if all_precomputed:
        for i in range(B):
            L = lengths[i]
            road_offset_x[i, :L] = road_x_list[i]
            road_offset_y[i, :L] = road_y_list[i]
    else:
        # 在线计算（训练集），使用批量 KDTree 查询，替换双重循环
        try:
            rn = get_road_network(Config.ROAD_PATH, Config.REF_LON, Config.REF_LAT)
        except Exception:
            rn = None

        if rn is not None and rn.kdtree is not None:
            for i in range(B):
                L = lengths[i]
                if L == 0:
                    continue
                # 提取该轨迹所有插值点 (L, 2)
                points = np.stack([batch_interp_xy[i, :L, 0],
                                   batch_interp_xy[i, :L, 1]], axis=-1)
                try:
                    # 批量查询最近路网点
                    dists, idxs = rn.kdtree.query(points)
                    nearest = rn.road_positions[idxs]   # (L, 2)
                    road_offset_x[i, :L] = (nearest[:, 0] - points[:, 0]) / Config.XY_STD
                    road_offset_y[i, :L] = (nearest[:, 1] - points[:, 1]) / Config.XY_STD
                except Exception:
                    continue

    # 3. 时间差特征
    delta_t = np.zeros((B, max_len), dtype=np.float32)
    for i in range(B):
        L = lengths[i]
        if L > 1:
            delta_t[i, 1:L] = times_list[i][1:] - times_list[i][:-1]

    # 4. 拼接 6 维特征
    batch_interp_xy_norm = batch_interp_xy / Config.XY_STD
    features = np.concatenate([
        batch_interp_xy_norm,
        batch_mask.astype(np.float32)[..., np.newaxis],
        (np.clip(delta_t, 0, 3600) / 60.0)[..., np.newaxis],
        road_offset_x[..., np.newaxis],
        road_offset_y[..., np.newaxis]
    ], axis=-1)

    features = torch.from_numpy(features)
    batch_xy_norm = torch.from_numpy(batch_xy / Config.XY_STD)
    batch_mask_tensor = torch.from_numpy(batch_mask)
    batch_padding_tensor = torch.from_numpy(batch_padding)

    batch_interp_xy_norm_t = torch.from_numpy(batch_interp_xy_norm)

    return features, batch_xy_norm, batch_interp_xy_norm_t, batch_mask_tensor, batch_padding_tensor, lengths


def get_dataloader(pkl_path, batch_size, shuffle=True, augment=True, num_workers=0):
    ds = TrajectoryDataset(pkl_path, augment=augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=pad_collate, num_workers=num_workers)
