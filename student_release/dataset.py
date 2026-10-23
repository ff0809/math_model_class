import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config
from utils import lonlat_to_local_xy

class TrajectoryDataset(Dataset):
    def __init__(self, pkl_path, augment=True):
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
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

        return coords, timestamps, mask_known

def pad_collate(batch):
    coords_list, times_list, mask_list = zip(*batch)
    lengths = [len(c) for c in coords_list]
    max_len = max(lengths)
    B = len(batch)

    # 准备张量
    batch_xy = np.zeros((B, max_len, 2), dtype=np.float32)
    batch_interp_xy = np.zeros((B, max_len, 2), dtype=np.float32)
    batch_mask = np.zeros((B, max_len), dtype=bool)
    batch_padding = np.ones((B, max_len), dtype=bool)

    for i in range(B):
        L = lengths[i]
        c = coords_list[i]
        m = mask_list[i]
        t = times_list[i]

        x, y = lonlat_to_local_xy(c[:, 0], c[:, 1], Config.REF_LON, Config.REF_LAT)
        batch_xy[i, :L, 0] = x
        batch_xy[i, :L, 1] = y
        batch_mask[i, :L] = m
        batch_padding[i, :L] = False

        known_idx = np.where(m)[0]
        batch_interp_xy[i, :L, 0] = np.interp(np.arange(L), known_idx, x[m])
        batch_interp_xy[i, :L, 1] = np.interp(np.arange(L), known_idx, y[m])

    batch_xy_norm = batch_xy / Config.XY_STD
    batch_interp_xy_norm = batch_interp_xy / Config.XY_STD

    delta_t = np.zeros((B, max_len), dtype=np.float32)
    for i in range(B):
        L = lengths[i]
        delta_t[i, 1:L] = times_list[i][1:] - times_list[i][:-1]
    
    feat_interp = torch.from_numpy(batch_interp_xy_norm)
    feat_mask = torch.from_numpy(batch_mask.astype(np.float32)).unsqueeze(-1)
    feat_dt = torch.from_numpy(np.clip(delta_t / 15.0, 0, 10)).unsqueeze(-1)

    features = torch.cat([feat_interp, feat_mask, feat_dt], dim=-1)
    
    return features, torch.from_numpy(batch_xy_norm), torch.from_numpy(batch_mask), torch.from_numpy(batch_padding), lengths

def get_dataloader(pkl_path, batch_size, shuffle=True, augment=True):
    ds = TrajectoryDataset(pkl_path, augment=augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)