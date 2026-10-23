# inference.py
import pickle
import numpy as np
import torch
import os
from config import Config
from model import TrajectoryRecoveryTransformer
from utils import haversine_distance, lonlat_to_local_xy, local_xy_to_lonlat

def prepare_input_batch(trajectories):
    # ===== 新增：检查是否包含路网特征 =====
    missing_road = any('road_offset_x' not in t for t in trajectories)
    if missing_road:
        print("⚠️ 警告：输入数据中缺少 'road_offset_x' 字段，路网特征将全为零，修复精度可能大幅下降！")
        print("   请先用 precompute_road_offset.py 对输入文件进行预处理。")
    # =====================================

    max_len = max(len(t['coords']) for t in trajectories)
    B = len(trajectories)

    batch_interp_xy = np.zeros((B, max_len, 2), dtype=np.float32)
    batch_mask = np.zeros((B, max_len), dtype=bool)
    batch_padding = np.ones((B, max_len), dtype=bool)
    batch_times = np.zeros((B, max_len), dtype=np.float32)

    for i, t in enumerate(trajectories):
        L = len(t['coords'])
        coords = np.array(t['coords'], dtype=np.float32)
        mask = np.array(t['mask'], dtype=bool)
        times = np.array(t['timestamps'], dtype=np.float32)

        x, y = lonlat_to_local_xy(coords[:, 0], coords[:, 1],
                                  Config.REF_LON, Config.REF_LAT)

        known_idx = np.where(mask)[0]
        if len(known_idx) >= 2:
            batch_interp_xy[i, :L, 0] = np.interp(np.arange(L), known_idx, x[mask])
            batch_interp_xy[i, :L, 1] = np.interp(np.arange(L), known_idx, y[mask])
        else:
            batch_interp_xy[i, :L, :] = 0.0

        batch_mask[i, :L] = mask
        batch_padding[i, :L] = False
        batch_times[i, :L] = times

    feat_interp = torch.from_numpy(batch_interp_xy / Config.XY_STD)
    feat_mask = torch.from_numpy(batch_mask.astype(np.float32)).unsqueeze(-1)

    # 时间差特征
    delta_t = np.zeros((B, max_len), dtype=np.float32)
    for i in range(B):
        L = len(trajectories[i]['coords'])
        if L > 1:
            delta_t[i, 1:L] = batch_times[i, 1:L] - batch_times[i, :L-1]
    feat_dt = torch.from_numpy(np.clip(delta_t, 0, 3600) / 60.0).unsqueeze(-1)

    # 直接读取预计算的路网偏移（若缺失则保持为 0）
    road_offset_x = np.zeros((B, max_len), dtype=np.float32)
    road_offset_y = np.zeros((B, max_len), dtype=np.float32)
    for i, t in enumerate(trajectories):
        L = len(t['coords'])
        if 'road_offset_x' in t and 'road_offset_y' in t:
            road_offset_x[i, :L] = t['road_offset_x']
            road_offset_y[i, :L] = t['road_offset_y']

    feat_road_x = torch.from_numpy(road_offset_x).unsqueeze(-1)
    feat_road_y = torch.from_numpy(road_offset_y).unsqueeze(-1)

    features = torch.cat([feat_interp, feat_mask, feat_dt, feat_road_x, feat_road_y], dim=-1)
    batch_padding_t = torch.from_numpy(batch_padding)

    return features, batch_padding_t, [len(t['coords']) for t in trajectories], trajectories


def velocity_constraint(coords, timestamps, mask_known, max_speed_kph=150):
    N = len(coords)
    if N < 3:
        return coords
    fixed = coords.copy()
    original = coords.copy()
    for i in range(1, N-1):
        if mask_known[i]:
            continue
        dt1 = timestamps[i] - timestamps[i-1]
        dt2 = timestamps[i+1] - timestamps[i]
        if dt1 <= 0 or dt2 <= 0:
            continue
        d1 = haversine_distance(fixed[i-1,0], fixed[i-1,1], fixed[i,0], fixed[i,1])
        d2 = haversine_distance(fixed[i,0], fixed[i,1], fixed[i+1,0], fixed[i+1,1])
        speed1, speed2 = d1/dt1*3.6, d2/dt2*3.6
        if speed1 > max_speed_kph or speed2 > max_speed_kph:
            alpha = dt1 / (dt1 + dt2)
            fixed[i, 0] = fixed[i-1, 0] + alpha * (fixed[i+1, 0] - fixed[i-1, 0])
            fixed[i, 1] = fixed[i-1, 1] + alpha * (fixed[i+1, 1] - fixed[i-1, 1])
    fixed[mask_known] = original[mask_known]
    return fixed


def infer_and_save(input_pkl, output_pkl, model_path, device):
    with open(input_pkl, 'rb') as f:
        input_data = pickle.load(f)

    model = TrajectoryRecoveryTransformer(
        input_dim=Config.input_dim,
        d_model=Config.d_model,
        nhead=Config.nhead,
        num_layers=Config.num_layers,
        dim_feedforward=Config.dim_feedforward,
        dropout=Config.dropout
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    results = []
    for idx in range(0, len(input_data), Config.batch_size):
        batch_trajs = input_data[idx : idx + Config.batch_size]
        features, padding_mask, lengths, raw_trajs = prepare_input_batch(batch_trajs)

        features = features.to(device)
        padding_mask = padding_mask.to(device)

        with torch.no_grad():
            pred_delta_norm = model(features, src_key_padding_mask=padding_mask)
            pred_norm = (features[..., :2] + pred_delta_norm).cpu().numpy()

        for i, traj in enumerate(raw_trajs):
            L = lengths[i]
            pred_x = pred_norm[i, :L, 0] * Config.XY_STD
            pred_y = pred_norm[i, :L, 1] * Config.XY_STD

            p_lon, p_lat = local_xy_to_lonlat(pred_x, pred_y, Config.REF_LON, Config.REF_LAT)

            final_coords = np.array(traj['coords'], dtype=float).copy()
            mask_known = np.array(traj['mask'], dtype=bool)
            unknown_mask = ~mask_known
            final_coords[unknown_mask, 0] = p_lon[unknown_mask]
            final_coords[unknown_mask, 1] = p_lat[unknown_mask]

            if Config.APPLY_VELOCITY_CONSTRAINT:
                final_coords = velocity_constraint(final_coords, np.array(traj['timestamps']), mask_known)
            final_coords[mask_known] = np.array(traj['coords'], dtype=float)[mask_known]

            results.append({
                'traj_id': traj['traj_id'],
                'coords': final_coords.tolist()
            })

    with open(output_pkl, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved predictions to {output_pkl}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message=".*nested tensor.*")
    device = torch.device(Config.device)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    infer_and_save(Config.VAL_INPUT_8, Config.VAL_PRED_8,
                   Config.MODEL_SAVE_PATH, device)
    infer_and_save(Config.VAL_INPUT_16, Config.VAL_PRED_16,
                   Config.MODEL_SAVE_PATH, device)
    print("Inference finished.")
