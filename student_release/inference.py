import pickle
import numpy as np
import torch
import os
from config import Config
from model import TrajectoryRecoveryTransformer
from utils import haversine_distance, lonlat_to_local_xy, local_xy_to_lonlat

def prepare_input_batch(trajectories):
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

        x, y = lonlat_to_local_xy(coords[:, 0], coords[:, 1], Config.REF_LON, Config.REF_LAT)

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

    delta_t = np.zeros((B, max_len), dtype=np.float32)
    delta_t[:, 1:] = batch_times[:, 1:] - batch_times[:, :-1]
    feat_dt = torch.from_numpy(np.clip(delta_t / 15.0, 0, 10)).unsqueeze(-1)

    features = torch.cat([feat_interp, feat_mask, feat_dt], dim=-1)
    batch_padding_t = torch.from_numpy(batch_padding)
    
    return features, batch_padding_t, [len(t['coords']) for t in trajectories], trajectories

def velocity_constraint(coords, timestamps, max_speed_kph=150):
    N = len(coords)
    if N < 3: return coords
    fixed = coords.copy()
    for i in range(1, N-1):
        dt1 = timestamps[i] - timestamps[i-1]
        dt2 = timestamps[i+1] - timestamps[i]
        if dt1 <= 0 or dt2 <= 0: continue
        d1 = haversine_distance(fixed[i-1,0], fixed[i-1,1], fixed[i,0], fixed[i,1])
        d2 = haversine_distance(fixed[i,0], fixed[i,1], fixed[i+1,0], fixed[i+1,1])
        speed1, speed2 = d1/dt1*3.6, d2/dt2*3.6
        if speed1 > max_speed_kph or speed2 > max_speed_kph:
            alpha = dt1 / (dt1 + dt2)
            fixed[i, 0] = fixed[i-1, 0] + alpha * (fixed[i+1, 0] - fixed[i-1, 0])
            fixed[i, 1] = fixed[i-1, 1] + alpha * (fixed[i+1, 1] - fixed[i-1, 1])
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
            pred_norm = model(features, src_key_padding_mask=padding_mask).cpu().numpy()

        for i, traj in enumerate(raw_trajs):
            L = lengths[i]

            pred_x = pred_norm[i, :L, 0] * Config.XY_STD
            pred_y = pred_norm[i, :L, 1] * Config.XY_STD

            p_lon, p_lat = local_xy_to_lonlat(pred_x, pred_y, Config.REF_LON, Config.REF_LAT)
            
            final_coords = np.array(traj['coords'], dtype=float).copy()
            unknown_mask = ~np.array(traj['mask'], dtype=bool)
            final_coords[unknown_mask, 0] = p_lon[unknown_mask]
            final_coords[unknown_mask, 1] = p_lat[unknown_mask]
            
            final_coords = velocity_constraint(final_coords, np.array(traj['timestamps']))
            
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
    infer_and_save(Config.VAL_INPUT_8, "outputs/val_pred_8.pkl", Config.MODEL_SAVE_PATH, device)
    infer_and_save(Config.VAL_INPUT_16, "outputs/val_pred_16.pkl", Config.MODEL_SAVE_PATH, device)
    print("Inference finished.")