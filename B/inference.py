import pickle, numpy as np, torch, os
from config import Config
from model import TrajectoryRecoveryTransformer
from utils import haversine_distance, lonlat_to_local_xy, local_xy_to_lonlat
from road_network import get_road_network

def prepare_input_batch(trajectories):
    max_len = max(len(t['coords']) for t in trajectories)
    B = len(trajectories)
    batch_interp = np.zeros((B, max_len, 2), dtype=np.float32)
    batch_mask   = np.zeros((B, max_len), dtype=bool)
    batch_pad    = np.ones((B, max_len), dtype=bool)
    batch_times  = np.zeros((B, max_len), dtype=np.float32)

    for i, t in enumerate(trajectories):
        L = len(t['coords'])
        coords = np.array(t['coords'], dtype=np.float32)
        mask   = np.array(t['mask'], dtype=bool)
        times  = np.array(t['timestamps'], dtype=np.float32)

        x, y = lonlat_to_local_xy(coords[:,0], coords[:,1], Config.REF_LON, Config.REF_LAT)
        known = np.where(mask)[0]
        if len(known) >= 2:
            batch_interp[i,:L,0] = np.interp(np.arange(L), known, x[mask])
            batch_interp[i,:L,1] = np.interp(np.arange(L), known, y[mask])
        else:
            batch_interp[i,:L,:] = 0.0
        batch_mask[i,:L] = mask
        batch_pad[i,:L]  = False
        batch_times[i,:L] = times

    batch_interp = np.nan_to_num(batch_interp, nan=0.0)

    # 路网偏移（推理时也需要）
    rn = get_road_network(Config.ROAD_PATH, Config.REF_LON, Config.REF_LAT)
    road_ox = np.zeros((B, max_len), dtype=np.float32)
    road_oy = np.zeros((B, max_len), dtype=np.float32)
    if rn.kdtree is not None:
        for i in range(B):
            L = len(trajectories[i]['coords'])
            pts = batch_interp[i,:L,:]
            dists, idxs = rn.kdtree.query(pts)
            nodes = rn.road_positions[idxs]
            road_ox[i,:L] = (nodes[:,0] - pts[:,0]) / Config.ROAD_STD
            road_oy[i,:L] = (nodes[:,1] - pts[:,1]) / Config.ROAD_STD

    delta_t = np.zeros((B, max_len), dtype=np.float32)
    for i in range(B):
        L = len(trajectories[i]['coords'])
        if L > 1:
            delta_t[i,1:L] = batch_times[i,1:L] - batch_times[i,:L-1]

    feat_interp = torch.from_numpy(batch_interp / Config.XY_STD)
    feat_mask   = torch.from_numpy(batch_mask.astype(np.float32)).unsqueeze(-1)
    feat_dt     = torch.from_numpy(np.clip(delta_t, 0, 3600)/60.0).unsqueeze(-1)
    feat_rx     = torch.from_numpy(road_ox).unsqueeze(-1)
    feat_ry     = torch.from_numpy(road_oy).unsqueeze(-1)

    features = torch.cat([feat_interp, feat_mask, feat_dt, feat_rx, feat_ry], dim=-1)
    return features, torch.from_numpy(batch_pad), [len(t['coords']) for t in trajectories], trajectories, batch_interp


def infer_and_save(input_pkl, output_pkl, model_path, device):
    with open(input_pkl, 'rb') as f:
        data = pickle.load(f)

    model = TrajectoryRecoveryTransformer(
        input_dim=Config.input_dim, d_model=Config.d_model,
        nhead=Config.nhead, num_layers=Config.num_layers,
        dim_feedforward=Config.dim_feedforward, dropout=Config.dropout
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    results = []
    for idx in range(0, len(data), Config.batch_size):
        batch = data[idx: idx+Config.batch_size]
        features, pad_mask, lengths, raw, interp = prepare_input_batch(batch)
        features = features.to(device)
        pad_mask = pad_mask.to(device)

        with torch.no_grad():
            pred_res = model(features, src_key_padding_mask=pad_mask).cpu().numpy()

        for i, traj in enumerate(raw):
            L = lengths[i]
            pred_x = interp[i,:L,0] + pred_res[i,:L,0] * Config.XY_STD
            pred_y = interp[i,:L,1] + pred_res[i,:L,1] * Config.XY_STD
            p_lon, p_lat = local_xy_to_lonlat(pred_x, pred_y, Config.REF_LON, Config.REF_LAT)

            final_coords = np.array(traj['coords'], dtype=float).copy()
            unknown = ~np.array(traj['mask'], dtype=bool)
            final_coords[unknown,0] = p_lon[unknown]
            final_coords[unknown,1] = p_lat[unknown]

            # 可选速度约束（仅缺失点）
            # final_coords = apply_velocity_constraint(...)

            results.append({'traj_id': traj['traj_id'], 'coords': final_coords.tolist()})

    with open(output_pkl, 'wb') as f:
        pickle.dump(results, f)
    print(f"保存至 {output_pkl}")

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    device = torch.device(Config.device)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    infer_and_save(Config.VAL_INPUT_8, "outputs/val_pred_8.pkl", Config.MODEL_SAVE_PATH, device)
    infer_and_save(Config.VAL_INPUT_16, "outputs/val_pred_16.pkl", Config.MODEL_SAVE_PATH, device)