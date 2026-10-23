import pickle
import numpy as np
from config import Config
from utils import haversine_distance

def evaluate(pred_file, input_file, gt_file):
    with open(pred_file, 'rb') as f:
        pred_data = pickle.load(f)
    with open(input_file, 'rb') as f:
        input_data = pickle.load(f)
    with open(gt_file, 'rb') as f:
        gt_data = pickle.load(f)

    all_dists = []
    for pred_traj, inp_traj, gt_traj in zip(pred_data, input_data, gt_data):
        pred_coords = np.array(pred_traj['coords'])
        gt_coords = np.array(gt_traj['coords'])
        mask_pred = ~np.array(inp_traj['mask'])   # True 表示待预测点

        dists = haversine_distance(pred_coords[mask_pred, 0], pred_coords[mask_pred, 1],
                                   gt_coords[mask_pred, 0], gt_coords[mask_pred, 1])
        if len(dists):
            all_dists.append(dists)
    
    if not all_dists:
        return 0, 0
    all_dists = np.concatenate(all_dists)
    total_mae = np.mean(all_dists)
    total_rmse = np.sqrt(np.mean(all_dists**2))
    return total_mae, total_rmse

if __name__ == "__main__":
    mae8, rmse8 = evaluate(
        Config.VAL_PRED_8,
        Config.RAW_VAL_INPUT_8,
        Config.VAL_GT
    )
    print(f"[1/8] MAE: {mae8:.2f} m, RMSE: {rmse8:.2f} m")

    mae16, rmse16 = evaluate(
        Config.VAL_PRED_16,
        Config.RAW_VAL_INPUT_16,
        Config.VAL_GT
    )
    print(f"[1/16] MAE: {mae16:.2f} m, RMSE: {rmse16:.2f} m")
