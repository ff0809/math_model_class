import pickle
import numpy as np
from utils import haversine_distance

def evaluate(pred_file, input_file, gt_file):
    with open(pred_file, 'rb') as f:
        pred_data = pickle.load(f)
    with open(input_file, 'rb') as f:
        input_data = pickle.load(f)
    with open(gt_file, 'rb') as f:
        gt_data = pickle.load(f)

    all_mae = []
    all_rmse = []
    for pred_traj, inp_traj, gt_traj in zip(pred_data, input_data, gt_data):
        pred_coords = np.array(pred_traj['coords'])
        gt_coords = np.array(gt_traj['coords'])
        mask_pred = ~np.array(inp_traj['mask'])   # True 表示待预测点

        dists = []
        for i in range(len(pred_coords)):
            if mask_pred[i]:
                d = haversine_distance(pred_coords[i, 0], pred_coords[i, 1],
                                       gt_coords[i, 0], gt_coords[i, 1])
                dists.append(d)
        if dists:
            dists = np.array(dists)
            all_mae.append(np.mean(dists))
            all_rmse.append(np.sqrt(np.mean(dists**2)))
    
    total_mae = np.mean(all_mae) if all_mae else 0
    total_rmse = np.mean(all_rmse) if all_rmse else 0
    return total_mae, total_rmse

if __name__ == "__main__":
    mae8, rmse8 = evaluate(
        "outputs/val_pred_8.pkl",
        "task_A_recovery/val_input_8.pkl",
        "task_A_recovery/val_gt.pkl"
    )
    print(f"[1/8] MAE: {mae8:.2f} m, RMSE: {rmse8:.2f} m")

    mae16, rmse16 = evaluate(
        "outputs/val_pred_16.pkl",
        "task_A_recovery/val_input_16.pkl",
        "task_A_recovery/val_gt.pkl"
    )
    print(f"[1/16] MAE: {mae16:.2f} m, RMSE: {rmse16:.2f} m")