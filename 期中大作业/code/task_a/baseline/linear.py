import pickle
import numpy as np
import pandas as pd

DATA_DIR = "./student_release/task_A_recovery/"

def load_data():
    """加载验证集数据"""
    try:
        import sys as _sys
        import numpy as _np
        if 'numpy._core' not in _sys.modules:
            _sys.modules['numpy._core'] = _np.core
    except Exception:
        pass
    with open(DATA_DIR + "val_input_16.pkl", "rb") as f:
        input_data = pickle.load(f)
    with open(DATA_DIR + "val_gt.pkl", "rb") as f:
        gt_data = pickle.load(f)
    return input_data, gt_data

class TrajectoryRecovery:
    def __init__(self):
        print("简易插值模型初始化完成")

    def reconstruct_path(self, start_coord, end_coord, start_time, end_time, num_missing):
        start_lon, start_lat = start_coord
        end_lon, end_lat = end_coord

        if num_missing == 0:
            return []

        time_ratio = np.linspace(0, 1, num_missing + 2)[1:-1]
        pred_lats = start_lat + time_ratio * (end_lat - start_lat)
        pred_lons = start_lon + time_ratio * (end_lon - start_lon)

        return list(zip(pred_lons, pred_lats))

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    计算两个经纬度点之间的 Haversine 距离（米）
    """
    # 将经纬度转换为弧度
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000
    return c * r

def main():
    input_data, gt_data = load_data()
    
    repair_model = TrajectoryRecovery()
    
    results = []
    
    for traj_idx, traj_dict in enumerate(input_data):
        traj_id = traj_dict['traj_id']
        timestamps = traj_dict['timestamps']
        coords = traj_dict['coords']
        mask = traj_dict['mask'] 
        
        print(f"正在处理轨迹 {traj_id}... (进度: {traj_idx+1}/{len(input_data)})")
        
        coords_array = np.array(coords)
        pred_coords = coords_array.copy() 
        
        i = 0
        while i < len(mask):
            if mask[i]:
                j = i + 1
                while j < len(mask) and not mask[j]:
                    j += 1
                if j < len(mask):
                    start_coord = coords_array[i]
                    end_coord = coords_array[j]
                    num_missing = j - i - 1
                    if num_missing > 0:
                        repaired_segment = repair_model.reconstruct_path(
                            start_coord, end_coord, timestamps[i], timestamps[j], num_missing
                        )
                        for k, coord in enumerate(repaired_segment):
                            pred_coords[i + 1 + k] = coord
                    i = j
                else:
                    i += 1
            else:
                i += 1

        results.append({
            'traj_id': traj_id,
            'coords': pred_coords.tolist()
        })

    print("\n开始计算评测指标...")
    
    gt_dict = {item['traj_id']: item for item in gt_data}
    
    all_errors = [] # 存储所有待预测点的误差
    
    for pred_traj in results:
        traj_id = pred_traj['traj_id']
        pred_coords = np.array(pred_traj['coords'])
        gt_coords = np.array(gt_dict[traj_id]['coords'])
        
        # 获取输入数据中的 mask，找出哪些点是待预测的 (False)
        input_traj = [t for t in input_data if t['traj_id'] == traj_id][0]
        mask = np.array(input_traj['mask'])
        unknown_indices = np.where(~mask)[0]
        
        for idx in unknown_indices:
            pred_lon, pred_lat = pred_coords[idx]
            gt_lon, gt_lat = gt_coords[idx]
            
            # 计算 Haversine 距离
            error = haversine_distance(pred_lon, pred_lat, gt_lon, gt_lat)
            all_errors.append(error)
    
    all_errors = np.array(all_errors)
    MAE = np.mean(all_errors)
    RMSE = np.sqrt(np.mean(all_errors**2))
    
    print(f"\n{'='*40}")
    print(f"评估完成！")
    print(f"平均绝对误差 (MAE): {MAE:.4f} 米")
    print(f"均方根误差 (RMSE): {RMSE:.4f} 米")
    print(f"{'='*40}")
    
    with open("submission_val_input_16.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\n预测结果已保存为 submission_val_input_16.pkl")

if __name__ == "__main__":
    main()