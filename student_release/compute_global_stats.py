import pickle, numpy as np
with open('data_ds15/train.pkl', 'rb') as f:
    train = pickle.load(f)
all_lons, all_lats = [], []
for traj in train:
    coords = np.array(traj['coords'])
    all_lons.extend(coords[:, 0])
    all_lats.extend(coords[:, 1])
print(f"GLOBAL_MEAN_LON = {np.mean(all_lons):.6f}")
print(f"GLOBAL_MEAN_LAT = {np.mean(all_lats):.6f}")
print(f"GLOBAL_STD_LON = {np.std(all_lons):.6f}")
print(f"GLOBAL_STD_LAT = {np.std(all_lats):.6f}")