# plot_sampling_comparison.py
import pickle
import matplotlib.pyplot as plt
import numpy as np
try:
    import plot_results as pr
except Exception:
    pr = None

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 

DATA_DIR = "./student_release/task_A_recovery/"
OUTPUT_PNG = "trajectory_sampling_comparison.png"

def load_data():
    """Load validation inputs (sampled points) and ground-truth (full trajectories)"""
    import sys
    try:
        import numpy as _np
    except Exception:
        _np = None

    # fix old numpy internal module references used in some pickles
    if _np is not None:
        try:
            if 'numpy._core.numeric' not in sys.modules and hasattr(_np, 'core'):
                sys.modules['numpy._core.numeric'] = _np.core.numeric
        except Exception:
            pass

    with open(DATA_DIR + "val_input_8.pkl", "rb") as f:
        try:
            input_data = pickle.load(f)
        except ModuleNotFoundError:
            f.seek(0)
            input_data = pickle.load(f, encoding='latin1')
    with open(DATA_DIR + "val_gt.pkl", "rb") as f:
        try:
            gt_data = pickle.load(f)
        except ModuleNotFoundError:
            f.seek(0)
            gt_data = pickle.load(f, encoding='latin1')
    return input_data, gt_data

def main():
    input_data, gt_data = load_data()
    sample_idx = 0 
    input_traj = input_data[sample_idx]
    gt_traj = gt_data[sample_idx]
    
    traj_id = input_traj['traj_id']
    print(f"Plotting trajectory: {traj_id} ...")

    # 原始完整轨迹
    gt_coords = np.array(gt_traj['coords'])
    gt_lons = gt_coords[:, 0]
    gt_lats = gt_coords[:, 1]

    # 1/8 采样后的轨迹
    input_coords = np.array(input_traj['coords'])
    mask = np.array(input_traj['mask']) # True=保留的点, False=被设为NaN的点

    sampled_coords = input_coords[mask]
    sampled_lons = sampled_coords[:, 0]
    sampled_lats = sampled_coords[:, 1]

    plt.figure(figsize=(14, 8))
    plt.plot(gt_lons, gt_lats, color='lightgray', linewidth=3, alpha=0.7, label='Original full trajectory (100%)')
    plt.scatter(sampled_lons, sampled_lats, color='red', s=100, zorder=5,
                edgecolors='darkred', linewidth=1.5, label='1/8 sampled points')
    plt.plot(sampled_lons, sampled_lats, color='red', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.title(f'Trajectory sampling comparison\nTrajectory ID: {traj_id} | Sampling density: 1/8', fontsize=14, fontweight='bold')
    if pr and getattr(pr, 'FONT_PROP', None):
        plt.legend(loc='best', prop=pr.FONT_PROP)
    else:
        plt.legend(loc='best', fontsize=11)
    
    plt.axis('equal') 
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches='tight')
    print(f"Saved plot as: {OUTPUT_PNG}")

    plt.show()

if __name__ == "__main__":
    main()