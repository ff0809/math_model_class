import os
import argparse
import pickle
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plot_results as pr
except Exception:
    pr = None


def safe_load_pickle(path: str):
    import sys
    try:
        import numpy as _np
    except Exception:
        _np = None
    if _np is not None:
        try:
            if 'numpy._core.numeric' not in sys.modules and hasattr(_np, 'core'):
                sys.modules['numpy._core.numeric'] = _np.core.numeric
        except Exception:
            pass

    with open(path, 'rb') as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError:
            f.seek(0)
            return pickle.load(f, encoding='latin1')


def extract_coords(obj) -> List[Tuple[float, float]]:
    if obj is None:
        return []
    if isinstance(obj, dict):
        for k in ('interpolated', 'interp', 'pred', 'pred_coords', 'coords', 'trajectory', 'points'):
            if k in obj:
                return extract_coords(obj[k])
        if 'coords' in obj and 'mask' in obj:
            coords = np.array(obj['coords'])
            mask = np.array(obj['mask']).astype(bool)
            sampled = coords[mask]
            return [tuple(map(float, p[:2])) for p in sampled]
        return []
    try:
        import numpy as _np
        if isinstance(obj, _np.ndarray):
            arr = obj
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return [tuple(map(float, row[:2])) for row in arr]
            return [tuple(map(float, x)) for x in arr]
    except Exception:
        pass
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        out = []
        for x in obj:
            try:
                out.append((float(x[0]), float(x[1])))
            except Exception:
                continue
        return out
    return []


def haversine_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lon1, lat1 = a
    lon2, lat2 = b
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    x = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(x))


def plot_interpolation(sample_pts: List[Tuple[float, float]], interp_pts: List[Tuple[float, float]],
                       gt_pts: Optional[List[Tuple[float, float]]], outpath: str, idx: int):
    plt.figure(figsize=(10, 8))
    ax_map = plt.subplot(2, 1, 1)
    # map
    if gt_pts:
        lons_g = [p[0] for p in gt_pts]
        lats_g = [p[1] for p in gt_pts]
        ax_map.plot(lons_g, lats_g, color='green', linestyle='--', linewidth=1.5, label='Ground truth')
    if interp_pts:
        lons_i = [p[0] for p in interp_pts]
        lats_i = [p[1] for p in interp_pts]
        ax_map.plot(lons_i, lats_i, color='blue', linewidth=1.5, label='Interpolated (linear)')
    if sample_pts:
        lons_s = [p[0] for p in sample_pts]
        lats_s = [p[1] for p in sample_pts]
        ax_map.scatter(lons_s, lats_s, c='red', s=40, zorder=5, label='Sampled')

    ax_map.set_aspect('equal', adjustable='datalim')
    if pr and getattr(pr, 'FONT_PROP', None):
        ax_map.set_xlabel('Longitude', fontproperties=pr.FONT_PROP)
        ax_map.set_ylabel('Latitude', fontproperties=pr.FONT_PROP)
        ax_map.set_title(f'Interpolation result #{idx}', fontproperties=pr.FONT_PROP)
        ax_map.legend(prop=pr.FONT_PROP)
    else:
        ax_map.set_xlabel('Longitude')
        ax_map.set_ylabel('Latitude')
        ax_map.set_title(f'Interpolation result #{idx}')
        ax_map.legend()

    ax_err = plt.subplot(2, 1, 2)
    if gt_pts and interp_pts and len(gt_pts) == len(interp_pts):
        dists = [haversine_dist(gt_pts[i], interp_pts[i]) for i in range(len(gt_pts))]
        ax_err.plot(dists, color='purple', label='Pointwise error (m)')
        ax_err.set_ylabel('Error (m)')
        ax_err.set_xlabel('Timestep')
        ax_err.grid(True, alpha=0.3)
        if pr and getattr(pr, 'FONT_PROP', None):
            ax_err.legend(prop=pr.FONT_PROP)
        else:
            ax_err.legend()
    else:
        ax_err.text(0.5, 0.5, 'Ground-truth not available or length mismatch; error plot skipped',
                    ha='center', va='center')
        ax_err.set_axis_off()

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print('Saved', outpath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', '-s', default='student_release/task_A_recovery/val_input_8.pkl', help='sampled input pickle (may contain coords+mask)')
    parser.add_argument('--interp', '-i', default='submission_val_input_8.pkl', help='interpolated/predicted pickle')
    parser.add_argument('--gt', '-g', default='student_release/task_A_recovery/val_gt.pkl', help='ground-truth pickle (optional)')
    parser.add_argument('--index', type=int, default=0, help='index of trajectory to plot')
    parser.add_argument('--outdir', '-o', default='figures/interpolation', help='output directory')
    parser.add_argument('--font-path', type=str, default=None, help='font file path (optional)')
    args = parser.parse_args()

    if pr and hasattr(pr, 'configure_chinese_font'):
        pr.configure_chinese_font(args.font_path)

    sample_data = safe_load_pickle(args.sample)
    interp_data = safe_load_pickle(args.interp)
    gt_data = None
    try:
        gt_data = safe_load_pickle(args.gt)
    except Exception:
        gt_data = None

    def pick(d, idx):
        try:
            return d[idx]
        except Exception:
            return None

    s = pick(sample_data, args.index)
    it = pick(interp_data, args.index)
    gt = pick(gt_data, args.index) if gt_data is not None else None

    sample_pts = extract_coords(s)
    interp_pts = extract_coords(it)
    gt_pts = extract_coords(gt) if gt is not None else None

    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, f'interp_{args.index:03d}.png')
    plot_interpolation(sample_pts, interp_pts, gt_pts, outpath, args.index)


if __name__ == '__main__':
    sns.set(style='whitegrid')
    main()
