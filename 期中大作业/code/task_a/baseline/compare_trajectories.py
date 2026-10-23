import os
import argparse
import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

import plot_results as pr


def load_results(path: str):
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
        except ModuleNotFoundError as e:
            f.seek(0)
            try:
                return pickle.load(f, encoding='latin1')
            except Exception:
                raise


def get_coords_from_traj(traj):
    if traj is None:
        return []
    if isinstance(traj, dict):
        if 'coords' in traj:
            traj = traj['coords']
        else:
            for key in ('trajectory', 'points', 'pts'):
                if key in traj:
                    traj = traj[key]
                    break

    try:
        import numpy as _np
        if isinstance(traj, _np.ndarray):
            arr = traj
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return [tuple(map(float, row[:2])) for row in arr]
            return arr.tolist()
    except Exception:
        pass

    if hasattr(traj, '__iter__') and not isinstance(traj, (str, bytes)):
        out = []
        for x in traj:
            try:
                out.append((float(x[0]), float(x[1])))
            except Exception:
                try:
                    out.append(tuple(x))
                except Exception:
                    continue
        return out

    return []


def plot_pair(pred, gt, outpath, idx: int):
    pred_coords = get_coords_from_traj(pred) or []
    gt_coords = get_coords_from_traj(gt) or []

    plt.figure(figsize=(10, 6))
    if pred_coords:
        lons_p = [c[0] for c in pred_coords]
        lats_p = [c[1] for c in pred_coords]
        plt.plot(lons_p, lats_p, color='C0', linewidth=1, label='Predicted')
    if gt_coords:
        lons_g = [c[0] for c in gt_coords]
        lats_g = [c[1] for c in gt_coords]
        plt.plot(lons_g, lats_g, color='C2', linestyle='--', linewidth=1, label='Ground truth')

    if pr.FONT_PROP:
        plt.xlabel('Longitude', fontproperties=pr.FONT_PROP)
        plt.ylabel('Latitude', fontproperties=pr.FONT_PROP)
        plt.title(f'Predicted vs Ground-truth trajectory comparison #{idx}', fontproperties=pr.FONT_PROP)
        plt.legend(prop=pr.FONT_PROP)
        ax = plt.gca()
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_fontproperties(pr.FONT_PROP)
    else:
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Predicted vs Ground-truth trajectory comparison #{idx}')
        plt.legend()

    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print('Saved', outpath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', '-p', default='submission_val_input_8.pkl', help='predictions pickle file')
    parser.add_argument('--gt', '-g', default='student_release/task_A_recovery/val_gt.pkl', help='ground-truth pickle file')
    parser.add_argument('--outdir', '-o', default='figures/compare', help='output directory')
    parser.add_argument('--count', type=int, default=10, help='number of trajectory pairs to plot')
    parser.add_argument('--font-path', type=str, default=None, help='font file path to load (optional)')
    args = parser.parse_args()

    pr.configure_chinese_font(args.font_path)

    preds = load_results(args.pred)
    gts = load_results(args.gt)
    os.makedirs(args.outdir, exist_ok=True)

    n = min(args.count, len(preds), len(gts))
    for i in range(n):
        outpath = os.path.join(args.outdir, f'compare_{i:03d}.png')
        plot_pair(preds[i], gts[i], outpath, i)


if __name__ == '__main__':
    sns.set(style='whitegrid')
    main()
