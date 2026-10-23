import importlib
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer


REF_LON = 108.946518
REF_LAT = 34.244678


def proj4():
    return (
        f"+proj=tmerc +lat_0={REF_LAT} +lon_0={REF_LON} +k=1 "
        "+x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
    )


TO_XY = Transformer.from_crs("EPSG:4326", proj4(), always_xy=True)
TO_LONLAT = Transformer.from_crs(proj4(), "EPSG:4326", always_xy=True)


def load_pickle(path):
    for alias in ("numpy._core", "numpy._core._multiarray_umath"):
        if alias not in sys.modules:
            try:
                sys.modules[alias] = importlib.import_module(
                    alias.replace("numpy._core", "numpy.core")
                )
            except Exception:
                pass
    with open(path, "rb") as f:
        return pickle.load(f)


def haversine(lon1, lat1, lon2, lat2):
    radius = 6371000.0
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def linear_prediction(inp):
    coords = np.asarray(inp["coords"], dtype=float)
    mask = np.asarray(inp["mask"], dtype=bool)
    x, y = TO_XY.transform(coords[:, 0], coords[:, 1])
    x = np.asarray(x)
    y = np.asarray(y)
    known = np.where(mask)[0]
    t = np.arange(len(coords))
    px = np.interp(t, known, x[known])
    py = np.interp(t, known, y[known])
    lon, lat = TO_LONLAT.transform(px, py)
    out = np.column_stack([lon, lat])
    out[mask] = coords[mask]
    return out


def collect_errors(preds, inputs, gt):
    errors = []
    for pred, inp, truth in zip(preds, inputs, gt):
        pred_coords = np.asarray(pred["coords"], dtype=float)
        gt_coords = np.asarray(truth["coords"], dtype=float)
        miss = ~np.asarray(inp["mask"], dtype=bool)
        d = haversine(
            pred_coords[miss, 0],
            pred_coords[miss, 1],
            gt_coords[miss, 0],
            gt_coords[miss, 1],
        )
        errors.append(d)
    return np.concatenate(errors)


def main():
    out_dir = Path("task_a_retrieval/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_pickle("student_release/task_A_recovery/val_input_8.pkl")
    gt = load_pickle("student_release/task_A_recovery/val_gt.pkl")
    pred_path = Path("task_a_retrieval/outputs/val_pred_8_fulltrain.pkl")
    if not pred_path.exists():
        pred_path = Path("/tmp/retrieval_timing_8_full_cached.pkl")
    preds = load_pickle(pred_path)

    gap_counts = {}
    for traj in inputs:
        mask = np.asarray(traj["mask"], dtype=bool)
        known = np.where(mask)[0]
        for left, right in zip(known[:-1], known[1:]):
            gap = int(right - left)
            gap_counts[gap] = gap_counts.get(gap, 0) + 1

    gaps = sorted(gap_counts)
    counts = [gap_counts[g] for g in gaps]
    plt.figure(figsize=(7, 4))
    plt.bar(gaps, counts, color="#4C78A8")
    plt.xlabel("Gap length between known points")
    plt.ylabel("Number of gaps")
    plt.title("Task A val_input_8 gap distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "fig1_gap_distribution.png", dpi=180)
    plt.close()

    retrieval_errors = collect_errors(preds, inputs, gt)

    linear_preds = []
    for inp in inputs:
        linear_preds.append({"traj_id": inp["traj_id"], "coords": linear_prediction(inp).tolist()})
    linear_errors = collect_errors(linear_preds, inputs, gt)

    methods = ["Linear", "Road A*", "Retrieval"]
    mae = [linear_errors.mean(), 94.12, retrieval_errors.mean()]
    rmse = [
        np.sqrt(np.mean(linear_errors**2)),
        130.01,
        np.sqrt(np.mean(retrieval_errors**2)),
    ]
    x = np.arange(len(methods))
    width = 0.36
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, mae, width, label="MAE", color="#4C78A8")
    plt.bar(x + width / 2, rmse, width, label="RMSE", color="#F58518")
    plt.xticks(x, methods)
    plt.ylabel("Meters")
    plt.title("Task A val_input_8 method comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig2_method_comparison.png", dpi=180)
    plt.close()

    clipped = np.clip(retrieval_errors, 0, 300)
    plt.figure(figsize=(7, 4))
    plt.hist(clipped, bins=60, color="#54A24B", alpha=0.85)
    plt.xlabel("Error distance (m), clipped at 300m")
    plt.ylabel("Predicted points")
    plt.title("Retrieval error distribution on val_input_8")
    plt.tight_layout()
    plt.savefig(out_dir / "fig3_error_distribution.png", dpi=180)
    plt.close()

    percentiles = np.percentile(retrieval_errors, [50, 75, 90, 95, 99])
    print("linear_mae", float(linear_errors.mean()))
    print("linear_rmse", float(np.sqrt(np.mean(linear_errors**2))))
    print("retrieval_mae", float(retrieval_errors.mean()))
    print("retrieval_rmse", float(np.sqrt(np.mean(retrieval_errors**2))))
    print("retrieval_percentiles", [float(x) for x in percentiles])
    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
