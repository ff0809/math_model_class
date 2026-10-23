from pathlib import Path
import math
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    import numpy.core as np_core
    import numpy.core.numeric as np_core_numeric

    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.numeric", np_core_numeric)
except Exception:
    pass


ROOT = Path("student_release")
OUT = Path("task_b_solution/report_figures")
PRED_PATH = Path("task_b_solution/val_pred_transformer_30e_min20.pkl")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def haversine_vec(coords):
    coords = np.asarray(coords, dtype=np.float64)
    if len(coords) <= 1:
        return np.zeros(0)
    lon1 = np.radians(coords[:-1, 0])
    lat1 = np.radians(coords[:-1, 1])
    lon2 = np.radians(coords[1:, 0])
    lat2 = np.radians(coords[1:, 1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def savefig(name):
    OUT.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=220)
    plt.close()
    print(OUT / name)


def main():
    val_input = load_pickle(ROOT / "task_B_tte" / "val_input.pkl")
    val_gt = load_pickle(ROOT / "task_B_tte" / "val_gt.pkl")
    pred = load_pickle(PRED_PATH)

    y_true = np.asarray([x["travel_time"] for x in val_gt], dtype=float)
    y_pred = np.asarray([x["travel_time"] for x in pred], dtype=float)
    lengths = np.asarray([len(x["coords"]) for x in val_input], dtype=int)
    distances_km = np.asarray([haversine_vec(x["coords"]).sum() / 1000 for x in val_input], dtype=float)
    errors = y_pred - y_true

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(val_input), size=min(3500, len(val_input)), replace=False)

    plt.figure(figsize=(7.2, 4.4))
    plt.hist(lengths, bins=40, color="#376795", edgecolor="white")
    plt.xlabel("Number of GPS points")
    plt.ylabel("Number of trajectories")
    plt.title("Distribution of trajectory lengths")
    savefig("fig1_point_count_distribution.png")

    plt.figure(figsize=(7.2, 4.4))
    plt.hist(y_true, bins=45, color="#6f9f4d", edgecolor="white")
    plt.xlabel("Travel time (s)")
    plt.ylabel("Number of trajectories")
    plt.title("Distribution of travel time")
    savefig("fig2_travel_time_distribution.png")

    plt.figure(figsize=(7.2, 4.8))
    plt.scatter(lengths[sample_idx], y_true[sample_idx], s=8, alpha=0.28, color="#2f5f98")
    plt.xlabel("Number of GPS points")
    plt.ylabel("Travel time (s)")
    plt.title("Trajectory length vs. travel time")
    savefig("fig3_point_count_vs_time.png")

    plt.figure(figsize=(7.2, 4.8))
    plt.scatter(distances_km[sample_idx], y_true[sample_idx], s=8, alpha=0.28, color="#8f5d3d")
    plt.xlabel("Path distance (km)")
    plt.ylabel("Travel time (s)")
    plt.title("Path distance vs. travel time")
    savefig("fig4_distance_vs_time.png")

    plt.figure(figsize=(5.6, 5.6))
    plt.scatter(y_true[sample_idx], y_pred[sample_idx], s=8, alpha=0.30, color="#3f7f6f")
    low = min(y_true.min(), y_pred.min())
    high = max(y_true.max(), y_pred.max())
    plt.plot([low, high], [low, high], color="#333333", linewidth=1.2)
    plt.xlabel("Ground truth travel time (s)")
    plt.ylabel("Predicted travel time (s)")
    plt.title("Predicted vs. ground truth travel time")
    savefig("fig5_pred_vs_gt.png")

    plt.figure(figsize=(7.2, 4.4))
    plt.hist(errors, bins=70, color="#8d6ca9", edgecolor="white")
    plt.axvline(0, color="#222222", linewidth=1)
    plt.xlabel("Prediction error (s)")
    plt.ylabel("Number of trajectories")
    plt.title("Distribution of prediction errors")
    savefig("fig6_error_distribution.png")

    bins = [0, 900, 1200, 1500, 1800, math.inf]
    labels = ["<900", "900-1200", "1200-1500", "1500-1800", ">1800"]
    maes = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_true >= lo) & (y_true < hi)
        maes.append(float(np.mean(np.abs(errors[mask]))) if mask.any() else 0.0)
    plt.figure(figsize=(7.2, 4.4))
    plt.bar(labels, maes, color="#c07b4d")
    plt.xlabel("Ground truth travel time bucket (s)")
    plt.ylabel("MAE (s)")
    plt.title("MAE by travel-time bucket")
    savefig("fig7_mae_by_time_bucket.png")

    val_mae = [
        17.037,
        16.352,
        16.146,
        16.074,
        16.150,
        16.294,
        16.181,
        15.999,
        16.169,
        16.029,
        15.986,
        16.079,
        16.035,
        16.031,
        16.106,
        16.224,
        16.041,
        16.111,
        16.158,
        16.237,
        16.185,
    ]
    epochs = np.arange(1, len(val_mae) + 1)
    plt.figure(figsize=(7.2, 4.4))
    plt.plot(epochs, val_mae, marker="o", color="#375c8d")
    best = int(np.argmin(val_mae))
    plt.scatter([epochs[best]], [val_mae[best]], color="#c23b22", zorder=3)
    plt.text(epochs[best] + 0.3, val_mae[best] + 0.01, f"best: {val_mae[best]:.3f}s")
    plt.xlabel("Epoch")
    plt.ylabel("Validation MAE (s)")
    plt.title("Validation MAE during training")
    savefig("fig8_training_curve.png")

    plt.figure(figsize=(6.8, 6.2))
    for idx in rng.choice(len(val_input), size=45, replace=False):
        coords = np.asarray(val_input[idx]["coords"])
        plt.plot(coords[:, 0], coords[:, 1], linewidth=0.8, alpha=0.45)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Random validation trajectories")
    savefig("fig9_sample_trajectories.png")

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors) / np.maximum(y_true, 1)) * 100
    print(f"MAE={mae:.3f} RMSE={rmse:.3f} MAPE={mape:.3f}%")


if __name__ == "__main__":
    main()
