import argparse
import importlib
import pickle
import sys

import numpy as np


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


def evaluate(preds, inputs, gt):
    all_dist = []
    for pred, inp, truth in zip(preds, inputs, gt):
        pred_coords = np.asarray(pred["coords"], dtype=float)
        gt_coords = np.asarray(truth["coords"], dtype=float)
        miss = ~np.asarray(inp["mask"], dtype=bool)
        dist = haversine(
            pred_coords[miss, 0],
            pred_coords[miss, 1],
            gt_coords[miss, 0],
            gt_coords[miss, 1],
        )
        if len(dist):
            all_dist.append(dist)
    if not all_dist:
        return None
    dist = np.concatenate(all_dist)
    return {
        "n_points": int(len(dist)),
        "mae": float(np.mean(dist)),
        "rmse": float(np.sqrt(np.mean(dist**2))),
        "p50": float(np.percentile(dist, 50)),
        "p75": float(np.percentile(dist, 75)),
        "p90": float(np.percentile(dist, 90)),
        "p95": float(np.percentile(dist, 95)),
        "p99": float(np.percentile(dist, 99)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Prediction pkl to inspect.")
    parser.add_argument("--input", help="Original sparse input pkl, needed for scoring.")
    parser.add_argument("--gt", help="Ground truth pkl, needed for scoring.")
    parser.add_argument("--head", type=int, default=2, help="How many examples to print.")
    return parser.parse_args()


def main():
    args = parse_args()
    preds = load_pickle(args.pred)
    print(f"pred_file: {args.pred}")
    print(f"num_trajectories: {len(preds)}")
    if preds:
        print(f"keys: {list(preds[0].keys())}")
    for idx, item in enumerate(preds[: args.head]):
        coords = np.asarray(item["coords"], dtype=float)
        print(
            f"example[{idx}] traj_id={item.get('traj_id')} "
            f"coords_shape={coords.shape} first3={coords[:3].tolist()}"
        )

    if args.input and args.gt:
        inputs = load_pickle(args.input)
        gt = load_pickle(args.gt)
        metrics = evaluate(preds, inputs, gt)
        if metrics is not None:
            print(
                "metrics: "
                f"MAE={metrics['mae']:.2f}m "
                f"RMSE={metrics['rmse']:.2f}m "
                f"p50={metrics['p50']:.2f}m "
                f"p75={metrics['p75']:.2f}m "
                f"p90={metrics['p90']:.2f}m "
                f"p95={metrics['p95']:.2f}m "
                f"p99={metrics['p99']:.2f}m "
                f"n_points={metrics['n_points']}"
            )
    elif args.input or args.gt:
        print("Scoring skipped: provide both --input and --gt.")


if __name__ == "__main__":
    main()
