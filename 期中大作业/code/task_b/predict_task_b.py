import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from task_b_common import (
    PointCountBaseline,
    ResidualTransformer,
    Standardizer,
    TTEDataset,
    collate_tte,
    evaluate_predictions,
    load_pickle,
    print_metrics,
    save_pickle,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def load_artifact(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict task B travel time.")
    parser.add_argument("--input", default="student_release/task_B_tte/val_input.pkl", help="Input pkl path.")
    parser.add_argument("--output", default="task_b_solution/pred_task_b.pkl", help="Output submission pkl path.")
    parser.add_argument("--model", default="task_b_solution/artifacts/task_b_transformer.pt", help="Model artifact path.")
    parser.add_argument(
        "--train-data",
        default="student_release/data_ds15/train.pkl",
        help="Training pkl path used to build baseline when --baseline-only is set and no model artifact exists.",
    )
    parser.add_argument("--gt", default="", help="Optional ground truth pkl path for validation metrics.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Use only point-count median baseline. Useful before Transformer training finishes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_pickle(args.input)
    show_progress = not args.no_progress

    artifact = None
    model_path = Path(args.model)
    if model_path.exists():
        artifact = load_artifact(model_path)

    if artifact is not None:
        baseline = PointCountBaseline.from_state(artifact["baseline"])
    elif args.baseline_only:
        train_records = load_pickle(args.train_data)
        baseline = PointCountBaseline.fit(train_records, use_median=True)
    else:
        raise FileNotFoundError(
            f"Model artifact not found: {args.model}. Train first, or use --baseline-only."
        )

    baseline_pred = np.asarray([baseline.predict_one(len(item["coords"])) for item in records], dtype=np.float32)

    if args.baseline_only:
        final_pred = baseline_pred
    else:
        assert artifact is not None
        seq_scaler = Standardizer.from_state(artifact["seq_scaler"])
        global_scaler = Standardizer.from_state(artifact["global_scaler"])
        dataset = TTEDataset(records, None, baseline, seq_scaler, global_scaler, is_train_record=False)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_tte,
            pin_memory=args.device.startswith("cuda"),
        )

        model = ResidualTransformer(**artifact["model_config"]).to(args.device)
        model.load_state_dict(artifact["model_state"])
        model.eval()

        residuals = []
        with torch.no_grad():
            iterator = tqdm(loader, desc="predict", leave=False) if show_progress and tqdm is not None else loader
            for batch in iterator:
                seq = batch["seq"].to(args.device)
                mask = batch["padding_mask"].to(args.device)
                global_feat = batch["global"].to(args.device)
                residuals.append(model(seq, mask, global_feat).cpu().numpy())
        final_pred = baseline_pred + np.concatenate(residuals)

    final_pred = np.maximum(1.0, final_pred)
    submission = [
        {"traj_id": int(item.get("traj_id", idx)), "travel_time": int(round(float(pred)))}
        for idx, (item, pred) in enumerate(zip(records, final_pred))
    ]
    save_pickle(submission, args.output)
    print(f"saved predictions: {args.output}", flush=True)

    if args.gt:
        gt = load_pickle(args.gt)
        y_true = [float(item["travel_time"]) for item in gt]
        print_metrics("prediction", evaluate_predictions(y_true, [x["travel_time"] for x in submission]))


if __name__ == "__main__":
    main()
