import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from task_b_common import (
    GLOBAL_FEATURE_NAMES,
    SEQ_FEATURE_NAMES,
    PointCountBaseline,
    ResidualTransformer,
    TTEDataset,
    build_scalers,
    collate_tte,
    evaluate_predictions,
    load_pickle,
    print_metrics,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train residual Transformer for task B travel time estimation.")
    parser.add_argument("--data-root", default="student_release", help="Path to student_release directory.")
    parser.add_argument("--out", default="task_b_solution/artifacts/task_b_transformer.pt", help="Model artifact path.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=6, help="Early-stopping patience. Set 0 to disable.")
    parser.add_argument("--min-epochs", type=int, default=0, help="Do not early-stop before this epoch.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_residuals(model, loader, device):
    model.eval()
    preds = []
    baselines = []
    traj_ids = []
    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            mask = batch["padding_mask"].to(device)
            global_feat = batch["global"].to(device)
            residual = model(seq, mask, global_feat).cpu().numpy()
            preds.append(residual)
            baselines.append(batch["baseline"].numpy())
            traj_ids.append(batch["traj_id"].numpy())
    return np.concatenate(traj_ids), np.concatenate(baselines), np.concatenate(preds)


def maybe_tqdm(iterable, enabled: bool, **kwargs):
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def main():
    args = parse_args()
    set_seed(args.seed)
    data_root = Path(args.data_root)
    show_progress = not args.no_progress

    train_records = load_pickle(data_root / "data_ds15" / "train.pkl")
    val_input = load_pickle(data_root / "task_B_tte" / "val_input.pkl")
    val_gt = load_pickle(data_root / "task_B_tte" / "val_gt.pkl")
    val_labels = [float(item["travel_time"]) for item in val_gt]

    baseline = PointCountBaseline.fit(train_records, use_median=True)
    seq_scaler, global_scaler = build_scalers(train_records)

    train_labels = [float(int(item["timestamps"][-1]) - int(item["timestamps"][0])) for item in train_records]
    train_set = TTEDataset(train_records, train_labels, baseline, seq_scaler, global_scaler, is_train_record=True)
    val_set = TTEDataset(val_input, val_labels, baseline, seq_scaler, global_scaler, is_train_record=False)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_tte,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_tte,
        pin_memory=args.device.startswith("cuda"),
    )

    baseline_pred = [baseline.predict_one(len(item["coords"])) for item in val_input]
    print_metrics("baseline", evaluate_predictions(val_labels, baseline_pred))

    model = ResidualTransformer(
        seq_dim=len(SEQ_FEATURE_NAMES),
        global_dim=len(GLOBAL_FEATURE_NAMES),
        d_model=args.d_model,
        nhead=args.heads,
        num_layers=args.layers,
        dropout=args.dropout,
        max_len=256,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=20.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_mae = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        progress = maybe_tqdm(
            train_loader,
            show_progress,
            desc=f"epoch {epoch:03d}/{args.epochs}",
            leave=False,
        )
        for batch in progress:
            seq = batch["seq"].to(args.device)
            mask = batch["padding_mask"].to(args.device)
            global_feat = batch["global"].to(args.device)
            target = batch["label"].to(args.device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(seq, mask, global_feat)
            loss = loss_fn(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = len(target)
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            if show_progress and tqdm is not None:
                progress.set_postfix(loss=f"{float(loss.item()):.3f}")

        _, val_base, val_residual = predict_residuals(model, val_loader, args.device)
        val_pred = np.maximum(1.0, val_base + val_residual)
        metrics = evaluate_predictions(val_labels, val_pred)
        scheduler.step(metrics["MAE"])
        train_loss = total_loss / max(total_count, 1)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_MAE={metrics['MAE']:.3f} val_RMSE={metrics['RMSE']:.3f} val_MAPE={metrics['MAPE']:.3f}%",
            flush=True,
        )

        if metrics["MAE"] < best_mae:
            best_mae = metrics["MAE"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if args.patience > 0 and epoch >= args.min_epochs and bad_epochs >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    artifact = {
        "model_state": best_state if best_state is not None else model.state_dict(),
        "model_config": {
            "seq_dim": len(SEQ_FEATURE_NAMES),
            "global_dim": len(GLOBAL_FEATURE_NAMES),
            "d_model": args.d_model,
            "nhead": args.heads,
            "num_layers": args.layers,
            "dropout": args.dropout,
            "max_len": 256,
        },
        "baseline": baseline.to_state(),
        "seq_scaler": seq_scaler.to_state(),
        "global_scaler": global_scaler.to_state(),
        "seq_feature_names": SEQ_FEATURE_NAMES,
        "global_feature_names": GLOBAL_FEATURE_NAMES,
        "best_val_mae": best_mae,
        "best_epoch": best_epoch,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, out)
    print(f"best_epoch={best_epoch} best_val_MAE={best_mae:.3f}", flush=True)
    print(f"saved artifact: {out}", flush=True)


if __name__ == "__main__":
    main()
