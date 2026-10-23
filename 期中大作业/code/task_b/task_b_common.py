import math
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


SEQ_FEATURE_NAMES = (
    "lon",
    "lat",
    "delta_lon",
    "delta_lat",
    "step_dist_m",
    "cum_dist_ratio",
    "index_ratio",
)

GLOBAL_FEATURE_NAMES = (
    "n_points",
    "total_dist_m",
    "straight_dist_m",
    "detour_ratio",
    "start_lon",
    "start_lat",
    "end_lon",
    "end_lat",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "peak_flag",
    "stop_ratio",
)


def load_pickle(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def haversine_vec(coords: np.ndarray) -> np.ndarray:
    """Return segment distances in meters for consecutive lon/lat points."""
    if len(coords) <= 1:
        return np.zeros(0, dtype=np.float32)
    lon1 = np.radians(coords[:-1, 0].astype(np.float64))
    lat1 = np.radians(coords[:-1, 1].astype(np.float64))
    lon2 = np.radians(coords[1:, 0].astype(np.float64))
    lat2 = np.radians(coords[1:, 1].astype(np.float64))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return (2.0 * 6371000.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))).astype(np.float32)


def straight_distance(coords: np.ndarray) -> float:
    if len(coords) <= 1:
        return 0.0
    return float(haversine_vec(np.asarray([coords[0], coords[-1]], dtype=np.float32))[0])


def time_features(timestamp: int) -> Tuple[float, float, float, float, float]:
    dt = datetime.fromtimestamp(int(timestamp))
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    dow = dt.weekday()
    hour_sin = math.sin(2.0 * math.pi * hour / 24.0)
    hour_cos = math.cos(2.0 * math.pi * hour / 24.0)
    dow_sin = math.sin(2.0 * math.pi * dow / 7.0)
    dow_cos = math.cos(2.0 * math.pi * dow / 7.0)
    peak_flag = 1.0 if (7 <= dt.hour <= 9 or 17 <= dt.hour <= 19) else 0.0
    return hour_sin, hour_cos, dow_sin, dow_cos, peak_flag


def extract_features(coords: np.ndarray, departure_timestamp: int) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=np.float32)
    n = len(coords)
    if n == 0:
        raise ValueError("Empty trajectory is not supported.")

    deltas = np.zeros_like(coords, dtype=np.float32)
    if n > 1:
        deltas[1:] = coords[1:] - coords[:-1]

    seg = haversine_vec(coords)
    step_dist = np.zeros(n, dtype=np.float32)
    if n > 1:
        step_dist[1:] = seg
    total_dist = float(seg.sum())
    cum_dist = np.zeros(n, dtype=np.float32)
    if n > 1:
        cum_dist[1:] = np.cumsum(seg)
    cum_ratio = cum_dist / max(total_dist, 1.0)
    index_ratio = np.linspace(0.0, 1.0, n, dtype=np.float32) if n > 1 else np.zeros(1, dtype=np.float32)

    seq = np.column_stack(
        [
            coords[:, 0],
            coords[:, 1],
            deltas[:, 0],
            deltas[:, 1],
            step_dist,
            cum_ratio,
            index_ratio,
        ]
    ).astype(np.float32)

    straight = straight_distance(coords)
    detour = total_dist / max(straight, 1.0)
    stop_ratio = float((seg < 1.0).mean()) if len(seg) else 0.0
    global_feat = np.asarray(
        [
            float(n),
            total_dist,
            straight,
            detour,
            float(coords[0, 0]),
            float(coords[0, 1]),
            float(coords[-1, 0]),
            float(coords[-1, 1]),
            *time_features(departure_timestamp),
            stop_ratio,
        ],
        dtype=np.float32,
    )
    return seq, global_feat


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, arrays: Iterable[np.ndarray]) -> "Standardizer":
        arrays = list(arrays)
        if not arrays:
            raise ValueError("Cannot fit a standardizer on empty data.")
        mat = np.concatenate(arrays, axis=0).astype(np.float64)
        mean = mat.mean(axis=0).astype(np.float32)
        std = mat.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_state(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state(cls, state: Dict[str, np.ndarray]) -> "Standardizer":
        return cls(mean=np.asarray(state["mean"], dtype=np.float32), std=np.asarray(state["std"], dtype=np.float32))


class PointCountBaseline:
    def __init__(self, table: Dict[int, float], default_value: float):
        self.table = {int(k): float(v) for k, v in table.items()}
        self.default_value = float(default_value)
        self._known_counts = sorted(self.table)

    @classmethod
    def fit(cls, records: Sequence[dict], use_median: bool = True) -> "PointCountBaseline":
        by_count: Dict[int, List[float]] = {}
        durations = []
        for item in records:
            n = len(item["coords"])
            duration = float(int(item["timestamps"][-1]) - int(item["timestamps"][0]))
            by_count.setdefault(n, []).append(duration)
            durations.append(duration)

        table = {}
        for n, values in by_count.items():
            table[n] = float(np.median(values) if use_median else np.mean(values))
        return cls(table=table, default_value=float(np.median(durations)))

    def predict_one(self, n_points: int) -> float:
        n_points = int(n_points)
        if n_points in self.table:
            return self.table[n_points]
        if not self._known_counts:
            return self.default_value
        nearest = min(self._known_counts, key=lambda x: abs(x - n_points))
        return self.table.get(nearest, self.default_value)

    def predict_many(self, counts: Sequence[int]) -> np.ndarray:
        return np.asarray([self.predict_one(c) for c in counts], dtype=np.float32)

    def to_state(self) -> Dict[str, object]:
        return {"table": self.table, "default_value": self.default_value}

    @classmethod
    def from_state(cls, state: Dict[str, object]) -> "PointCountBaseline":
        return cls(table=state["table"], default_value=state["default_value"])


class TTEDataset(Dataset):
    def __init__(
        self,
        records: Sequence[dict],
        labels: Sequence[float] | None,
        baseline: PointCountBaseline,
        seq_scaler: Standardizer,
        global_scaler: Standardizer,
        is_train_record: bool,
    ):
        self.samples = []
        for idx, item in enumerate(records):
            coords = item["coords"]
            timestamp = int(item["timestamps"][0]) if is_train_record else int(item["departure_timestamp"])
            seq, global_feat = extract_features(coords, timestamp)
            n_points = len(coords)
            base_pred = baseline.predict_one(n_points)
            label = None if labels is None else float(labels[idx]) - base_pred
            self.samples.append(
                {
                    "traj_id": int(item.get("traj_id", idx)),
                    "seq": seq_scaler.transform(seq),
                    "global": global_scaler.transform(global_feat[None, :])[0],
                    "n_points": n_points,
                    "baseline": np.float32(base_pred),
                    "label": None if label is None else np.float32(label),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_tte(batch: Sequence[dict]) -> dict:
    lengths = torch.tensor([len(x["seq"]) for x in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    feat_dim = batch[0]["seq"].shape[1]
    seq = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    mask = torch.ones(len(batch), max_len, dtype=torch.bool)
    for i, item in enumerate(batch):
        n = len(item["seq"])
        seq[i, :n] = torch.from_numpy(item["seq"])
        mask[i, :n] = False

    out = {
        "seq": seq,
        "padding_mask": mask,
        "global": torch.tensor(np.stack([x["global"] for x in batch]), dtype=torch.float32),
        "baseline": torch.tensor([x["baseline"] for x in batch], dtype=torch.float32),
        "traj_id": torch.tensor([x["traj_id"] for x in batch], dtype=torch.long),
        "lengths": lengths,
    }
    if batch[0]["label"] is not None:
        out["label"] = torch.tensor([x["label"] for x in batch], dtype=torch.float32)
    return out


class ResidualTransformer(nn.Module):
    def __init__(
        self,
        seq_dim: int,
        global_dim: int,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 256,
    ):
        super().__init__()
        self.max_len = max_len
        self.input_proj = nn.Linear(seq_dim, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, seq: torch.Tensor, padding_mask: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = seq.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.max_len}.")
        pos = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch, seq_len)
        x = self.input_proj(seq) + self.pos_embedding(pos)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).unsqueeze(-1).float()
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        global_emb = self.global_proj(global_feat)
        return self.head(torch.cat([pooled, global_emb], dim=-1)).squeeze(-1)


def build_scalers(train_records: Sequence[dict]) -> Tuple[Standardizer, Standardizer]:
    seq_arrays = []
    global_arrays = []
    for item in train_records:
        seq, global_feat = extract_features(item["coords"], int(item["timestamps"][0]))
        seq_arrays.append(seq)
        global_arrays.append(global_feat[None, :])
    return Standardizer.fit(seq_arrays), Standardizer.fit(global_arrays)


def evaluate_predictions(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE": float(np.mean(np.abs(err) / np.maximum(y_true, 1.0)) * 100.0),
    }


def print_metrics(prefix: str, metrics: Dict[str, float]) -> None:
    print(
        f"{prefix} MAE={metrics['MAE']:.3f} "
        f"RMSE={metrics['RMSE']:.3f} MAPE={metrics['MAPE']:.3f}%"
    )
