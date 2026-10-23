import argparse
import importlib
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree


REF_LON = 108.946518
REF_LAT = 34.244678
DEFAULT_TRAIN = "student_release/data_ds15/train.pkl"


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


def save_pickle(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f)


def lonlat_to_xy(coords):
    arr = np.asarray(coords, dtype=np.float64)
    x, y = TO_XY.transform(arr[:, 0], arr[:, 1])
    return np.column_stack([x, y]).astype(np.float32)


def xy_to_lonlat(xy):
    lon, lat = TO_LONLAT.transform(xy[:, 0], xy[:, 1])
    return np.column_stack([lon, lat])


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


def observed_gap_lengths(inputs):
    gaps = set()
    for traj in inputs:
        mask = np.asarray(traj["mask"], dtype=bool)
        known = np.where(mask)[0]
        for left, right in zip(known[:-1], known[1:]):
            if right - left > 1:
                gaps.add(int(right - left))
    return sorted(gaps)


class GapIndex:
    def __init__(
        self,
        d,
        features,
        residuals,
        residual_mode="xy",
        neighbor_context=None,
        context_features=None,
        context_ids=None,
    ):
        self.d = int(d)
        self.features = features.astype(np.float32, copy=False)
        self.residuals = residuals.astype(np.float32, copy=False)
        self.residual_mode = residual_mode
        self.neighbor_context = (
            neighbor_context.astype(np.float32, copy=False)
            if neighbor_context is not None
            else None
        )
        self.tree = cKDTree(self.features)
        self.context_features = None
        self.context_ids = None
        self.context_tree = None
        if context_features is not None and len(context_features):
            self.context_features = context_features.astype(np.float32, copy=False)
            self.context_ids = context_ids.astype(np.int64, copy=False)
            self.context_tree = cKDTree(self.context_features)

    def query_residual(
        self,
        feature,
        k,
        bandwidth,
        max_endpoint_error,
        context_feature=None,
        context_k=None,
        context_max_error=None,
        neighbor_feature=None,
        rerank_pool=None,
        context_weight=0.0,
    ):
        if (
            neighbor_feature is not None
            and self.neighbor_context is not None
            and context_weight > 0
        ):
            got = self._query_rerank(
                feature.astype(np.float32),
                neighbor_feature.astype(np.float32),
                k=k,
                pool=rerank_pool or max(k * 16, 128),
                bandwidth=bandwidth,
                max_error=max_endpoint_error,
                context_weight=context_weight,
            )
            if got is not None:
                return got
        if (
            context_feature is not None
            and self.context_tree is not None
            and (context_k or 0) > 0
            and (context_max_error or 0) > 0
        ):
            got = self._query_tree(
                self.context_tree,
                context_feature.astype(np.float32),
                k=min(context_k or k, len(self.context_features)),
                bandwidth=bandwidth,
                max_error=context_max_error or max_endpoint_error * 1.8,
                ids=self.context_ids,
            )
            if got is not None:
                return got
        return self._query_tree(
            self.tree,
            feature.astype(np.float32),
            k=k,
            bandwidth=bandwidth,
            max_error=max_endpoint_error,
            ids=None,
        )

    def _query_rerank(
        self, feature, neighbor_feature, k, pool, bandwidth, max_error, context_weight
    ):
        pool = min(pool, len(self.features))
        dists, idxs = self.tree.query(feature, k=pool, workers=-1)
        dists = np.atleast_1d(dists).astype(np.float64)
        idxs = np.atleast_1d(idxs)
        valid = dists <= max_error
        ctx = self.neighbor_context[idxs]
        has_ctx = np.isfinite(ctx).all(axis=1)
        valid &= has_ctx
        if not np.any(valid):
            return None
        dists = dists[valid]
        idxs = idxs[valid]
        ctx = ctx[valid]
        ctx_dist = np.linalg.norm(ctx - neighbor_feature.reshape(1, 4), axis=1)
        score = dists + context_weight * ctx_dist
        order = np.argsort(score)[: min(k, len(score))]
        score = score[order]
        idxs = idxs[order]
        weights = np.exp(-0.5 * (score / max(bandwidth, 1e-6)) ** 2)
        weights /= max(weights.sum(), 1e-12)
        avg = np.tensordot(weights.astype(np.float32), self.residuals[idxs], axes=(0, 0))
        return self._to_global_residual(avg, feature)

    def _query_tree(self, tree, feature, k, bandwidth, max_error, ids=None):
        k = min(k, len(self.features))
        dists, idxs = tree.query(feature, k=k, workers=-1)
        dists = np.atleast_1d(dists).astype(np.float64)
        idxs = np.atleast_1d(idxs)
        valid = dists <= max_error
        if not np.any(valid):
            return None
        dists = dists[valid]
        idxs = idxs[valid]
        if ids is not None:
            idxs = ids[idxs]
        weights = np.exp(-0.5 * (dists / max(bandwidth, 1e-6)) ** 2)
        weights /= max(weights.sum(), 1e-12)
        res = self.residuals[idxs]
        avg = np.tensordot(weights.astype(np.float32), res, axes=(0, 0))
        return self._to_global_residual(avg, feature)

    def _to_global_residual(self, avg, feature):
        if self.residual_mode == "xy":
            return avg

        start = feature[:2].astype(np.float32)
        end = feature[2:4].astype(np.float32)
        vec = end - start
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            return np.zeros_like(avg)
        unit = vec / length
        perp = np.array([-unit[1], unit[0]], dtype=np.float32)
        if self.residual_mode == "local_norm":
            avg = avg * length
        return avg[:, :1] * unit.reshape(1, 2) + avg[:, 1:2] * perp.reshape(1, 2)


def train_xy_sequences(train_data, org_stride=1):
    sequences = []
    for traj in train_data:
        xy = lonlat_to_xy(traj["coords"])
        if org_stride <= 1:
            sequences.append(xy)
            continue
        for phase in range(org_stride):
            seq = xy[phase::org_stride]
            if len(seq) >= 3:
                sequences.append(seq)
    return sequences


def build_gap_indices(
    train_data,
    gap_lengths,
    max_windows_per_gap=0,
    seed=2026,
    use_context=False,
    org_stride=1,
    residual_mode="xy",
):
    rng = np.random.default_rng(seed)
    xy_train = train_xy_sequences(train_data, org_stride=org_stride)

    indices = {}
    for d in gap_lengths:
        feats = []
        ress = []
        neighs = []
        ctx_feats = []
        ctx_ids = []
        base_offset = 0
        for xy in xy_train:
            n = len(xy)
            if n <= d:
                continue
            starts = xy[:-d]
            ends = xy[d:]
            m = len(starts)
            feat = np.column_stack([starts, ends]).astype(np.float32)
            alpha = (np.arange(1, d, dtype=np.float32) / d).reshape(1, -1, 1)
            line = starts[:, None, :] + alpha * (ends - starts)[:, None, :]
            residual = xy[np.arange(1, d)[:, None] + np.arange(m)[None, :]]
            residual = np.moveaxis(residual, 0, 1) - line
            if residual_mode in ("local", "local_norm"):
                vec = ends - starts
                length = np.linalg.norm(vec, axis=1).astype(np.float32)
                unit = vec / np.maximum(length[:, None], 1e-6)
                perp = np.column_stack([-unit[:, 1], unit[:, 0]])
                along = np.sum(residual * unit[:, None, :], axis=2)
                across = np.sum(residual * perp[:, None, :], axis=2)
                residual = np.stack([along, across], axis=2)
                if residual_mode == "local_norm":
                    residual = residual / np.maximum(length[:, None, None], 1e-6)
            feats.append(feat)
            ress.append(residual.astype(np.float32))
            neighbor = np.full((m, 4), np.nan, dtype=np.float32)
            js = np.arange(m)
            valid_neighbor = (js - d >= 0) & (js + 2 * d < n)
            if np.any(valid_neighbor):
                jv = js[valid_neighbor]
                neighbor[valid_neighbor, 0:2] = xy[jv - d]
                neighbor[valid_neighbor, 2:4] = xy[jv + 2 * d]
            neighs.append(neighbor)
            ctx_start = d
            ctx_stop = n - 2 * d
            if use_context and ctx_stop > ctx_start:
                mids = np.arange(ctx_start, ctx_stop)
                prevs = xy[mids - d]
                starts_ctx = xy[mids]
                ends_ctx = xy[mids + d]
                nexts = xy[mids + 2 * d]
                ctx = np.column_stack([prevs, starts_ctx, ends_ctx, nexts])
                ctx_feats.append(ctx.astype(np.float32))
                ctx_ids.append((base_offset + mids).astype(np.int64))
            base_offset += m
        if not feats:
            continue
        features = np.vstack(feats)
        residuals = np.vstack(ress)
        neighbor_context = np.vstack(neighs)
        if max_windows_per_gap and len(features) > max_windows_per_gap:
            keep = rng.choice(len(features), size=max_windows_per_gap, replace=False)
            features = features[keep]
            residuals = residuals[keep]
            neighbor_context = neighbor_context[keep]
            id_map = np.full(int(keep.max()) + 1, -1, dtype=np.int64)
            id_map[keep] = np.arange(len(keep), dtype=np.int64)
            if use_context and ctx_feats:
                raw_ctx_features = np.vstack(ctx_feats)
                raw_ctx_ids = np.concatenate(ctx_ids)
                good = raw_ctx_ids < len(id_map)
                raw_ctx_features = raw_ctx_features[good]
                raw_ctx_ids = raw_ctx_ids[good]
                mapped = id_map[raw_ctx_ids]
                good = mapped >= 0
                context_features = raw_ctx_features[good]
                context_ids = mapped[good]
            else:
                context_features = None
                context_ids = None
        else:
            context_features = np.vstack(ctx_feats) if use_context and ctx_feats else None
            context_ids = np.concatenate(ctx_ids) if use_context and ctx_ids else None
        print(f"gap={d}: windows={len(features)}")
        if use_context and context_features is not None:
            print(f"gap={d}: context_windows={len(context_features)}")
        indices[d] = GapIndex(
            d,
            features,
            residuals,
            residual_mode,
            neighbor_context,
            context_features,
            context_ids,
        )
    return indices


def linear_xy(coords, mask):
    xy = lonlat_to_xy(coords)
    known = np.where(mask)[0]
    full = np.zeros_like(xy)
    t = np.arange(len(xy))
    full[:, 0] = np.interp(t, known, xy[known, 0])
    full[:, 1] = np.interp(t, known, xy[known, 1])
    return full, xy


def recover_one(
    traj,
    gap_indices,
    k,
    bandwidth,
    max_endpoint_error,
    context_k,
    context_max_error,
    residual_scale,
    rerank_pool,
    context_weight,
):
    coords = np.asarray(traj["coords"], dtype=np.float64)
    mask = np.asarray(traj["mask"], dtype=bool)
    pred_xy, known_xy = linear_xy(coords, mask)
    known = np.where(mask)[0]

    for left, right in zip(known[:-1], known[1:]):
        d = int(right - left)
        if d <= 1 or d not in gap_indices:
            continue
        feature = np.r_[known_xy[left], known_xy[right]].astype(np.float32)
        context_feature = None
        neighbor_feature = None
        prev_known = left - d
        next_known = right + d
        if prev_known >= 0 and next_known < len(known_xy) and mask[prev_known] and mask[next_known]:
            neighbor_feature = np.r_[
                known_xy[prev_known],
                known_xy[next_known],
            ].astype(np.float32)
            context_feature = np.r_[
                known_xy[prev_known],
                known_xy[left],
                known_xy[right],
                known_xy[next_known],
            ].astype(np.float32)
        residual = gap_indices[d].query_residual(
            feature,
            k=k,
            bandwidth=bandwidth,
            max_endpoint_error=max_endpoint_error,
            context_feature=context_feature,
            context_k=context_k,
            context_max_error=context_max_error,
            neighbor_feature=neighbor_feature,
            rerank_pool=rerank_pool,
            context_weight=context_weight,
        )
        if residual is None:
            continue
        pred_xy[left + 1 : right] += residual_scale * residual

    out = xy_to_lonlat(pred_xy)
    out[mask] = coords[mask]
    return out


def evaluate(preds, inputs, gt):
    dists = []
    for pred, inp, truth in zip(preds, inputs, gt):
        pred_coords = np.asarray(pred["coords"], dtype=np.float64)
        gt_coords = np.asarray(truth["coords"], dtype=np.float64)
        miss = ~np.asarray(inp["mask"], dtype=bool)
        d = haversine(
            pred_coords[miss, 0],
            pred_coords[miss, 1],
            gt_coords[miss, 0],
            gt_coords[miss, 1],
        )
        if len(d):
            dists.append(d)
    if not dists:
        return 0.0, 0.0, []
    dists = np.concatenate(dists)
    return (
        float(np.mean(dists)),
        float(np.sqrt(np.mean(dists**2))),
        list(np.percentile(dists, [50, 75, 90, 95, 99])),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gt")
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--org-stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--max-windows-per-gap", type=int, default=1500000)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--context-k", type=int, default=64)
    parser.add_argument("--bandwidth", type=float, default=80.0)
    parser.add_argument("--max-endpoint-error", type=float, default=160.0)
    parser.add_argument("--context-max-error", type=float, default=500.0)
    parser.add_argument("--use-context", action="store_true")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument(
        "--residual-mode",
        choices=("xy", "local", "local_norm"),
        default="xy",
    )
    parser.add_argument("--rerank-pool", type=int, default=0)
    parser.add_argument("--context-weight", type=float, default=0.15)
    parser.add_argument("--cache", help="Optional pickle path for cached gap indices.")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    inputs = load_pickle(args.input)
    if args.limit:
        inputs = inputs[: args.limit]
    train = load_pickle(args.train)
    if args.train_limit:
        train = train[: args.train_limit]

    gaps = observed_gap_lengths(inputs)
    print(f"gap lengths: {gaps}")
    indices = None
    if args.cache and Path(args.cache).exists() and not args.rebuild_cache:
        print(f"Loading cached indices: {args.cache}")
        indices = load_pickle(args.cache)
        missing = [d for d in gaps if d not in indices]
        if missing:
            print(f"Cache missing gaps {missing}; rebuilding.")
            indices = None
    if indices is None:
        indices = build_gap_indices(
            train,
            gaps,
            max_windows_per_gap=args.max_windows_per_gap,
            use_context=args.use_context,
            org_stride=args.org_stride,
            residual_mode=args.residual_mode,
        )
        if args.cache:
            print(f"Saving cached indices: {args.cache}")
            save_pickle(args.cache, indices)

    results = []
    for i, traj in enumerate(inputs, 1):
        coords = recover_one(
            traj,
            indices,
            k=args.k,
            bandwidth=args.bandwidth,
            max_endpoint_error=args.max_endpoint_error,
            context_k=args.context_k,
            context_max_error=args.context_max_error,
            residual_scale=args.residual_scale,
            rerank_pool=args.rerank_pool,
            context_weight=args.context_weight,
        )
        results.append({"traj_id": traj["traj_id"], "coords": coords.tolist()})
        if i % 1000 == 0:
            print(f"Recovered {i}/{len(inputs)}")
    save_pickle(args.output, results)
    print(f"Saved {args.output}")

    if args.gt:
        gt = load_pickle(args.gt)
        if args.limit:
            gt = gt[: args.limit]
        mae, rmse, pct = evaluate(results, inputs, gt)
        print(f"MAE={mae:.2f}m RMSE={rmse:.2f}m percentiles={pct}")


if __name__ == "__main__":
    main()
