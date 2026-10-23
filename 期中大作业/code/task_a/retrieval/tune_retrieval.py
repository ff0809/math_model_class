import argparse
import itertools

from retrieval_recover import (
    build_gap_indices,
    evaluate,
    load_pickle,
    observed_gap_lengths,
    recover_one,
)


def parse_floats(text):
    return [float(x) for x in text.split(",") if x]


def parse_ints(text):
    return [int(x) for x in text.split(",") if x]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--train", default="student_release/data_ds15/train.pkl")
    parser.add_argument("--org-stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--train-limit", type=int, default=60000)
    parser.add_argument("--max-windows-per-gap", type=int, default=3000000)
    parser.add_argument("--ks", default="16,32,48,64")
    parser.add_argument("--bandwidths", default="40,60,80,100")
    parser.add_argument("--max-errors", default="120,160,220,300")
    parser.add_argument("--scales", default="0.6,0.8,1.0,1.2")
    parser.add_argument("--rerank-pools", default="0")
    parser.add_argument("--context-weights", default="0")
    parser.add_argument(
        "--residual-mode",
        choices=("xy", "local", "local_norm"),
        default="xy",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    inputs = load_pickle(args.input)
    gt = load_pickle(args.gt)
    train = load_pickle(args.train)
    if args.limit:
        inputs = inputs[: args.limit]
        gt = gt[: args.limit]
    if args.train_limit:
        train = train[: args.train_limit]

    gaps = observed_gap_lengths(inputs)
    print(f"gap lengths: {gaps}")
    indices = build_gap_indices(
        train,
        gaps,
        max_windows_per_gap=args.max_windows_per_gap,
        use_context=False,
        org_stride=args.org_stride,
        residual_mode=args.residual_mode,
    )

    best = None
    combos = itertools.product(
        parse_ints(args.ks),
        parse_floats(args.bandwidths),
        parse_floats(args.max_errors),
        parse_floats(args.scales),
        parse_ints(args.rerank_pools),
        parse_floats(args.context_weights),
    )
    for k, bandwidth, max_error, scale, rerank_pool, context_weight in combos:
        preds = []
        for traj in inputs:
            coords = recover_one(
                traj,
                indices,
                k=k,
                bandwidth=bandwidth,
                max_endpoint_error=max_error,
                context_k=0,
                context_max_error=0,
                residual_scale=scale,
                rerank_pool=rerank_pool,
                context_weight=context_weight,
            )
            preds.append({"traj_id": traj["traj_id"], "coords": coords.tolist()})
        mae, rmse, pct = evaluate(preds, inputs, gt)
        row = (
            f"k={k} bw={bandwidth:g} maxerr={max_error:g} scale={scale:g} "
            f"pool={rerank_pool} cw={context_weight:g} "
            f"MAE={mae:.2f} RMSE={rmse:.2f} p50={pct[0]:.2f} p90={pct[2]:.2f}"
        )
        print(row)
        if best is None or mae < best[0]:
            best = (mae, rmse, pct, k, bandwidth, max_error, scale, rerank_pool, context_weight)
            print("BEST", row)

    mae, rmse, pct, k, bandwidth, max_error, scale, rerank_pool, context_weight = best
    print(
        "FINAL_BEST "
        f"k={k} bw={bandwidth:g} maxerr={max_error:g} scale={scale:g} "
        f"pool={rerank_pool} cw={context_weight:g} "
        f"MAE={mae:.2f} RMSE={rmse:.2f} p50={pct[0]:.2f} p90={pct[2]:.2f}"
    )


if __name__ == "__main__":
    main()
