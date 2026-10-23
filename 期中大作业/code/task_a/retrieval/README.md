# Task A Retrieval Recovery

This is a no-deep-learning baseline aimed at large gains over linear interpolation.
It uses complete training trajectories as a historical route template database.

For each missing gap between two known points:

1. Convert the start/end points to local meters.
2. Query training windows with similar start and end coordinates.
3. Average their residual shape relative to straight-line interpolation.
4. Add that residual to the input's straight-line interpolation.

Run in the `mathmodel` conda environment:

```bash
conda run -n mathmodel python task_a_retrieval/retrieval_recover.py \
  --input student_release/task_A_recovery/val_input_8.pkl \
  --gt student_release/task_A_recovery/val_gt.pkl \
  --output /tmp/retrieval_val_8.pkl \
  --limit 100 \
  --max-windows-per-gap 1000000 \
  --k 32
```

Use `--limit` for quick validation. Remove it for full prediction.

