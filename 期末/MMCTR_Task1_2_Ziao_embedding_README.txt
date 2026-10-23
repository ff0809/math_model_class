MMCTR Task 1/2 Ziao embedding package

Core file:
  item_info_ziao.parquet

This file is ZiaoGao/MMCTR precomputed item embedding data, renamed from
item_info_updated.parquet. Its schema matches the official item_info.parquet:
item_id, item_tags, item_emb_d128.

Usage:
  Keep the official train/valid/test split and CTR model unchanged.
  Replace the config item_info path from:
    data/MicroLens_1M_x1/item_info.parquet
  to:
    data/MicroLens_1M_x1/item_info_ziao.parquet

The included README.md and Jina_text_img_v8.py are for source/method reference
only. We did not rerun Jina/PCA/embedding extraction for this experiment.
