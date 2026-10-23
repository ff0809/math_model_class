# prediction.py
import os
import sys
import logging
import argparse
import shutil
import pandas as pd
import numpy as np

np.Inf = np.inf

os.chdir(os.path.dirname(os.path.realpath(__file__)))

from fuxictr.utils import load_config, set_logger, print_to_json
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess import FeatureProcessor, build_dataset

import src as model_zoo
from src.mmctr_dataloader import MMCTRDataLoader
from bridge_processor import run_cascade_bridge

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/Transformer_DCN_microlens_mmctr_config/')
    parser.add_argument('--expid', type=str, default='Cascade_Multimodal_DCN_v1')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.5, help="级联控制强度")
    args = vars(parser.parse_args())

    run_cascade_bridge(task1_emb_path="./data/item_info_ziao.parquet", alpha=args['alpha'])
    
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    set_logger(params)
    seed_everything(seed=params['seed'])

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")

    feature_encoder = FeatureProcessor(**params)
    params["train_data"], params["valid_data"], params["test_data"] = build_dataset(feature_encoder, **params)
        
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)

    model_class = getattr(model_zoo, params['model'])
    model = model_class(feature_map, **params)

    params["data_loader"] = MMCTRDataLoader
    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()
    
    logging.info("🌟 正在拉起网络进行全量端到端拟合训练...")
    model.fit(train_gen, epochs=params['epochs'], validation_data=valid_gen)
    
    model.load_weights(model.checkpoint)

    logging.info('正在生成测试集预测概率 pCTR...')
    test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
    test_pred = model.predict(test_gen)
    ans = pd.DataFrame({"ID": range(test_pred.shape[0]), "Task1": test_pred})
    
    os.makedirs("submission", exist_ok=True)
    ans.to_csv("submission/prediction.csv", index=False)
    shutil.make_archive(f'submission/{experiment_id}', 'zip', 'submission/', 'prediction.csv')
    logging.info(f"大功告成！全自主成果提交流水线已打包至: submission/{experiment_id}.zip")