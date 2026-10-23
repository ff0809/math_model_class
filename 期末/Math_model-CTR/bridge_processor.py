# bridge_processor.py
import os
import pandas as pd
import numpy as np

def calculate_semantic_similarity(click_df, item_emb_dict):
    """
    数学建模原创成果：显式计算用户历史序列与目标物品的多模态语义空间余弦相似度
    """
    print("[Innovation] 正在基于 Task 1 向量计算用户-物品多模态语义契合度...")
    sim_scores = []
    
    for idx, row in click_df.iterrows():
        target_id = row['item_id']
        raw_seq = row.get('item_seq', [])
        
        # 兼容处理历史行为序列格式 (可能是字符串、列表或数组)
        if isinstance(raw_seq, str):
            history_seq = [int(x) for x in raw_seq.split('^') if x.isdigit()]
        elif isinstance(raw_seq, (list, np.ndarray)):
            history_seq = [int(x) for x in raw_seq]
        else:
            history_seq = []
            
        target_vec = item_emb_dict.get(int(target_id), np.zeros(128))
        history_vecs = [item_emb_dict.get(int(hid), np.zeros(128)) for hid in history_seq]
        
        if len(history_vecs) > 0 and np.linalg.norm(target_vec) > 0:
            user_mean_vec = np.mean(history_vecs, axis=0)
            if np.linalg.norm(user_mean_vec) > 0:
                # 余弦相似度计算公式
                sim = np.dot(user_mean_vec, target_vec) / (np.linalg.norm(user_mean_vec) * np.linalg.norm(target_vec))
                sim_scores.append(float(sim))
                continue
        sim_scores.append(0.0) 
        
    return sim_scores

def run_cascade_bridge(task1_emb_path="./data/item_info_ziao.parquet", 
                       official_item_info_path="./data/MicroLens_1M_x1/item_info.parquet", 
                       alpha=0.5):
    """
    级联优化总线：安全更新官方 item_info.parquet (保持行序一致性)，并为 train/valid/test 注入创新特征
    """
    print(f"\n[Bridge] 启动级联优化总线。当前特征缩放因子 alpha = {alpha}")
    
    if not os.path.exists(task1_emb_path):
        raise FileNotFoundError(f"未找到 Task 1 特征文件: {task1_emb_path}")
    if not os.path.exists(official_item_info_path):
        raise FileNotFoundError(f"未找到官方基础属性表: {official_item_info_path}")
        
    # 1. 读取 Task 1 同学的特征表
    task1_df = pd.read_parquet(task1_emb_path)
    
    # 自动识别向量所在的列名
    emb_col = next((col for col in ['embedding', 'item_emb_d128', 'features'] if col in task1_df.columns), None)
    if emb_col is None:
        emb_col = [c for c in task1_df.columns if c != 'item_id'][0]
        
    print(f"[Bridge] 成功识别 Task 1 向量列名: '{emb_col}'")
    
    # 2. 特征白化去噪与 alpha 信号增强
    raw_embs = np.vstack(task1_df[emb_col].values)
    centered_embs = (raw_embs - np.mean(raw_embs, axis=0)) * alpha
    item_emb_dict = dict(zip(task1_df['item_id'].astype(int), centered_embs))
    
    # 3. 严格安全地更新官方的 item_info.parquet（采用 map 映射，绝不打乱原有的 .iloc 顺序）
    official_item_df = pd.read_parquet(official_item_info_path)
    
    # 将计算好的 128维 向量映射回官方表的对应 item_id 中
    official_item_df['item_emb_d128'] = official_item_df['item_id'].map(item_emb_dict)
    
    # 对缺失或未匹配到的项目用全 0 向量做数学兜底
    official_item_df['item_emb_d128'] = official_item_df['item_emb_d128'].apply(
        lambda x: x if isinstance(x, (list, np.ndarray)) else [0.0] * 128
    )
    
    official_item_df.to_parquet(official_item_info_path)
    print(f"[Bridge] 级联多模态向量已安全织入官方大本营: {official_item_info_path}")
    
    # 4. 动态为点击流表格更新原创特征项
    for stage in ['train', 'valid', 'test']:
        stage_path = f"./data/MicroLens_1M_x1/{stage}.parquet"
        if os.path.exists(stage_path):
            df = pd.read_parquet(stage_path)
            df['semantic_sim_score'] = calculate_semantic_similarity(df, item_emb_dict)
            df.to_parquet(stage_path)
            print(f"[Innovation] 成功为下游 {stage}.parquet 注入原创特征 'semantic_sim_score'")

if __name__ == "__main__":
    run_cascade_bridge()