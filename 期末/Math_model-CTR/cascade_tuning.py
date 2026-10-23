# cascade_tuning.py
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def evaluate_model(model, valid_loader, device='cuda'):
    """
    Task 2 同学提供的核心评估函数，完美兼容多重返回格式
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in valid_loader:
            x, y = batch
            x = {k: v.to(device) for k, v in x.items()}
            y = y.to(device)
            logits = model(x)
            if isinstance(logits, dict):
                logits = logits["y_pred"]
            y_pred.append(logits.sigmoid().cpu().numpy())
            y_true.append(y.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return roc_auc_score(y_true, y_pred)

def run_cascade_optimization_loop(ctr_model, valid_loader):
    """
    通过自动化网格搜索找到最完美的微调控制因子
    """
    best_auc = 0
    best_alpha = 0.5
    
    for alpha in [0.2, 0.5, 0.8]:
        print(f"\n[级联控制] 开启调参探针，测试当前控制权重 alpha = {alpha}")
        from bridge_processor import run_cascade_bridge
        run_cascade_bridge(alpha=alpha)
        
        current_auc = evaluate_model(ctr_model, valid_loader)
        print(f"=> 在 alpha={alpha} 下，下游第一名网络反馈的验证集 AUC = {current_auc:.4f}")
        
        if current_auc > best_auc:
            best_auc = current_auc
            best_alpha = alpha
            
    print(f"\n[级联调优完毕] 最佳配置系数 alpha = {best_alpha}, 本地最高 AUC 期望值 = {best_auc:.4f}")