# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config import Config
from dataset import get_dataloader
from model import TrajectoryRecoveryTransformer

def loss_fn(pred, target, mask_known, padding_mask):
    """
    计算损失函数
    mask_known: True 为已知点，False 为需要修复的缺失点
    """
    # 1. 修复损失：只针对模型填补的缺失点计算 MSE
    mask_to_predict = (~mask_known) & (~padding_mask)

    # 2. 重构损失：针对已知点计算 MSE，确保模型不会破坏已知点
    mask_to_reconstruct = mask_known & (~padding_mask)

    if mask_to_predict.sum() == 0:
        # 如果没有缺失点，强制对所有非 padding 点计算，防止 Loss 为 0
        active = ~padding_mask
        return nn.MSELoss()(pred[active], target[active])

    loss_repair = nn.MSELoss()(pred[mask_to_predict], target[mask_to_predict])
    loss_reconstruct = nn.MSELoss()(pred[mask_to_reconstruct], target[mask_to_reconstruct])

    # 总损失 = 修复损失 + 0.1 * 重构损失（辅助模型理解整体轨迹结构）
    return loss_repair + 0.1 * loss_reconstruct

def train():
    device = torch.device(Config.device)
    print(f"Using device: {device}")

    # 初始化模型
    model = TrajectoryRecoveryTransformer(
        input_dim=Config.input_dim,
        d_model=Config.d_model,
        nhead=Config.nhead,
        num_layers=Config.num_layers,
        dim_feedforward=Config.dim_feedforward,
        dropout=Config.dropout
    ).to(device)

    # 训练和训练期验证都在完整轨迹上随机挖洞；任务 A 的真实 1/8、1/16 分数由 eval.py 计算。
    # 注意：num_workers 设为 0 可以避免每个子进程重复加载路网；
    # 若路网特征已预计算并存入 pkl，可将 num_workers 调大以加速读取。
    train_loader = get_dataloader(Config.TRAIN_FILE, Config.batch_size,
                                  shuffle=True, augment=True, num_workers=0)
    val_loader = get_dataloader(Config.VAL_FILE, Config.batch_size,
                                shuffle=False, augment=True, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    epochs = int(os.environ.get("TASK_A_EPOCHS", Config.epochs))
    max_train_batches = int(os.environ.get("TASK_A_MAX_TRAIN_BATCHES", "0"))
    max_val_batches = int(os.environ.get("TASK_A_MAX_VAL_BATCHES", "0"))

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        valid_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, (features, batch_xy_norm, batch_interp_xy_norm, mask_known, padding_mask, lengths) in enumerate(pbar):
            features = features.to(device)
            batch_xy_norm = batch_xy_norm.to(device)
            batch_interp_xy_norm = batch_interp_xy_norm.to(device)
            mask_known = mask_known.to(device)
            padding_mask = padding_mask.to(device)

            # 模型只学习“相对线性插值的残差”，降低路网噪声把绝对坐标带偏的风险。
            pred_delta_norm = model(features, src_key_padding_mask=padding_mask)
            pred_norm = batch_interp_xy_norm + pred_delta_norm

            # 计算损失
            loss = loss_fn(pred_norm, batch_xy_norm, mask_known, padding_mask)

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪防止超长轨迹导致梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            valid_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")
            if max_train_batches and batch_idx + 1 >= max_train_batches:
                break

        avg_train_loss = train_loss / max(valid_batches, 1)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_idx, (features, batch_xy_norm, batch_interp_xy_norm, mask_known, padding_mask, lengths) in enumerate(val_loader):
                features = features.to(device)
                batch_xy_norm = batch_xy_norm.to(device)
                batch_interp_xy_norm = batch_interp_xy_norm.to(device)
                mask_known = mask_known.to(device)
                padding_mask = padding_mask.to(device)

                pred_delta_norm = model(features, src_key_padding_mask=padding_mask)
                pred_norm = batch_interp_xy_norm + pred_delta_norm
                loss = loss_fn(pred_norm, batch_xy_norm, mask_known, padding_mask)

                val_loss += loss.item()
                val_batches += 1
                if max_val_batches and batch_idx + 1 >= max_val_batches:
                    break

        avg_val_loss = val_loss / max(val_batches, 1)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

        scheduler.step(avg_val_loss)

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  --> Best Model Saved (Loss: {best_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= Config.early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print("Training Complete.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    train()
