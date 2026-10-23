# train.py
import os, torch, torch.nn as nn, torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from config import Config
from dataset import get_dataloader
from model import TrajectoryRecoveryTransformer
from road_network import get_road_network

def differentiable_road_loss(pred_coords, rn, device):
    """
    计算预测点到最近路网节点的欧氏距离，可微。
    pred_coords: (N, 2) 或 (B, L, 2)
    rn: RoadNetwork 实例
    """
    if rn.kdtree is None:
        return torch.tensor(0.0, device=device)

    # 转 numpy 查最近节点索引
    if pred_coords.dim() == 3:
        B, L, _ = pred_coords.shape
        pts = pred_coords.reshape(-1, 2).detach().cpu().numpy()
    else:
        pts = pred_coords.detach().cpu().numpy()

    _, idxs = rn.kdtree.query(pts)
    nearest_xy = rn.road_positions[idxs]                     # (N,2) numpy
    nearest_xy = torch.from_numpy(nearest_xy).float().to(device)  # (N,2)

    # 与原始坐标计算距离（保留计算图）
    if pred_coords.dim() == 3:
        pts_tensor = pred_coords.reshape(-1, 2)
    else:
        pts_tensor = pred_coords

    dists = torch.norm(pts_tensor - nearest_xy, dim=1)      # (N,)
    return dists.mean()


def loss_fn(pred_res, target_res, mask_known, padding_mask,
            pred_coords=None, rn=None, weight_road=0.5, device='cuda'):
    if torch.isnan(pred_res).any() or torch.isnan(target_res).any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    mask_predict = (~mask_known) & (~padding_mask)
    mask_reconst = mask_known & (~padding_mask)

    if mask_predict.sum() == 0:
        active = ~padding_mask
        return nn.MSELoss()(pred_res[active], target_res[active]) if active.sum() > 0 else torch.tensor(0.0, device=device, requires_grad=True)

    loss_repair = nn.MSELoss()(pred_res[mask_predict], target_res[mask_predict])
    loss_recon  = nn.MSELoss()(pred_res[mask_reconst], target_res[mask_reconst])
    total = loss_repair + 0.1 * loss_recon

    if pred_coords is not None and rn is not None and rn.kdtree is not None and weight_road > 0:
        # 只对预测点计算路网损失
        road_loss = 0.0
        count = 0
        B, L, _ = pred_coords.shape
        for i in range(B):
            m_pred = mask_predict[i]   # (L,) bool
            pts = pred_coords[i, m_pred, :]   # (N,2)
            if pts.shape[0] == 0:
                continue
            road_loss += differentiable_road_loss(pts, rn, device)
            count += 1
        if count > 0:
            road_loss = road_loss / count
            total += weight_road * road_loss
    return total


def train():
    device = torch.device(Config.device)
    model = TrajectoryRecoveryTransformer(
        input_dim=Config.input_dim, d_model=Config.d_model,
        nhead=Config.nhead, num_layers=Config.num_layers,
        dim_feedforward=Config.dim_feedforward, dropout=Config.dropout
    ).to(device)

    train_loader = get_dataloader(Config.TRAIN_FILE, Config.batch_size, augment=True)
    val_loader   = get_dataloader(Config.VAL_SPARSE_INPUT, Config.batch_size, augment=False)

    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None

    rn = get_road_network(Config.ROAD_PATH, Config.REF_LON, Config.REF_LAT)

    best_val_loss = float('inf')
    no_improve = 0
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.epochs}")
        for features, target_res, mask_known, pad_mask, lengths in pbar:
            features = features.to(device)
            target_res = target_res.to(device)
            mask_known = mask_known.to(device)
            pad_mask   = pad_mask.to(device)

            if torch.isnan(features).any() or torch.isnan(target_res).any():
                continue

            optimizer.zero_grad()
            with torch.amp.autocast(device.type) if scaler else torch.no_grad():
                pred_res = model(features, src_key_padding_mask=pad_mask)
                interp_norm = features[..., 0:2]
                interp = interp_norm * Config.XY_STD
                pred_coords = interp + pred_res * Config.XY_STD
                loss = loss_fn(pred_res, target_res, mask_known, pad_mask,
                               pred_coords=pred_coords, rn=rn,
                               weight_road=Config.road_loss_weight,
                               device=device)
            if torch.isnan(loss):
                continue

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = total_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, target_res, mask_known, pad_mask, lengths in val_loader:
                features = features.to(device)
                target_res = target_res.to(device)
                mask_known = mask_known.to(device)
                pad_mask = pad_mask.to(device)
                if torch.isnan(features).any() or torch.isnan(target_res).any():
                    continue
                pred_res = model(features, src_key_padding_mask=pad_mask)
                loss = loss_fn(pred_res, target_res, mask_known, pad_mask,
                               pred_coords=None, rn=None, weight_road=0)
                if not torch.isnan(loss):
                    val_loss += loss.item()
        avg_val = val_loss / max(1, len(val_loader))

        print(f"Epoch {epoch+1}: Train {avg_train:.6f}  Val {avg_val:.6f}")
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            no_improve = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print("  -> Best model saved")
        else:
            no_improve += 1
            if no_improve >= Config.early_stop_patience:
                print("Early stopping")
                break

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    train()