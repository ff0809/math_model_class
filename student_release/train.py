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
    mask_to_predict = (~mask_known) & (~padding_mask)
    mask_to_reconstruct = mask_known & (~padding_mask)

    if mask_to_predict.sum() == 0:
        active = ~padding_mask
        return nn.MSELoss()(pred[active], target[active])
    
    loss_repair = nn.MSELoss()(pred[mask_to_predict], target[mask_to_predict])
    loss_reconstruct = nn.MSELoss()(pred[mask_to_reconstruct], target[mask_to_reconstruct])

    return loss_repair + 0.1 * loss_reconstruct

def train():
    device = torch.device(Config.device)
    print(f"Using device: {device}")

    model = TrajectoryRecoveryTransformer(
        input_dim=Config.input_dim,
        d_model=Config.d_model,
        nhead=Config.nhead,
        num_layers=Config.num_layers,
        dim_feedforward=Config.dim_feedforward,
        dropout=Config.dropout
    ).to(device)

    train_loader = get_dataloader(Config.TRAIN_FILE, Config.batch_size, augment=True)
    val_loader = get_dataloader(Config.VAL_FILE, Config.batch_size, augment=True)

    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0.0
        valid_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.epochs} [Train]")
        for features, batch_xy_norm, mask_known, padding_mask, lengths in pbar:
            features = features.to(device)
            batch_xy_norm = batch_xy_norm.to(device)
            mask_known = mask_known.to(device)
            padding_mask = padding_mask.to(device)

            pred_norm = model(features, src_key_padding_mask=padding_mask)
            
            loss = loss_fn(pred_norm, batch_xy_norm, mask_known, padding_mask)

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            valid_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_train_loss = train_loss / max(valid_batches, 1)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for features, batch_xy_norm, mask_known, padding_mask, lengths in val_loader:
                features = features.to(device)
                batch_xy_norm = batch_xy_norm.to(device)
                mask_known = mask_known.to(device)
                padding_mask = padding_mask.to(device)
                
                pred_norm = model(features, src_key_padding_mask=padding_mask)
                loss = loss_fn(pred_norm, batch_xy_norm, mask_known, padding_mask)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / max(val_batches, 1)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        scheduler.step(avg_val_loss)

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