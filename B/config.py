import os
import torch

class Config:
    DATA_ROOT = "."
    TRAIN_FILE = os.path.join(DATA_ROOT, "data_ds15", "train.pkl")
    VAL_SPARSE_INPUT = "task_A_recovery/val_input_8.pkl"   # 验证也用稀疏输入
    VAL_INPUT_8 = "task_A_recovery/val_input_8.pkl"
    VAL_INPUT_16 = "task_A_recovery/val_input_16.pkl"

    ROAD_PATH = "shaanxi-260507-free.shp/gis_osm_roads_free_1.shp"

    REF_LON = 108.946518
    REF_LAT = 34.244678
    XY_STD = 2000.0

    # 路网偏移专用的缩放系数，使其量级与插值坐标相近
    ROAD_STD = 50.0            # 假设路网偏移通常 0~100 米，除以 50 → 0~2

    input_dim = 6              # 重新回到 6 维

    # 缩小模型加速
    d_model = 128
    nhead = 4
    num_layers = 4
    dim_feedforward = 512
    dropout = 0.1

    batch_size = 128
    learning_rate = 5e-4
    epochs = 100
    early_stop_patience = 10

    # 路网约束损失的权重
    road_loss_weight = 0.5     # 可根据实际效果调节

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR = "outputs"
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model_road.pth")