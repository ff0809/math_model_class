import os
import torch

class Config:
    DATA_ROOT = "."
    TRAIN_FILE = os.path.join(DATA_ROOT, "data_ds15", "train.pkl")
    VAL_FILE = os.path.join(DATA_ROOT, "data_ds15", "val.pkl")
    VAL_INPUT_8 = os.path.join(DATA_ROOT, "task_A_recovery", "val_input_8.pkl")
    VAL_INPUT_16 = os.path.join(DATA_ROOT, "task_A_recovery", "val_input_16.pkl")
    VAL_GT = os.path.join(DATA_ROOT, "task_A_recovery", "val_gt.pkl")

    REF_LON = 108.946518
    REF_LAT = 34.244678

    XY_STD = 2000.0 

    d_model = 256
    nhead = 8
    num_layers = 6
    dim_feedforward = 1024
    dropout = 0.1
    input_dim = 4

    batch_size = 64
    learning_rate = 5e-4
    epochs = 100
    early_stop_patience = 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    OUTPUT_DIR = "outputs"
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")