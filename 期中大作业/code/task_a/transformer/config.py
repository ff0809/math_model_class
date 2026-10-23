# config.py
from pathlib import Path

try:
    import torch
except Exception:
    torch = None

class Config:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    TASK_A_ROOT = PROJECT_ROOT / "task_a"
    STUDENT_ROOT = PROJECT_ROOT / "student_release"
    DATA_ROOT = STUDENT_ROOT / "data_ds15"
    TASK_A_RELEASE_ROOT = STUDENT_ROOT / "task_A_recovery"

    TRAIN_FILE = DATA_ROOT / "train.pkl"

    # 验证集完整轨迹只用于构造训练过程中的验证 mask；最终分数以 eval.py 为准。
    VAL_FILE = DATA_ROOT / "val.pkl"

    VAL_GT = TASK_A_RELEASE_ROOT / "val_gt.pkl"
    RAW_VAL_INPUT_8 = TASK_A_RELEASE_ROOT / "val_input_8.pkl"
    RAW_VAL_INPUT_16 = TASK_A_RELEASE_ROOT / "val_input_16.pkl"
    VAL_INPUT_8 = TASK_A_RELEASE_ROOT / "val_input_8_road.pkl"
    VAL_INPUT_16 = TASK_A_RELEASE_ROOT / "val_input_16_road.pkl"

    ROAD_PATH = TASK_A_ROOT / "shaanxi-260507-free.shp" / "gis_osm_roads_free_1.shp"

    REF_LON = 108.946518
    REF_LAT = 34.244678
    XY_STD = 2000.0
    ROAD_DENSIFY_STEP_M = 15.0
    APPLY_VELOCITY_CONSTRAINT = False

    input_dim = 6

    d_model = 128
    nhead = 4
    num_layers = 4
    dim_feedforward = 512
    dropout = 0.1

    batch_size = 256
    learning_rate = 5e-4
    epochs = 100
    early_stop_patience = 15

    device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    OUTPUT_DIR = TASK_A_ROOT / "outputs"
    MODEL_SAVE_PATH = OUTPUT_DIR / "best_model_road_residual.pth"
    VAL_PRED_8 = OUTPUT_DIR / "val_pred_8.pkl"
    VAL_PRED_16 = OUTPUT_DIR / "val_pred_16.pkl"
