# 02_TRAIN_FINAL_MODELS.py
# 功能：對每一個 IP × 每一個 DeltaT 訓練「最終版」XGBoost_AR 模型並儲存
# 說明：
#   - 不再做交叉驗證
#   - 使用所有可用事件（AD.xlsx 中 Depth 不是全 -1 的事件）
#   - 特徵 = AD+IOLag 展開後的數值 + PrevDepth（上一時間步真實深度）
#   - Y = 真實淹水深度 (cm)，不做正規化
#   - 會存兩種檔案：
#       1) 模型       : MODEL_ROOT/XGBoost_AR/IPxx/Ty_model.bin
#       2) 正規化參數 : MODEL_ROOT/XGBoost_AR/IPxx/Ty_scaler.npz

import os
import numpy as np
import pandas as pd

import RunModel as RM          # 用裡面的 data_preprocess, _build_skip_mask_from_AD, _build_prev_depth_per_sample
import data_processor as dp
from my_models import My_XGBoost

# ================================
# 基本設定
# ================================
ModelName = "XGBoost_AR"

# 全部實際存在的 IP
ALL_IP_LIST = [
    1, 2, 3, 5, 7, 8,
    10, 12, 13, 14, 15, 16,
    17, 18, 25, 27, 28
]

FutureTime = 4    # T+1 ~ T+4
use_autoreg = True

# 專案資料夾
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/'
DATA_FOLDER    = PROJECT_FOLDER + 'DATA/'
# 模型輸出根目錄（建議不要跟 OUTPUTS 混在一起）
MODEL_ROOT     = PROJECT_FOLDER + 'MODEL_WEIGHTS/'
os.makedirs(MODEL_ROOT, exist_ok=True)


# ================================
# 工具：訓練單一 IP × DeltaT 的模型
# ================================
def train_one_model(ip_id, DeltaT):
    """
    ip_id  : int，例如 1,2,3,5,7...
    DeltaT : int，1~4，代表 T+1, T+2, ...
    """

    print("\n===============================")
    print(f"開始訓練 IP{ip_id:02d} 的 T+{DeltaT} 模型")
    print("===============================")

    # -------------------------------------------------
    # 1) AD + IOLag → SD.xlsx
    #    IPnum, DeltaT 這裡要保持跟以前一樣的字串格式
    # -------------------------------------------------
    SD_FILE = RM.data_preprocess(
        DATA_FOLDER,
        str(ip_id),         # "1", "2", "3"... 讓 RunModel 去自己補 0
        str(DeltaT)
    )

    print(f"SD 檔已產生: {SD_FILE}")

    # -------------------------------------------------
    # 2) 讀 SD.xlsx → events (list of DataFrame)
    # -------------------------------------------------
    events = dp.load_data(SD_FILE)

    # 用 AD.xlsx 決定哪些事件要跳過（Depth 全為 -1）
    skip_mask = RM._build_skip_mask_from_AD(SD_FILE, events)

    filtered_events = []
    for ev, skip in zip(events, skip_mask):
        if not skip:
            filtered_events.append(ev)

    if len(filtered_events) == 0:
        print(f"⚠ IP{ip_id:02d} T+{DeltaT}：所有事件 Depth (cm) 皆為 -1，跳過模型訓練。")
        return None, None

    events = filtered_events

    # -------------------------------------------------
    # 3) 建 boundary：每場事件的累計長度（不為交叉驗證，只是給 PrevDepth 用）
    # -------------------------------------------------
    boundary = []
    total_len = 0
    for ev in events:
        total_len += len(ev)
        boundary.append(total_len)

    # -------------------------------------------------
    # 4) 產生 X_raw, Y_raw（尚未正規化）
    # -------------------------------------------------
    X_raw, Y_raw = dp.create_sequences(events)    # X_raw: (samples, features, 1)
    Y_raw = Y_raw.reshape(-1).astype(np.float32)  # (samples,)

    num_samples = X_raw.shape[0]
    # 展開特徵：把 (samples, features, 1) 攤平成 (samples, D_base)
    X_base_flat = X_raw.reshape(num_samples, -1)

    # -------------------------------------------------
    # 5) 建 PrevDepth（訓練階段：用真實深度）
    # -------------------------------------------------
    if use_autoreg:
        from RunModel import _build_prev_depth_per_sample
        prev_depth_full = _build_prev_depth_per_sample(Y_raw, boundary)  # (samples,)
        prev_col = prev_depth_full.reshape(-1, 1)
        X_full = np.concatenate([X_base_flat, prev_col], axis=1)         # (samples, D_base+1)
    else:
        X_full = X_base_flat

    # -------------------------------------------------
    # 6) 只用「全部訓練資料」做 X 的 min/max（這裡沒有 test，全部拿來訓練）
    # -------------------------------------------------
    x_min = X_full.min(axis=0)
    x_max = X_full.max(axis=0)
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0   # 防止除以 0

    X_norm = (X_full - x_min) / x_range
    Y = Y_raw

    # -------------------------------------------------
    # 7) 訓練 XGBoost 模型
    # -------------------------------------------------
    model = My_XGBoost(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=500
    )

    model = model.train(X_norm, Y)

    # -------------------------------------------------
    # 8) 儲存模型 + scaler
    # -------------------------------------------------
    model_dir = os.path.join(
        MODEL_ROOT,
        ModelName,
        f"IP{ip_id:02d}"
    )
    os.makedirs(model_dir, exist_ok=True)

    # 模型檔，例如：MODEL_WEIGHTS/XGBoost_AR/IP01/T1_model.bin
    model_path = os.path.join(
        model_dir,
        f"T{DeltaT}_model.bin"
    )

    # 正規化參數檔，例如：MODEL_WEIGHTS/XGBoost_AR/IP01/T1_scaler.npz
    scaler_path = os.path.join(
        model_dir,
        f"T{DeltaT}_scaler.npz"
    )

    model.save(model_path)
    np.savez(scaler_path, x_min=x_min, x_max=x_max)

    print(f"✅ 模型已儲存：{model_path}")
    print(f"✅ 正規化參數已儲存：{scaler_path}")

    return model_path, scaler_path


# ================================
# 主流程：訓練全部 68 個模型
# ================================
def train_all_models():
    for ip_id in ALL_IP_LIST:
        for DeltaT in range(1, FutureTime + 1):
            train_one_model(ip_id, DeltaT)


# 直接執行這支程式時，就會開始訓練全部模型
train_all_models()
