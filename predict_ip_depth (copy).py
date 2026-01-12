# predict_ip_depth.py
# 原本的 02_predict_ip_depth.py 重構版本
# 功能：
# 1. 從 DATA/AD_realtime.xlsx 讀最新一列雨量特徵
# 2. 針對 ALL_IP_LIST 裡的每個 IP，以及 T1~T4
#    讀取對應的：
#      MODEL_WEIGHTS/XGBoost_AR/IPxx/Ty_model.bin
#      MODEL_WEIGHTS/XGBoost_AR/IPxx/Ty_scaler.npz  (內含 x_min, x_max)
# 3. 做 min-max 正規化 → XGBoost 預測淹水深度 (cm)，負值強制設成 0
# 4. 輸出 OUTPUTS/realtime_ip_depth.xlsx
#    欄位：IP_ID, Depth_T1, Depth_T2, Depth_T3, Depth_T4
#
# 注意：
# - 現在不會在 import 時自動執行
# - 由 run_all_predictions() 負責跑一輪預測流程

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.core import XGBoostError
import joblib

# ==============================
# 基本設定（跟 02_TRAIN_FINAL_MODELS 對齊）
# ==============================
ModelName = "XGBoost_AR"

ALL_IP_LIST = [
    1, 2, 3, 5, 7, 8,
    10, 12, 13, 14, 15, 16,
    17, 18, 25, 27, 28
]

FutureTime = 4  # T1~T4 -> T+1~T+4

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER   = os.path.join(BASE_DIR, "DATA")
MODEL_ROOT    = os.path.join(BASE_DIR, "MODEL_WEIGHTS", ModelName)
OUTPUT_FOLDER = os.path.join(BASE_DIR, "OUTPUTS")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

AD_REALTIME_PATH = os.path.join(DATA_FOLDER, "AD_realtime.xlsx")
IP_DEPTH_OUTPUT  = os.path.join(OUTPUT_FOLDER, "realtime_ip_depth.xlsx")


# ==============================
# 1. 載入模型 & scaler 的工具
# ==============================
def load_xgb_model_flexible(model_path: str):
    """
    優先用 XGBRegressor.load_model()
    不行就改用 joblib.load()
    """
    try:
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        print(f"   ✔ 以 XGBoost 原生 load_model() 載入：{model_path}")
        return model
    except Exception as e:  # ★★★ 關鍵：改成抓所有 Exception
        print(f"   ⚠ load_model() 失敗，改試 joblib：{e}")

    try:
        model = joblib.load(model_path)
        print(f"   ✔ 以 joblib.load() 載入：{model_path}")
        return model
    except Exception as e2:
        raise RuntimeError(
            f"❌ 無法載入模型檔：{model_path}\n"
            f"   load_model() 與 joblib.load() 都失敗。\n"
            f"   joblib 錯誤：{e2}"
        )



def load_minmax_params(scaler_path: str, expected_n_features: int):
    """
    從 npz 檔讀取 x_min, x_max，並檢查維度
    （這要跟 02_TRAIN_FINAL_MODELS.py 裡 save 的內容一致）
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"❌ 找不到 scaler 檔：{scaler_path}")

    scaler_npz = np.load(scaler_path)
    keys = list(scaler_npz.files)
    print(f"   ℹ 讀取 scaler 檔 {os.path.basename(scaler_path)}，內含鍵：{keys}")

    if "x_min" not in keys or "x_max" not in keys:
        raise KeyError(
            f"❌ scaler 檔必須包含 'x_min' 與 'x_max'，目前只有：{keys}"
        )

    x_min = scaler_npz["x_min"]
    x_max = scaler_npz["x_max"]

    if x_min.shape[0] != expected_n_features:
        raise ValueError(
            f"❌ scaler 特徵數 {x_min.shape[0]} 與即時輸入特徵數 {expected_n_features} 不一致。\n"
            f"   → 請確認 AD_realtime.xlsx 的欄位順序與訓練時 SD.xlsx 完全相同。"
        )

    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0

    return x_min, x_range


def load_model_and_scaler(ip_id: int, t_id: int, expected_n_features: int):
    """
    ip_id : 真實 IP 編號（1,2,3,5,...,28）
    t_id  : 1~4 對應 T1~T4
    """
    ip_folder   = os.path.join(MODEL_ROOT, f"IP{ip_id:02d}")
    model_path  = os.path.join(ip_folder, f"T{t_id}_model.bin")
    scaler_path = os.path.join(ip_folder, f"T{t_id}_scaler.npz")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 模型不存在：{model_path}")

    model = load_xgb_model_flexible(model_path)
    x_min, x_range = load_minmax_params(scaler_path, expected_n_features=expected_n_features)

    return model, x_min, x_range


# ==============================
# 2. 封裝成「跑一輪預測」的函式
# ==============================
def run_all_predictions():
    """
    讀 AD_realtime.xlsx 最新一列 → 建立 X_raw (含 PrevDepth) →
    17 測站 × T1~T4 預測 → 輸出 realtime_ip_depth.xlsx
    """
    # 2-1 讀 AD_realtime.xlsx
    if not os.path.exists(AD_REALTIME_PATH):
        raise FileNotFoundError("❌ 找不到 AD_realtime.xlsx，請先執行即時雨量抓取程式。")

    df_ad = pd.read_excel(AD_REALTIME_PATH)

    if df_ad.shape[0] == 0:
        raise ValueError("❌ AD_realtime.xlsx 沒有任何列，至少要有一列資料。")

    # 取最後一列（最新時間）的雨量特徵
    last_row = df_ad.iloc[-1]
    X_rain = last_row.values.reshape(1, -1)  # 目前只有雨量特徵（例如 11 維）

    # === 補上一個 PrevDepth 特徵，讓維度變成 12 ===
    # 目前先用 0.0 當作上一時間步的淹水深度（簡化版自回歸）
    prev_depth_dummy = 0.0
    X_with_prev = np.concatenate(
        [X_rain, np.array([[prev_depth_dummy]])],
        axis=1
    )  # shape: (1, N_features)

    X_raw = X_with_prev
    n_features_input = X_raw.shape[1]

    print(f"👉 最新一筆即時輸入特徵維度 = {n_features_input}  (含 PrevDepth)")

    # 2-2 逐 IP / T 做預測
    rows = []

    for ip_id in ALL_IP_LIST:
        row = {"IP_ID": ip_id}

        for t in range(1, FutureTime + 1):
            print(f"🔄 預測 IP{ip_id:02d} 的 T{t} ...")

            model, x_min, x_range = load_model_and_scaler(ip_id, t, expected_n_features=n_features_input)

            # min-max 正規化：跟訓練時一樣
            X_norm = (X_raw - x_min) / x_range

            y_pred = model.predict(X_norm)[0]

            if y_pred < 0:
                y_pred = 0.0

            row[f"Depth_T{t}"] = float(y_pred)

        rows.append(row)

    df_out = pd.DataFrame(rows)

    # 2-3 輸出結果
    df_out.to_excel(IP_DEPTH_OUTPUT, index=False)

    print("=====================================")
    print("✅ 即時 17 測站淹水深度預測完成")
    print("📄 已輸出：", IP_DEPTH_OUTPUT)
    print("=====================================")

    # 你要的話這裡也可以 return df_out 或 IP_DEPTH_OUTPUT
    return df_out
