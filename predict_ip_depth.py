import os
import numpy as np
import pandas as pd
import joblib

# ==============================
# 基本設定
# ==============================
ModelName = "XGBoost_AR"

ALL_IP_LIST = [
    1, 2, 3, 5, 7, 8,
    10, 12, 13, 14, 15, 16,
    17, 18, 25, 27, 28
]

FutureTime = 4  # T1~T4

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER   = os.path.join(BASE_DIR, "DATA")
MODEL_ROOT    = os.path.join(BASE_DIR, "MODEL_WEIGHTS", ModelName)
OUTPUT_FOLDER = os.path.join(BASE_DIR, "OUTPUTS")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

AD_REALTIME_PATH = os.path.join(DATA_FOLDER, "AD_realtime.xlsx")
IP_DEPTH_OUTPUT  = os.path.join(OUTPUT_FOLDER, "realtime_ip_depth.xlsx")


# ==============================
# 1. 載入模型工具
# ==============================
def load_xgb_model_flexible(model_path: str):
    model = joblib.load(model_path)
    print(f"   ✔ 以 joblib.load() 載入：{model_path}")
    return model


def load_minmax_params(scaler_path: str, expected_n_features: int):
    """
    舊版 min-max scaler 讀取（目前已停用）。
    為了不破壞既有 def 名稱而保留，但部署流程不再呼叫它。
    """
    raise RuntimeError("目前部署端已停用 min-max scaler；請勿再呼叫 load_minmax_params()。")


def load_model_and_scaler(ip_id: int, t_id: int, expected_n_features: int):
    """
    保留 def 名稱以確保相容。
    新版只載入 model，不載入 scaler。
    回傳：(model, None, None)
    """
    ip_folder  = os.path.join(MODEL_ROOT, f"IP{ip_id:02d}")
    model_path = os.path.join(ip_folder, f"T{t_id}_model.bin")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 模型不存在：{model_path}")

    model = load_xgb_model_flexible(model_path)
    return model, None, None


# ==============================
# 2. 序列 rolling 自回歸推進
# ==============================
def _extract_rain_matrix(df_ad: pd.DataFrame) -> np.ndarray:
    """
    從 AD_realtime.xlsx 抽出雨量特徵矩陣 (N, 11)
    - 只保留數值欄位
    - 轉 float32
    """
    df_num = df_ad.apply(pd.to_numeric, errors="coerce")
    df_num = df_num.dropna(axis=1, how="all")  # 去掉整欄都非數值的欄
    if df_num.shape[1] == 0:
        raise ValueError("❌ AD_realtime.xlsx 沒有任何可用的數值雨量欄位。")

    X_rain = df_num.values.astype(np.float32)
    return X_rain


def _roll_autoreg_over_sequence(model, X_rain_seq: np.ndarray) -> float:
    """
    對某一個 (IP, T) 的模型，沿著整段雨量序列做 rolling 自回歸：
      prev = 0
      for each row i:
         y = model.predict([rain_i..., prev])
         prev = y
      return 最後一筆的 y
    """
    prev = 0.0
    last_pred = 0.0

    for i in range(X_rain_seq.shape[0]):
        x = X_rain_seq[i:i+1, :]  # (1, 11)
        X_in = np.concatenate([x, np.array([[prev]], dtype=np.float32)], axis=1)  # (1, 12)
        y = float(model.predict(X_in)[0])
        if y < 0:
            y = 0.0
        last_pred = y
        prev = y

    return float(last_pred)


# ==============================
# 3. 跑一輪預測（保持原 def 名稱）
# ==============================
def run_all_predictions():
    """
    新版定義（符合你想要的「每輪初始化」）：
    - 讀 AD_realtime.xlsx 全部列（最多 100 列）
    - 對每個 IP、每個 T(1~4)：
        用整段序列 rolling 自回歸（首筆 PrevDepth=0）
        回傳最後一筆的 Depth_Tt
    - 輸出 OUTPUTS/realtime_ip_depth.xlsx
    """
    if not os.path.exists(AD_REALTIME_PATH):
        raise FileNotFoundError("❌ 找不到 AD_realtime.xlsx，請先執行即時雨量抓取程式。")

    df_ad = pd.read_excel(AD_REALTIME_PATH)
    if df_ad.shape[0] == 0:
        raise ValueError("❌ AD_realtime.xlsx 沒有任何列，至少要有一列資料。")

    X_rain_seq = _extract_rain_matrix(df_ad)  # (N, 11)
    print(f"👉 AD_realtime 時序列數 = {X_rain_seq.shape[0]}，雨量特徵數 = {X_rain_seq.shape[1]}")

    rows = []
    for ip_id in ALL_IP_LIST:
        row = {"IP_ID": ip_id}

        for t in range(1, FutureTime + 1):
            print(f"🔄 預測 IP{ip_id:02d} 的 T{t}（序列 rolling, 首筆 PrevDepth=0）...")

            model, _, _ = load_model_and_scaler(ip_id, t, expected_n_features=X_rain_seq.shape[1] + 1)
            y_last = _roll_autoreg_over_sequence(model, X_rain_seq)
            row[f"Depth_T{t}"] = y_last

        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_excel(IP_DEPTH_OUTPUT, index=False)

    print("=====================================")
    print("✅ 即時 17 測站淹水深度預測完成（每輪從序列首筆 PrevDepth=0 初始化）")
    print("📄 已輸出：", IP_DEPTH_OUTPUT)
    print("=====================================")

    return df_out
