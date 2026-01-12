# ========================================================================
# 02_MAIN_export_weights.py
# ========================================================================
# 功能：
# - 不跑交叉驗證
# - 依照 main.py 的資料來源與 data_preprocess
# - 使用「全部事件資料」訓練 XGBoost_AR（含 PrevDepth）
# - 將最終部署模型輸出到：
#   MODEL_WEIGHTS/XGBoost_AR/IPxx/Ty_model.bin
# ========================================================================

import os
import time
import gc
from keras import backend as K

import RunModel as RM

# ------------------------------------------------------------------------
# 基本設定（與 main.py 對齊）
# ------------------------------------------------------------------------
ModelName   = 'XGBoost_AR'
use_autoreg = True

ALL_IP_LIST = [
    1, 2, 3, 5, 7, 8,
    10, 12, 13, 14, 15, 16,
    17, 18, 25, 27, 28
]

StartPoint = 1
EndPoint   = 28
FutureTime = 4

# ------------------------------------------------------------------------
# 路徑設定（與 main.py 一致）
# ------------------------------------------------------------------------
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/'
DATA_FOLDER    = PROJECT_FOLDER + 'DATA/'

# ⚠️ 最重要的輸出位置（最終定案）
WEIGHTS_ROOT = PROJECT_FOLDER + 'MODEL_WEIGHTS/XGBoost_AR/'

os.makedirs(WEIGHTS_ROOT, exist_ok=True)

ACTIVE_IP_LIST = [ip for ip in ALL_IP_LIST if StartPoint <= ip <= EndPoint]

# ========================================================================
# 主程式
# ========================================================================
start_time = time.time()

print("======================================")
print("🚀 02_MAIN_export_weights")
print("🚀 使用全部事件資料訓練部署模型")
print("🚀 輸出至 MODEL_WEIGHTS/XGBoost_AR/")
print("======================================")

for ip_id in ACTIVE_IP_LIST:
    ip_folder = os.path.join(WEIGHTS_ROOT, f"IP{int(ip_id):02d}")
    os.makedirs(ip_folder, exist_ok=True)

    for DeltaT in range(FutureTime):
        print(f"\n● [EXPORT] IP{int(ip_id):02d}  T+{DeltaT+1}")

        # ------------------------------------------------------------
        # 1) 與 main.py 完全相同的資料前處理（產 SD.xlsx）
        # ------------------------------------------------------------
        SD_FILE = RM.data_preprocess(
            DATA_FOLDER,
            str(ip_id),
            str(DeltaT + 1)
        )

        # ------------------------------------------------------------
        # 2) 用全部事件資料訓練並存部署模型
        # ------------------------------------------------------------
        RM.export_final_xgb_ar_weights(
            SD_FILE=SD_FILE,
            WEIGHTS_ROOT=ip_folder,     # ← 已經是 IPxx 這層
            delta_t=DeltaT + 1,
            use_autoreg=use_autoreg
        )

        K.clear_session()
        gc.collect()
    
end_time = time.time()
print("\n✅ 全部權重輸出完成")
print(f"總耗時：{(end_time - start_time)/60:.2f} 分鐘")
