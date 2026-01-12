# ========================================================================
# 使用說明
# ========================================================================
# 此文件使用 "SD.xlsx" 作為輸入資料，此檔案位於 "DATA" 資料夾。
# 使用交叉驗證，第一場到最後一場事件輪流做為測試資料。
# 測試結果的折線圖將輸出到 "OUTPUTS" 資料夾。
# 訓練資料的預測值與觀測值資料將寫入 "RES-train.xlsx"，
# 測試資料的預測值與觀測值資料將寫入 "RES-test.xlsx"。
# ========================================================================
# 參數設定
# ========================================================================
import os
import time
import gc
from keras import backend as K

import RunModel as RM
import data_processor as dp
import RES_gen
import index_processor as idp

# -------------------------------
# 基本參數設定
# -------------------------------
ModelName   = 'XGBoost'   # 設定模式名稱 (output 用，改名避免覆蓋舊結果)
use_autoreg = False        # ✅ 總開關：是否啟用自回歸特徵（PrevDepth）

# 全部「實際存在」的淹水點（1~28 但中間有缺）
ALL_IP_LIST = [
    1, 2, 3, 5, 7, 8,
    10, 12, 13, 14, 15, 16,
    17, 18, 25, 27, 28
]

StartPoint = 1    # 想從第幾個淹水點開始（依 IP 編號）
EndPoint   = 28   # 想到第幾個淹水點結束（依 IP 編號）

FutureTime  = 4   # ✅ 直接用 int，比較乾淨：預報 T+1 ~ T+4
index_names = ['RMSE', 'MAE', 'CE', 'CC', 'EQp', 'ETp']  # 計算指標

# -------------------------------
# 模型超參數（現在主要給 XGBoost 用）
# -------------------------------
epochs     = 50
batch_size = 32
lr         = 0.001
loss_fn    = 'mse'

# BPNN 專用參數（暫時沒用，但先留著）
bpnn_hidden_units  = 64
bpnn_learning_rate = lr

# SVM 專用參數（暫時沒用）
svm_kernel  = 'rbf'
svm_C       = 1.0
svm_epsilon = 0.01

# CNN 專用參數（暫時沒用）
cnn_dropout_rate = 0.25
cnn_filters_1    = 64
cnn_filters_2    = 32
cnn_dense_units  = 32

# LSTM & GRU 專用參數（暫時沒用）
dropout_rate = 0.25
units        = 64
num_layers   = 4

# -------------------------------
# 路徑設定
# -------------------------------
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/'
DATA_FOLDER    = PROJECT_FOLDER + 'DATA/'
OUTPUT_ROOT    = PROJECT_FOLDER + 'OUTPUTS/'
OUTPUT_FOLDER_NAME = ModelName + '/'   # e.g. "XGBoost_AR/"

# 確保 OUTPUTS 至少存在，方便最後寫 log
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ========================================================================
# 主程式
# ========================================================================
start_time = time.time()

# 依 StartPoint, EndPoint 篩選出要跑的 IP（例如 1~28，但自動略過不存在的）
ACTIVE_IP_LIST = [ip for ip in ALL_IP_LIST if StartPoint <= ip <= EndPoint]

for ip_id in ACTIVE_IP_LIST:
    for DeltaT in range(FutureTime):
        print('● 淹水點:', str(ip_id), '● 預報未來T+', str(DeltaT+1))

        # 輸出結果資料夾路徑（維持你原本這個命名規則）
        OUTPUT_FOLDER = (
            OUTPUT_ROOT
            + OUTPUT_FOLDER_NAME
            + ModelName
            + str(ip_id)
            + '-'
            + str(DeltaT+1)
            + '/'
        )

        WEIGHTS_FOLDER      = OUTPUT_FOLDER + 'weights/'
        HYDROGRAPH_FOLDER   = OUTPUT_FOLDER + 'Hydrographs/'
        SCATTER_PLOT_FOLDER = OUTPUT_FOLDER + 'Scatter_Plots/'

        # 資料前處理：AD + IOLag → SD.xlsx
        SD_FILE = RM.data_preprocess(
            DATA_FOLDER,
            str(ip_id),
            str(DeltaT + 1)
        )

        # 建置模式 + 交叉驗證 + 自回歸推論
        Y, boundaries, RES_train, RES_test, events, event_orders = RM.ConstructModel(
            SD_FILE,
            OUTPUT_FOLDER,
            HYDROGRAPH_FOLDER,
            SCATTER_PLOT_FOLDER,
            WEIGHTS_FOLDER,
            epochs, batch_size, lr, loss_fn,
            dropout_rate, units, num_layers,
            bpnn_hidden_units, bpnn_learning_rate,
            svm_kernel, svm_C, svm_epsilon,
            cnn_dropout_rate, cnn_filters_1, cnn_filters_2, cnn_dense_units,
            DeltaT,
            use_autoreg=False   # ← 用上面那個總開關
        )

        # 訓練完釋放 Keras / TF 的 graph
        K.clear_session()
        gc.collect()

        # ====================================================
        # 輸出結果 (寫入 excel 檔)
        # ====================================================
        RES_train_path  = OUTPUT_FOLDER + 'RES-train.xlsx'
        RES_test_path   = OUTPUT_FOLDER + 'RES-test.xlsx'
        eventWithMaxVal = dp.get_eventWithMaxVal(Y, boundaries)

        # 寫入 train / test 預測結果
        RES_gen.gen_RES_train(
            RES_train, events, eventWithMaxVal,
            boundaries, event_orders, RES_train_path
        )
        RES_gen.gen_RES_test(
            RES_test, events, eventWithMaxVal, RES_test_path
        )

        # 計算 Index 並寫入 Index.xlsx
        Index_path = OUTPUT_FOLDER + 'Index.xlsx'
        num_events = len(events)

        index_train, index_test = idp.get_all_indices(
            num_events, RES_train, RES_test, index_names
        )
        idp.write_Index(
            num_events, eventWithMaxVal,
            index_names, index_train, index_test, Index_path
        )

# ========================================================================
# 結束時間 & 執行紀錄
# ========================================================================
end_time = time.time()

total_time_seconds = end_time - start_time
total_time_minutes = total_time_seconds / 60
total_time_hours   = total_time_minutes / 60

print()
print(f"程式總執行時間：{total_time_seconds:.2f} 秒")
print(f"程式總執行時間：{total_time_minutes:.2f} 分鐘")
print(f"程式總執行時間：{total_time_hours:.2f} 小時")

log_text = []
log_text.append("======= 模型執行時間記錄 =======")
log_text.append(f"總執行時間（秒）：{total_time_seconds:.2f}")
log_text.append(f"總執行時間（分鐘）：{total_time_minutes:.2f}")
log_text.append(f"總執行時間（小時）：{total_time_hours:.2f}")
log_text.append("================================")

with open(OUTPUT_ROOT + "run_time_log.txt", "w", encoding="utf-8") as f:
    for line in log_text:
        f.write(line + "\n")

print("\n✅ 模型執行時間已儲存至 'OUTPUTS/run_time_log.txt'")
