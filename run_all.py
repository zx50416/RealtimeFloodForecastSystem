# run_all.py
# 功能：每次執行流程：
#   1) 從中央氣象署 API 抓即時雨量 → DATA/AD_realtime.xlsx
#   2) 讀 AD_realtime.xlsx，用 17 個 IP × T1~T4 模型做預測
#      → OUTPUTS/realtime_ip_depth.xlsx
#   3) 用 IDW 把 17 測站淹水深度插值到村里
#      → OUTPUTS/village_depth_for_risk.xlsx
#   4) 把 H_1~H_4 merge 回 final_output.xlsx（直接覆蓋）
#   5) 每 1 小時重複以上流程一次（while True + sleep(3600)）

import os
import time
import traceback

from realtime_rain_fetcher import realtime_rain_window
from predict_ip_depth import run_all_predictions
from depth_to_village import build_village_risk_table
from merge_village_levels_to_final import merge_village_levels_to_final

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER   = os.path.join(BASE_DIR, "DATA")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "OUTPUTS")

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

AD_REALTIME_PATH = os.path.join(DATA_FOLDER, "AD_realtime.xlsx")

# === 你的 CWA API KEY ===
CWA_API_KEY = "CWA-3C092B52-6E17-412E-8AD5-A7083F41BBAA"


def run_once():
    """執行一次完整預報流程"""
    print("\n==============================")
    print("🚀 開始執行一次淹水預報流程")
    print("==============================")

    # 1) 更新雨量
    print("▶ [Step 1] 更新即時雨量 AD_realtime.xlsx ...")
    df = realtime_rain_window(
        api_key=CWA_API_KEY,
        output_path=AD_REALTIME_PATH,
        rain_key="Past1hr",
    )
    print("   ✅ 完成，即時雨量已寫入：", AD_REALTIME_PATH)
    print(df.tail(3))

    # 2) 17 測站預測
    print("\n▶ [Step 2] 預測各測站未來 4 小時淹水深度 ...")
    run_all_predictions()
    print("   ✅ 完成。")

    # 3) 村里深度與等級表
    print("\n▶ [Step 3] 進行 IDW，建立村里風險資料 ...")
    build_village_risk_table()
    print("   ✅ 完成。")

    # 4) 回寫 final_output.xlsx（不再產生 updated 檔案）
    print("\n▶ [Step 4] 更新 final_output.xlsx 的 H_1~H_4 ...")
    merge_village_levels_to_final()
    print("   ✅ final_output.xlsx 已成功更新。")

    print("\n🎉 本次預報流程全部完成！")


# 執行無限循環，1 小時跑一次
while True:
    try:
        run_once()
    except Exception as e:
        print("\n❌ 執行過程發生錯誤：", e)
        traceback.print_exc()

    print("\n⏳ 休息 1 小時後再次執行 ...")
    time.sleep(3600)
