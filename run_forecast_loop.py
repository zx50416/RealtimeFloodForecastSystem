# run_forecast_loop.py
# 功能：每 1 小時自動跑一輪：
#   1) 17 測站淹水深度預報
#   2) IDW → 村里 H_1~H_4
#   3) merge 到 final_output_updated.xlsx
#
# 說明：
# - 會自動抓 exe / .py 所在資料夾當成 BASE_DIR
# - 會寫一份 forecast_flood_log.txt 做簡單紀錄
# - while True + sleep(3600) 無限 loop

import os
import sys
import time
import traceback

from predict_ip_depth import run_all_predictions
from depth_to_village import build_village_risk_table
from merge_village_levels_to_final import merge_village_levels_to_final


def get_base_dir():
    """
    回傳專案根目錄：
      - 開發時：這支 .py 所在路徑
      - 打包成 exe 後：exe 所在路徑
    """
    if getattr(sys, "frozen", False):
        # 被 PyInstaller 打包成 exe
        return os.path.dirname(sys.executable)
    else:
        # 一般用 python 執行 .py
        return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = get_base_dir()

# 確保程式的工作目錄就是專案根目錄
os.chdir(BASE_DIR)

# 簡單 log 檔
LOG_PATH = os.path.join(BASE_DIR, "forecast_flood_log.txt")


def log(msg):
    """同時印在螢幕 & 寫進 log 檔"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"[{ts}] {msg}"
    print(line, flush=True)

    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # 就算寫 log 失敗也不要讓主程式掛掉
        pass


def run_once():
    """執行一輪完整預報流程"""
    log("=== 開始一輪即時淹水預報 ===")

    try:
        # 1) 讀 AD_realtime.xlsx → 17 測站淹水深度（realtime_ip_depth.xlsx）
        run_all_predictions()
        log("步驟 1 完成：17 測站淹水深度預報已更新 (realtime_ip_depth.xlsx)")

        # 2) IDW → 村里 centroid → H_1~H_4（village_depth_for_risk.xlsx）
        build_village_risk_table()
        log("步驟 2 完成：村里平均淹水深度與等級已更新 (village_depth_for_risk.xlsx)")

        # 3) merge → final_output_updated.xlsx
        merge_village_levels_to_final()
        log("步驟 3 完成：final_output_updated.xlsx 已更新")

        log("✅ 本輪預報流程成功結束\n")

    except Exception as e:
        err = repr(e)
        tb = traceback.format_exc()
        log("❌ 本輪預報流程失敗：" + err)
        log(tb)
        log("⚠ 發生錯誤，本輪流程中止，等待下一輪再試。\n")


# 不用 if __name__ == '__main__'，直接進 while True
while True:
    run_once()
    log("⏰ 休息 1 小時後再執行下一輪 ...")
    # 3600 秒 = 1 小時，如要測試可以暫時改成 30 或 60
    time.sleep(3600)
