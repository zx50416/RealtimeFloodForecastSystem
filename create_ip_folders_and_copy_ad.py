"""
批次建立 IP 子資料夾並複製 AD.xlsx

功能說明：
- 從 ./adjusted_DATA 底下讀取 IPXX.xlsx
- 在 ./DATA 底下，對每個 IPXX 建立四個子資料夾：
    IPXX-1, IPXX-2, IPXX-3, IPXX-4
- 將對應的 ./adjusted_DATA/IPXX.xlsx 複製到每個子資料夾中，
  並命名為 AD.xlsx

注意：
- 只會處理 IP_LIST 裡面列出的 IP 編號
- 若 adjusted_DATA 裡找不到該 IPXX.xlsx 會印出警告，略過
"""

import os
import shutil

# ===== 資料夾路徑設定 =====
ADJUSTED_DIR = "./adjusted_DATA"  # 已整理好的 IPXX.xlsx 在這裡
DATA_DIR = "./DATA"               # 要建立子資料夾的根目錄

# 確保 ./DATA 存在
os.makedirs(DATA_DIR, exist_ok=True)

# ===== 需要處理的 IP 編號（你目前有的 17 個點）=====
IP_LIST = [
    1, 2, 3, 5, 7, 8,
    10, 12, 13, 14, 15, 16,
    17, 18, 25, 27, 28
]

# 每個 IP 要建立幾個時間子資料夾（IPXX-1 ~ IPXX-4）
NUM_TIME_FOLDERS = 4


def create_ip_subfolders_and_copy():
    """
    對 IP_LIST 中每個 IPXX：

    1. 檢查 ./adjusted_DATA/IPXX.xlsx 是否存在
    2. 在 ./DATA 建立 IPXX-1 ~ IPXX-4 四個資料夾
    3. 在每個資料夾中複製一份 AD.xlsx
    """
    for ip in IP_LIST:
        ip_name = f"IP{ip:02d}"  # 轉成 IP01、IP02 這種格式
        src_file = os.path.join(ADJUSTED_DIR, f"{ip_name}.xlsx")

        if not os.path.exists(src_file):
            print(f"⚠️ 找不到來源檔案：{src_file}，此 IP 略過。")
            continue

        print(f"\n📂 處理 {ip_name} ...")

        for t in range(1, NUM_TIME_FOLDERS + 1):
            folder_name = f"{ip_name}-{t}"
            folder_path = os.path.join(DATA_DIR, folder_name)

            # 建立子資料夾
            os.makedirs(folder_path, exist_ok=True)

            # 目標檔案路徑：./DATA/IPXX-k/AD.xlsx
            dst_file = os.path.join(folder_path, "AD.xlsx")

            # 複製檔案
            shutil.copyfile(src_file, dst_file)

            print(f"   ✅ 建立資料夾 {folder_name} 並複製 AD.xlsx")


# 你在其他程式要用就呼叫這個函式
# 例如在 main.py 寫：from create_ip_folders_and_copy_ad import create_ip_subfolders_and_copy
# 然後執行 create_ip_subfolders_and_copy()
