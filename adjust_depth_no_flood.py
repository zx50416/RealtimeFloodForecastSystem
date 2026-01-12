import os
import shutil
import pandas as pd

# ===== 目錄設定 =====
INPUT_DIR = "./DATA"           # 已經縮減好的 IPxx.xlsx 在這裡
OUTPUT_DIR = "./adjusted_DATA" # 調整後要輸出的資料夾
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 颱風分頁名稱（欄位順序要跟你表格一致）=====
TYPHOON_SHEETS = [
    "2001_桃芝",
    "2004_敏督利",
    "2005_海棠",
    "2008_辛樂克",
    "2009_莫拉克",
    "2012_蘇拉",
    "2013_蘇力",
    "2015_蘇迪勒",
    "2016_梅姬",
    "2017_海棠",
]

DEPTH_COL = "Depth (cm)"   # 需要被改成 -1 的欄位名稱

# ===== 這一塊是你提供的 0 / 1 表，改成程式可用的 dict =====
# 1 = 有淹水（保留原本 Depth）
# 0 = 沒淹水（把 Depth 改成 -1）

IP_FLOOD_FLAG = {
    1:  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 麥寮-IP1
    2:  [1, 1, 1, 0, 1, 1, 1, 1, 1, 0],  # 崙背-IP2
    3:  [1, 1, 0, 0, 1, 1, 0, 0, 0, 0],  # 莿桐-IP3
    5:  [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],  # 斗六(1)-IP5
    7:  [1, 1, 1, 0, 1, 1, 0, 0, 0, 0],  # 斗南-IP7
    8:  [1, 1, 0, 0, 1, 1, 1, 0, 1, 0],  # 虎尾-IP8
    10: [1, 1, 0, 0, 1, 1, 0, 1, 0, 1],  # 土庫(2)-IP10
    12: [1, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # 褒忠-IP12
    13: [1, 1, 0, 1, 1, 1, 0, 0, 0, 0],  # 東勢-IP13
    14: [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 臺西(1)-IP14
    15: [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],  # 臺西(2)-IP15
    16: [1, 1, 0, 0, 1, 1, 0, 0, 0, 0],  # 四湖-IP16
    17: [0, 1, 0, 1, 1, 1, 0, 0, 0, 0],  # 口湖(1)-IP17
    18: [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],  # 水林(1)-IP18
    25: [1, 1, 0, 0, 1, 1, 0, 0, 0, 0],  # 元長(3)-IP25
    27: [1, 1, 0, 0, 1, 1, 0, 0, 0, 0],  # 大埤(1)-IP27
    28: [1, 1, 0, 1, 0, 1, 0, 0, 0, 0],  # 大埤(2)-IP28
}


def adjust_one_ip_file(ip_index):
    """
    調整單一 IPxx 檔案：
    - 依照 IP_FLOOD_FLAG 判斷哪些颱風沒淹水
    - 對那些分頁把 Depth (cm) 改成 -1
    - 另存到 ./adjusted_DATA/IPxx.xlsx
    """
    filename = f"IP{ip_index:02d}.xlsx"
    input_path = os.path.join(INPUT_DIR, filename)

    if not os.path.exists(input_path):
        print("⚠️ 找不到檔案，略過：", input_path)
        return

    output_path = os.path.join(OUTPUT_DIR, filename)

    # 如果這個 IP 不在表內，就原封不動 copy 過去
    if ip_index not in IP_FLOOD_FLAG:
        print(f"ℹ️ IP{ip_index:02d} 不在 flood 表內，直接複製原檔。")
        shutil.copyfile(input_path, output_path)
        return

    print(f"\n📂 處理 {filename} ...")

    flags = IP_FLOOD_FLAG[ip_index]

    # 建一個 dict：typhoon_name -> flag (0 or 1)
    typhoon_to_flag = {}
    for i in range(len(TYPHOON_SHEETS)):
        name = TYPHOON_SHEETS[i]
        value = flags[i]
        typhoon_to_flag[name] = value

    # 讀取原始 Excel
    xls = pd.ExcelFile(input_path)
    sheet_names = xls.sheet_names

    writer = pd.ExcelWriter(output_path, engine="openpyxl")

    for sheet in sheet_names:
        df = pd.read_excel(input_path, sheet_name=sheet)

        # 只對我們有列在 TYPHOON_SHEETS 的分頁做處理
        if sheet in typhoon_to_flag:
            flag = typhoon_to_flag[sheet]

            # flag = 0 代表該 IP 在這場沒淹水 → Depth 改成 -1
            if flag == 0:
                if DEPTH_COL in df.columns:
                    print(f"   ▶ {sheet}：沒淹水，將 Depth 改成 -1")
                    df[DEPTH_COL] = -1
                else:
                    print(f"   ❗ {sheet}：找不到欄位「{DEPTH_COL}」，無法調整 Depth")
            else:
                print(f"   ▶ {sheet}：有淹水，保留原始 Depth")
        else:
            # 不在我們的颱風名單（可能是其他測試分頁），直接照原樣寫回去
            print(f"   ▶ {sheet}：非指定颱風分頁，原樣保留")

        df.to_excel(writer, sheet_name=sheet, index=False)

    writer.close()
    print("   💾 已輸出：", output_path)


def run_adjust_all_ip():
    """
    對 IP01 ~ IP28 依序執行調整。
    沒檔案的會略過，不在 IP_FLOOD_FLAG 的會直接 copy 原檔。
    """
    for ip in range(1, 29):
        adjust_one_ip_file(ip)
