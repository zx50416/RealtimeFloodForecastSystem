"""
即時雨量抓取並維持固定長度時間序列 (最多 100 列)

功能：
- 呼叫中央氣象署 O-A0002-001 自動雨量站 API
- 只抓以下 11 個測站：
    C0K330, 01J100, 01J930, 01J970, 01K060,
    01L360, 01L390, 01L480, 01L490, 01L910, 01M010
- 取 RainfallElement 裡某一個欄位（預設 Past10Min）
- 每次執行：
    1. 產生一列新的雨量資料（1 x 11）
    2. 若 output xlsx 不存在 → 建立新檔，只含這一列
    3. 若 output xlsx 已存在 → 讀舊檔、在最後加一列
    4. 若總列數 > 100 → 只保留最後 100 列（最舊那列會被刪掉）
- 檔案長相大概是：

C0K330  01J100  01J930  ...  01M010
0.0     1.5     0.0           0.3
...

-----------------------------------------------------------
RainKey（rain_key 參數）說明：

中央氣象署 O-A0002-001 的每個測站裡，會有：

"RainfallElement": {
    "Now": {"Precipitation": "0.0"},
    "Past10Min": {"Precipitation": "0.0"},
    "Past1hr": {"Precipitation": "1.5"},
    "Past3hr": {"Precipitation": "12.0"},
    "Past6hr": {...},
    "Past12hr": {...},
    "Past24hr": {...},
    "Daily": {...}
}

你可以選擇的 rain_key 範例：
- "Now"        → 當下雨量
- "Past10Min"  → 過去 10 分鐘累積雨量（預設）
- "Past1hr"    → 過去 1 小時累積雨量
- "Past3hr"    → 過去 3 小時累積雨量
- "Past6hr"    → 過去 6 小時累積雨量
- "Past12hr"   → 過去 12 小時累積雨量
- "Past24hr"   → 過去 24 小時累積雨量
- "Daily"      → 今日累積雨量（00:00 起算）

請自行確保：你訓練模型時用的是哪一個時間尺度，
這裡的 rain_key 就要設定成一樣的。
-----------------------------------------------------------

使用方式（在別的檔案）：

from realtime_rain_fetcher import realtime_rain_window

df = realtime_rain_window(
    api_key="你的 API KEY",
    output_path="./realtime_rain_input.xlsx",
    rain_key="Past10Min",   # 或 "Past1hr" 等
)

"""

import os
import requests
import pandas as pd

# ======= 你自己的授權碼要填這裡（也可以呼叫時傳入 api_key 覆蓋）=======
API_KEY = "CWA-xxxxx"  # TODO: 換成你的授權碼

# ======= CWA 自動雨量站 API URL =======
API_URL = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0002-001"

# ======= 固定順序的 11 個測站 =======
TARGET_STATIONS = [
    "C0K330",
    "01J100",
    "01J930",
    "01J970",
    "01K060",
    "01L360",
    "01L390",
    "01L480",
    "01L490",
    "01L910",
    "01M010",
]

# 想用哪一個雨量欄位就改這裡：
# 可選像 "Now", "Past10Min", "Past1hr", "Past3hr" ...
RAIN_KEY = "Past1hr"

# 時間序列最大長度（最多保留幾列）
MAX_ROWS = 24


def fetch_cwa_json(api_url: str, api_key: str) -> dict:
    """呼叫中央氣象署 API，回傳 JSON（dict）。"""
    params = {
        "Authorization": api_key
    }
    resp = requests.get(api_url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def safe_get_precip(station_data: dict, rain_key: str) -> float:
    """
    從單一測站 JSON 取出指定時間區間的 Precipitation。
    若找不到或格式怪怪的，回傳 0.0（你可以改成 None）。
    """
    rainfall = station_data.get("RainfallElement")
    if rainfall is None:
        return 0.0

    block = rainfall.get(rain_key)
    if block is None:
        return 0.0

    value_str = block.get("Precipitation")
    if value_str is None:
        return 0.0

    try:
        return float(value_str)
    except ValueError:
        return 0.0


def get_one_row_from_api(
    api_key: str,
    rain_key: str = RAIN_KEY
) -> pd.DataFrame:
    """
    呼叫 API，取出 11 測站的雨量，做成一列 DataFrame（1 x 11）。
    """
    data = fetch_cwa_json(API_URL, api_key)

    records = data.get("records", {})
    stations_list = records.get("Station", [])
    if not isinstance(stations_list, list):
        raise ValueError("JSON 結構異常：records['Station'] 不是 list")

    # 建立 StationId -> 該站 JSON 的查詢表
    id_to_station = {}
    for st in stations_list:
        sid = st.get("StationId")
        if sid is None:
            continue
        id_to_station[sid] = st

    values = []
    missing_ids = []

    for sid in TARGET_STATIONS:
        st_json = id_to_station.get(sid)
        if st_json is None:
            values.append(0.0)
            missing_ids.append(sid)
        else:
            val = safe_get_precip(st_json, rain_key)
            values.append(val)

    if len(missing_ids) > 0:
        print("⚠️ 這次 API 沒有找到下列測站，對應值將為 0.0：", missing_ids)

    df_new = pd.DataFrame([values], columns=TARGET_STATIONS)
    return df_new


def realtime_rain_window(
    api_key: str = API_KEY,
    output_path: str = "./realtime_rain_input.xlsx",
    rain_key: str = RAIN_KEY,
    max_rows: int = MAX_ROWS
) -> pd.DataFrame:
    """
    1. 呼叫 API 取得 11 測站一筆新雨量資料（1 列）
    2. 若檔案不存在：建立新檔，只有這 1 列
    3. 若檔案存在：
        - 讀舊檔
        - 在最後加上新的一列
        - 若列數大於 max_rows，則只保留最後 max_rows 列
    4. 儲存到 output_path，並回傳完整 DataFrame
    """
    # 先取得這次 API 的最新一列資料
    df_new = get_one_row_from_api(api_key=api_key, rain_key=rain_key)

    if os.path.exists(output_path):
        # 已有檔案 → 讀舊資料
        df_old = pd.read_excel(output_path)

        # 確保欄位順序與 TARGET_STATIONS 一致
        df_old = df_old.reindex(columns=TARGET_STATIONS)

        # 合併舊資料與新資料
        df_all = pd.concat([df_old, df_new], ignore_index=True)

        # 如果超過 max_rows，就只保留最後 max_rows 列
        if len(df_all) > max_rows:
            df_all = df_all.tail(max_rows).reset_index(drop=True)

        print(f"📈 追加一列，即時雨量資料已更新，目前總列數：{len(df_all)}（最多 {max_rows} 列）")

    else:
        # 檔案不存在 → 直接使用這一列當起始
        df_all = df_new.copy()
        print(f"🆕 建立新檔案：{output_path}（目前列數：{len(df_all)}）")

    # 存成 xlsx
    df_all.to_excel(output_path, index=False)
    print("✅ 已儲存即時雨量時間序列檔案：", output_path)
    print("   欄位順序：", TARGET_STATIONS)
    print("   使用雨量欄位 RainfallElement['%s']" % rain_key)

    return df_all
