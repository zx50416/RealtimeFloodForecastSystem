import os
from realtime_rain_fetcher import realtime_rain_window

# 取得這支 Python 或 EXE 所在的資料夾
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 固定把即時雨量檔案放在：專案資料夾 / DATA / AD_realtime.xlsx
DATA_FOLDER = os.path.join(BASE_DIR, "DATA")
os.makedirs(DATA_FOLDER, exist_ok=True)  # 沒有 DATA 就建一個

OUTPUT_XLSX = os.path.join(DATA_FOLDER, "AD_realtime.xlsx")

df = realtime_rain_window(
    api_key="CWA-3C092B52-6E17-412E-8AD5-A7083F41BBAA",      # ← 你的中央氣象署 API Key
    output_path=OUTPUT_XLSX,  # ← 只改這一個參數
    rain_key="Past1hr",
)

print(df.tail(3))



"""
(版本: 3.10.7)
####
api_key 需要自行到中央氣象署網站申請
--根據下面的提示，跟新rain_key並依照設定的時間重複執行這段程式碼，就可以獲得xlsx (程式會自動增加即控制行數，只要重複執行即可)。
(xlsx即為AI模式的即時資料，意即透過xlsx的時間數據，讓AI計算未來四小時之淹水深度)
####


===========================================================
RainKey（rain_key 參數）說明

中央氣象署自動雨量站 API（O-A0002-001）中，
每個測站都有一個 RainfallElement 區塊，裡面包含
多種雨量時間尺度的欄位。

你可以自由選擇要抓哪一種雨量欄位，只要把
rain_key="XXXXX" 這個字串改掉即可。

以下為常見可用欄位（實際欄位會依 API 回傳為準）：

-----------------------------------------------------------
可選 RainKey 及其意義：

"Now"
    ➤ 當下雨量（立即值）

"Past10Min"
    ➤ 過去 10 分鐘累積雨量（常用）

"Past1hr"
    ➤ 過去 1 小時累積雨量（模型常用）

"Past3hr"
    ➤ 過去 3 小時累積雨量

"Past6hr"
    ➤ 過去 6 小時累積雨量

"Past12hr"
    ➤ 過去 12 小時累積雨量

"Past24hr"
    ➤ 過去 24 小時累積雨量

"Daily"
    ➤ 今日累積雨量（每日 00:00 開始計算）

-----------------------------------------------------------
⚠ 注意事項：

1. 你要確保模型訓練時使用的特徵跟這裡一致。
   例如模型用 3 小時雨量訓練，那 rain_key 就要用 "Past3hr"。

2. 若 API 沒回傳某個欄位，程式會自動填 0.0。

3. 所有測站欄位順序固定由 TARGET_STATIONS 控制，
   不會因為 API 回傳順序不同而亂掉。

===========================================================
"""

