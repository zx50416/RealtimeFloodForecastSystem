import numpy as np
import pandas as pd
import openpyxl

# 載入資料
def load_data(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    events = [wb[sheetname] for sheetname in wb.sheetnames]
    for i in range(len(events)):
        events[i] = pd.DataFrame(get_values(events[i]))
    return events

# 取得資料值並儲存為列表
def get_values(sheet):
    arr = []
    for row in sheet:
        temp = []
        for column in row:
            temp.append(column.value)
        arr.append(temp)
    return arr

# ✅ 已移除 min-max 正規化功能：保留介面避免其他程式報錯
def normalize(data):
    return data

# ✅ 已移除反正規化功能：保留介面避免其他程式報錯
def denormalize(data, original):
    return data

# 分割訓練及測試資料
def split_data(X, Y, split_boundary):
    X_train = X[:split_boundary]
    Y_train = Y[:split_boundary]
    X_test = X[split_boundary:]
    Y_test = Y[split_boundary:]
    return X_train, Y_train, X_test, Y_test

# ✅ 產生可輸入模型的資料及標籤（自動加入 PrevDepth）
def create_sequences(events):
    """
    每個事件（sheet）的資料格式假設為：
      - 前面欄位：雨量特徵
      - 最後一欄：label（淹水深度）

    自動新增 PrevDepth（自回歸特徵）：
      - 每個事件第一筆 PrevDepth = 0
      - 後面 PrevDepth = 前一筆 label（同事件內）
    """
    X_all, Y_all = [], []

    for ev_df in events:
        df = ev_df.copy()

        # features：除最後一欄以外
        ref = df.iloc[:, :(len(df.columns) - 1)]
        # label：最後一欄
        pred = df.iloc[:, (len(df.columns) - 1)].astype(np.float32)

        # PrevDepth：事件內 shift，一開始補 0
        prev_depth = pred.shift(1).fillna(0.0).astype(np.float32)

        # 合併 features：雨量 + PrevDepth
        ref_with_pd = ref.copy()
        ref_with_pd["PrevDepth"] = prev_depth

        for i in range(len(ref_with_pd)):
            X_all.append(ref_with_pd.iloc[i, :].values.astype(np.float32))
            Y_all.append(float(pred.iloc[i]))

    X = np.array(X_all, dtype=np.float32)  # (N, num_features+1)
    Y = np.array(Y_all, dtype=np.float32).reshape(-1, 1)  # (N, 1)

    # 保留你原本輸出格式：(samples, features, 1)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, Y

# 整理輸入資料順序以便輸入模型
def reorder_events(events, event_order):
    test_event = events[0]  # 取得第一場事件資料為測試資料
    events = events[1:]     # 第二場~最後一場
    events.append(test_event)  # 將測試資料移到陣列尾端

    temp = event_order[0]
    event_order = event_order[1:]
    event_order.append(temp)

    boundary = []
    num_events = len(events)
    for i in range(num_events):
        boundary.append(sum([len(events[j]) for j in range(i + 1)]))
    split_boundary = boundary[-2]  # 訓練與測試資料的分割邊界

    return events, event_order, boundary, split_boundary

def convert_negative_to_zero(data):
    return np.array([max(0, num) for num in data], dtype=np.float32)

def get_eventWithMaxVal(Y, boundaries):
    indexOfMaxVal = list(Y).index(np.max(Y))  # 計算 event with max value
    for i in range(len(boundaries[0])):
        if indexOfMaxVal < boundaries[0][i]:
            eventWithMaxVal = i + 1
            break
    return eventWithMaxVal
