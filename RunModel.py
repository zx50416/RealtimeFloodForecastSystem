import os
import numpy as np
import AD2SD
import data_processor as dp
from my_models import My_XGBoost  # 目前只用 XGBoost，之後有需要再擴充
import plotting_utils as pltUT
import gc
import pandas as pd
import psutil


# =========================================================
# 資料前處理：AD + IOLag → SD.xlsx
# =========================================================
def data_preprocess(DATA_FOLDER, IPnum, DeltaT):
    """
    依照你的原本資料夾結構，把 AD.xlsx + IOLag.xlsx 轉成 SD.xlsx
    路徑例如：DATA/IP01-1/AD.xlsx, IOLag.xlsx → SD.xlsx
    """
    AD_FILE    = DATA_FOLDER + 'IP0' + IPnum + '-' + DeltaT + '/AD.xlsx'
    IOLAG_FILE = DATA_FOLDER + 'IP0' + IPnum + '-' + DeltaT + '/IOLag.xlsx'
    SD_FILE    = DATA_FOLDER + 'IP0' + IPnum + '-' + DeltaT + '/SD.xlsx'

    ADevents = AD2SD.read_AD(AD_FILE)
    iolag    = AD2SD.read_IOLag(IOLAG_FILE)
    sd       = AD2SD.ad2sd(ADevents, iolag)
    AD2SD.write_SD(ADevents, sd, SD_FILE)

    return SD_FILE


# =========================================================
# 內部工具：用 AD.xlsx 決定要不要跳過事件
# =========================================================
def _build_skip_mask_from_AD(sd_file, events):
    """
    回傳 skip_mask：True 表示該事件 Depth 全為 -1，要跳過。
    事件數量多少都可以，跟 10 場 / 7 場無關。
    """
    ad_folder = os.path.dirname(sd_file)
    ad_file   = os.path.join(ad_folder, "AD.xlsx")

    skip_mask = []
    if os.path.isfile(ad_file):
        xls_ad = pd.ExcelFile(ad_file)
        for sheet in xls_ad.sheet_names:
            df_ad = pd.read_excel(ad_file, sheet_name=sheet)
            df_ad.columns = df_ad.columns.astype(str).str.strip()
            if "Depth (cm)" in df_ad.columns:
                depth = pd.to_numeric(df_ad["Depth (cm)"], errors="coerce")
                # 該事件所有 Depth 都是 -1（或 NaN → 填成 -1）就跳過
                flag_all_minus1 = depth.notna().size > 0 and (depth.fillna(-1) == -1).all()
                skip_mask.append(flag_all_minus1)
            else:
                # 沒有 Depth 欄 → 不跳（保守處理）
                skip_mask.append(False)
    else:
        # 找不到 AD.xlsx → 全部不跳
        skip_mask = [False] * len(events)

    # 保險：長度不對就全部不跳
    if len(skip_mask) != len(events):
        skip_mask = [False] * len(events)

    return skip_mask


# =========================================================
# 內部工具：根據 boundary 產生 PrevDepth（不跨颱風）
# =========================================================
def _build_prev_depth_per_sample(Y_flat, boundary_list):
    """
    Y_flat: shape (num_samples,) 的真實水深
    boundary_list: 例如 [len(ev1), len(ev1)+len(ev2), ...]
    回傳 prev_depth_flat: 同樣 shape (num_samples,)
      - 每個事件的第一個樣本 prev_depth = 0
      - 之後 prev_depth(i) = Y_flat(i-1)（同一事件內）
    """
    num_samples = len(Y_flat)
    prev_depth = np.zeros_like(Y_flat, dtype=np.float32)

    start = 0
    for b in boundary_list:
        end = b  # [start, end) 是同一事件
        if end > start:
            # 該事件第一筆 → 0（保持 0）
            # 後面每一筆 = 前一筆 Y
            prev_depth[start + 1:end] = Y_flat[start:end - 1]
        start = end

    return prev_depth


# =========================================================
# 主模型建構＋交叉驗證
# =========================================================
def ConstructModel(
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
    use_autoreg=False,
):
    """
    這裡做的事：
    1. 讀 SD.xlsx → events (list of DataFrame)，每個 sheet 一個事件
    2. 根據 AD.xlsx 決定要跳過哪些事件（Depth 全 -1）
    3. 做「事件級交叉驗證」：每一折把最後一個事件當 test，其餘當 train
    4. 如果 use_autoreg=True：
       - 訓練時：PrevDepth = 同一事件內前一筆真實水深
       - 測試時：PrevDepth = 模型上一時間步的預測值（自回歸）
    5. 回傳：
       - Y_first_fold: 第一折的 Y 序列
       - boundaries  : 每一折的 boundary
       - RES_train / RES_test: 各折的 (觀測, 預測)
       - events, event_orders: 最終一次迭代的事件內容與順序
    """

    # -----------------------------------------------------
    # 1. 載入 SD.xlsx → events（list of DataFrame）
    # -----------------------------------------------------
    events = dp.load_data(SD_FILE)

    # 用 AD.xlsx 決定哪些事件要跳過（Depth 全 -1）
    skip_mask = _build_skip_mask_from_AD(SD_FILE, events)

    filtered_events = []
    for ev, skip in zip(events, skip_mask):
        if not skip:
            filtered_events.append(ev)

    if len(filtered_events) == 0:
        raise ValueError("AD.xlsx 中所有事件的 Depth (cm) 皆為 -1，無可訓練之事件。")

    events = filtered_events
    num_events = len(events)

    # 保留原本事件編號（1-based）
    event_order = []
    for idx, skip in enumerate(skip_mask):
        if not skip:
            event_order.append(idx + 1)

    # -----------------------------------------------------
    # 2. 交叉驗證需要記錄的東西
    # -----------------------------------------------------
    boundaries   = []   # 每一折對應的 boundary（累積長度）
    event_orders = []   # 每一折的事件順序
    RES_train    = []   # 每一折訓練集 [obv, est]
    RES_test     = []   # 每一折測試集 [obv, est]

    # 用來回傳給外面的 Y（搭配 boundaries[0] 使用）
    Y_first_fold = None

    # =====================================================
    # 3. 交叉驗證：每一折輪流拿最後一個事件當 test
    #    num_events 可大可小，跟 10 場沒關係
    # =====================================================
    for ev_idx in range(num_events):

        # -------------------------------------------------
        # 3-1. 重新排序事件／決定 train/test 切點
        # -------------------------------------------------
        events, event_order, boundary, split_boundary = dp.reorder_events(events, event_order)
        event_orders.append(event_order)
        boundaries.append(boundary)

        # -------------------------------------------------
        # 3-2. 產生 X_raw, Y_raw（尚未正規化）
        # -------------------------------------------------
        # X_raw: (samples, num_features, 1)
        # Y_raw: (samples, 1)
        X_raw, Y_raw = dp.create_sequences(events)
        Y_raw_vec = Y_raw.reshape(-1)   # (samples,)

        # 第一折的 Y + boundary 會被 get_eventWithMaxVal 使用
        if ev_idx == 0:
            Y_first_fold = Y_raw_vec.copy()

        num_samples = X_raw.shape[0]

        # 共用的「基礎特徵」（不包含 PrevDepth），攤平成 (samples, D_base)
        X_base_flat = X_raw.reshape(num_samples, -1)

        # -------------------------------------------------
        # 3-3. 如果有啟用自回歸：先算「真實前一格水深」（用來訓練）
        # -------------------------------------------------
        if use_autoreg:
            print("⚙ 啟用自回歸特徵（訓練）：PrevDepth = 同一颱風事件內前一筆真實水深（首筆 = 0）")
            prev_depth_full = _build_prev_depth_per_sample(Y_raw_vec, boundary)
        else:
            prev_depth_full = None

        # -------------------------------------------------
        # 3-4. 在「原始尺度」切出 train / test
        # -------------------------------------------------
        X_train_base = X_base_flat[:split_boundary]      # (N_train, D_base)
        Y_train_raw  = Y_raw_vec[:split_boundary]        # (N_train,)

        X_test_base  = X_base_flat[split_boundary:]      # (N_test, D_base)
        Y_test_raw   = Y_raw_vec[split_boundary:]        # (N_test,)

        if use_autoreg:
            prev_train = prev_depth_full[:split_boundary]    # (N_train,)
        else:
            prev_train = None

        # -------------------------------------------------
        # 3-5. 建立「訓練用特徵矩陣」：X_train_with_prev
        # -------------------------------------------------
        if use_autoreg:
            prev_train_col = prev_train.reshape(-1, 1)       # (N_train, 1)
            X_train_with_prev = np.concatenate([X_train_base, prev_train_col], axis=1)
        else:
            X_train_with_prev = X_train_base

        # -------------------------------------------------
        # 3-6. 只用「訓練資料」算 min/max，再縮放 train
        # -------------------------------------------------
        X_train_final = X_train_with_prev
        Y_train = Y_train_raw.reshape(-1)
        Y_test  = Y_test_raw.reshape(-1)

        # -------------------------------------------------
        # 3-7. 建立權重檔儲存路徑
        # -------------------------------------------------
        if not os.path.exists(WEIGHTS_FOLDER):
            os.makedirs(WEIGHTS_FOLDER, exist_ok=True)

        weights_path = os.path.join(
            WEIGHTS_FOLDER,
            'Weights_EV' + "%02d" % (ev_idx + 1) + '.h5'
        )

        # =================================================
        # 4. 訓練 XGBoost 模型
        # =================================================
        print('\n[第 %d/%d 次訓練]' % (ev_idx + 1, num_events))
        print(
            '▶ 以第 '
            + ', '.join(str(x) for x in sorted(event_order[:-1]))
            + ' 場事件為訓練資料，以第 '
            + str(event_order[-1])
            + ' 場事件為測試資料\n'
        )

        model = My_XGBoost(
            max_depth=4,
            learning_rate=0.05,
            n_estimators=500
        )

        model = model.train(X_train_final, Y_train)
        Y_train_predict = model.predict(X_train_final).reshape(-1)

        # =================================================
        # 5. 測試資料：改成「真正自回歸」推論（只看自己前一格預測）
        # =================================================
        if use_autoreg:
            print("⚙ 測試階段使用『自回歸推論』：PrevDepth = 前一時間步模型預測值（首筆 = 0）")

            num_test = len(Y_test_raw)
            Y_pred_roll = np.zeros(num_test, dtype=np.float32)

            # 測試事件在「整體序列」中的起點 index
            start_idx_global = split_boundary

            # 第一筆 PrevDepth = 0
            prev_pred = 0.0

            for i_local in range(num_test):
                idx_global = start_idx_global + i_local

                # 該筆的基礎特徵
                x_base = X_base_flat[idx_global]  # (D_base,)

                # 把上一時間步的預測值接到最後
                x_with_prev = np.concatenate(
                    [x_base, np.array([prev_pred], dtype=np.float32)],
                    axis=0
                )  # (D_base+1,)

                # 用訓練時的 min/max 正規化
                x_final = x_with_prev.reshape(1, -1)
                y_hat = model.predict(x_final)[0]

                Y_pred_roll[i_local] = y_hat

                # 更新下一步 PrevDepth
                prev_pred = y_hat

            Y_predict = Y_pred_roll

        else:
            # 沒開自回歸就單純一次性預測（全部 test 一次丟進去）
            X_test_with_prev = X_test_base
            X_test_final = X_test_with_prev
            Y_predict = model.predict(X_test_final).reshape(-1)


        # =================================================
        # 6. 資料後處理（負值改成 0）
        # =================================================
        obv_train = dp.convert_negative_to_zero(Y_train)
        est_train = dp.convert_negative_to_zero(Y_train_predict)
        obv_test  = dp.convert_negative_to_zero(Y_test)
        est_test  = dp.convert_negative_to_zero(Y_predict)

        RES_train.append([obv_train, est_train])
        RES_test.append([obv_test, est_test])

        # =================================================
        # 7. 畫圖
        # =================================================
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            print(f"Created directory: {OUTPUT_FOLDER}")
        else:
            print(f"Directory already exists: {OUTPUT_FOLDER}")

        if not os.path.exists(HYDROGRAPH_FOLDER):
            os.makedirs(HYDROGRAPH_FOLDER, exist_ok=True)
            print(f"Created directory: {HYDROGRAPH_FOLDER}")
        else:
            print(f"Directory already exists: {HYDROGRAPH_FOLDER}")

        if not os.path.exists(SCATTER_PLOT_FOLDER):
            os.makedirs(SCATTER_PLOT_FOLDER, exist_ok=True)
            print(f"Created directory: {SCATTER_PLOT_FOLDER}")
        else:
            print(f"Directory already exists: {SCATTER_PLOT_FOLDER}")

        fig_names = ['Hydrograph', 'Scatter plot']
        fig_folders = {
            'Hydrograph': HYDROGRAPH_FOLDER,
            'Scatter plot': SCATTER_PLOT_FOLDER
        }

        pltUT.draw_all(fig_names, fig_folders, ev_idx, obv_train, est_train, obv_test, est_test)

        # =================================================
        # 8. 記憶體清理（簡單版：只 gc.collect，不亂 del 未宣告變數）
        # =================================================
        gc.collect()

        process = psutil.Process(os.getpid())
        print(f"🧠 目前記憶體使用：{process.memory_info().rss / 1024 ** 2:.2f} MB")

    # 回傳「第一折」的 Y（對應 boundaries[0]），給 get_eventWithMaxVal 使用
    return Y_first_fold, boundaries, RES_train, RES_test, events, event_orders




# ================== RunModel.py 最底部新增 ==================

import data_processor as dp
from my_models import My_XGBoost

def export_final_xgb_ar_weights(SD_FILE, WEIGHTS_ROOT, delta_t, use_autoreg=True):
    """
    使用全部事件資料訓練 XGBoost_AR，並輸出部署用權重
    輸出格式：
      WEIGHTS_ROOT/T{delta_t}_model.bin
    """

    print("  ▶ Final training with all events")

    events = dp.load_data(SD_FILE)

    X_all = []
    y_all = []

    for ev in events:
        df = ev.copy()
        X_rain = df.iloc[:, :-1].astype(np.float32)
        y = df.iloc[:, -1].astype(np.float32)

        if use_autoreg:
            prev = y.shift(1).fillna(0.0).astype(np.float32)
            X_ev = X_rain.copy()
            X_ev["PrevDepth"] = prev.values
        else:
            X_ev = X_rain

        X_all.append(X_ev.values)
        y_all.append(y.values)

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0).reshape(-1)

    model = My_XGBoost(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=800
    )
    model.train(X, y)

    out_path = os.path.join(WEIGHTS_ROOT, f"T{delta_t}_model.bin")
    model.save(out_path)

    print("  💾 Saved:", out_path)
