#!/usr/bin/env python
# coding: utf-8

# # 淹水空間點位數量
# IPnums_str = '9'; IPnums = int(IPnums_str)

# # 定義母資料夾路徑
# PROJECT_FOLDER = 'C:/MyPython/Model_淹水預報/01_Regular Model/'
# SVM_PATH = PROJECT_FOLDER +  'outputs/SVM/'
# LSTM_PATH = PROJECT_FOLDER + 'outputs/LSTM/'
# GRU_PATH = PROJECT_FOLDER +  'outputs/GRU/'
# main_folders = [SVM_PATH, LSTM_PATH, GRU_PATH]

# # 指定檔案路徑(畫圖 & 輸出Excel用)
# output_folder = 'C:/MyPython/Model_淹水預報/01_Regular Model/outputs/Analysis (Graphs)'
# os.makedirs(output_folder, exist_ok=True)     # 如果路徑不存在，則創建路徑


import os
import pandas as pd
import numpy as np
import plotting_utils as pltUT
import time

# 定義路徑
PROJECT_FOLDER = 'C:/MyPython/Model_淹水預報/01_Regular Model/'
HYDROGRAPH_FOLDER = PROJECT_FOLDER + 'outputs/Analysis (Hydrographs)/'    # 歷線圖的資料夾路徑
SCATTER_PLOT_FOLDER = PROJECT_FOLDER + 'outputs/Analysis (Scatters)/'     # 散點圖的資料夾路徑

# 定義訓練集和測試集的大小
train_size = 100
test_size = 50

# 產生隨機數組
obv_train = np.random.rand(train_size)
est_train = np.random.rand(train_size)
obv_test = np.random.rand(test_size)
est_test = np.random.rand(test_size)

#***************************************************************************
# 畫圖
#***************************************************************************
# 判斷路徑中是否已有"Hydrographs"及"Scatter_Plots"兩資料夾
if not os.path.exists(HYDROGRAPH_FOLDER):
    os.mkdir(HYDROGRAPH_FOLDER)
    print(f"Created directory: {HYDROGRAPH_FOLDER}")
else:
    print(f"Directory already exists: {HYDROGRAPH_FOLDER}")

if not os.path.exists(SCATTER_PLOT_FOLDER):
    os.mkdir(SCATTER_PLOT_FOLDER)
    print(f"Created directory: {SCATTER_PLOT_FOLDER}")
else:
    print(f"Directory already exists: {SCATTER_PLOT_FOLDER}")
    
fig_names = ['Hydrograph', 'Scatter plot']
fig_folders = {'Hydrograph':HYDROGRAPH_FOLDER, 'Scatter plot':SCATTER_PLOT_FOLDER}
pltUT.draw_all(fig_names, fig_folders, 0, obv_train, est_train, obv_test, est_test)