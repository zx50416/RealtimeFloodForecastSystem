# 村里平均淹水深度與等級計算用程式（UTF-8 正常版）
# 依據你剛剛已在 QGIS 另存 UTF-8 的情況，已完全移除亂碼編碼猜測器

import os
import numpy as np
import pandas as pd
import geopandas as gpd

# -----------------------------
# 路徑設定（全部用相對路徑）
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER    = os.path.join(BASE_DIR, "DATA")
SPATIAL_FOLDER = os.path.join(DATA_FOLDER, "SPATIAL")
OUTPUT_FOLDER  = os.path.join(BASE_DIR, "OUTPUTS")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

VILLAGE_SHP_PATH = os.path.join(SPATIAL_FOLDER, "雲林縣_village.shp")
IP_POINTS_SHP_PATH = os.path.join(SPATIAL_FOLDER, "過濾後淹水點.shp")

IP_DEPTH_XLSX_PATH = os.path.join(OUTPUT_FOLDER, "realtime_ip_depth.xlsx")
VILLAGE_OUTPUT_XLSX = os.path.join(OUTPUT_FOLDER, "village_depth_for_risk.xlsx")


# -----------------------------
# 1. 區間 → 等級
# -----------------------------
def depth_to_level(depth_cm):
    if depth_cm is None:
        return 0
    if depth_cm < 0:
        depth_cm = 0.0

    if depth_cm < 20:  return 1
    if depth_cm < 40:  return 2
    if depth_cm < 60:  return 3
    if depth_cm < 80:  return 4
    return 5


# -----------------------------
# 2. IDW
# -----------------------------
def idw_for_villages(ip_xy, vil_xy, ip_depth, power=2.0):
    num_vil = vil_xy.shape[0]
    num_ip  = ip_xy.shape[0]

    vil_xy_exp = vil_xy.reshape(num_vil, 1, 2)
    ip_xy_exp  = ip_xy.reshape(1, num_ip, 2)

    diff = vil_xy_exp - ip_xy_exp
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    dist[dist == 0] = 1e-6

    weights = 1.0 / (dist ** power)

    depth_matrix = ip_depth.reshape(1, num_ip)

    num = np.sum(weights * depth_matrix, axis=1)
    den = np.sum(weights, axis=1)

    z = num / den
    z = np.maximum(z, 0.0)
    return z


# -----------------------------
# 3. 主流程
# -----------------------------
def build_village_risk_table():

    # 3-1. 讀 Village（UTF-8，不必再猜 encoding）
    if not os.path.exists(VILLAGE_SHP_PATH):
        raise FileNotFoundError("找不到 Village shp：" + VILLAGE_SHP_PATH)

    gdf_vil = gpd.read_file(VILLAGE_SHP_PATH)

    for col in ["COUNTYNAME", "TOWNNAME", "VILLNAME"]:
        if col not in gdf_vil.columns:
            raise ValueError("Village shp 缺少欄位：" + col)

    gdf_vil["centroid"] = gdf_vil.geometry.centroid
    vil_xy = np.column_stack([
        gdf_vil["centroid"].x.values,
        gdf_vil["centroid"].y.values
    ])

    # 3-2. 讀淹水點（UTF-8）
    if not os.path.exists(IP_POINTS_SHP_PATH):
        raise FileNotFoundError("找不到淹水點 shp：" + IP_POINTS_SHP_PATH)

    gdf_ip = gpd.read_file(IP_POINTS_SHP_PATH)

    if "OBJECTID" not in gdf_ip.columns:
        raise ValueError("淹水點 shp 缺少 OBJECTID 欄位")

    ip_ids_all = gdf_ip["OBJECTID"].astype(int).tolist()
    ip_xy = np.column_stack([gdf_ip.geometry.x.values,
                             gdf_ip.geometry.y.values])

    # 3-3. 讀深度
    if not os.path.exists(IP_DEPTH_XLSX_PATH):
        raise FileNotFoundError("找不到預報檔：" + IP_DEPTH_XLSX_PATH)

    df_depth = pd.read_excel(IP_DEPTH_XLSX_PATH)

    if "IP_ID" not in df_depth.columns:
        raise ValueError("深度檔必須含 IP_ID 欄位")

    df_depth = df_depth.set_index("IP_ID")

    common_ids = [i for i in ip_ids_all if i in df_depth.index]
    if len(common_ids) == 0:
        raise ValueError("沒有任何共同 OBJECTID，請檢查")

    df_depth = df_depth.loc[common_ids]

    id_to_idx = {oid: idx for idx, oid in enumerate(ip_ids_all)}
    ip_xy = np.array([[gdf_ip.geometry.x.values[id_to_idx[oid]],
                       gdf_ip.geometry.y.values[id_to_idx[oid]]]
                      for oid in common_ids])

    # 3-4. 建輸出
    out_df = pd.DataFrame({
        "COUNTYNAME": gdf_vil["COUNTYNAME"],
        "TOWNNAME":   gdf_vil["TOWNNAME"],
        "VILLNAME":   gdf_vil["VILLNAME"]
    })

    # 3-5. T1~T4
    for t in range(1, 5):
        col = f"Depth_T{t}"
        ip_depth = df_depth[col].values.astype(float)

        vil_depth = idw_for_villages(ip_xy, vil_xy, ip_depth)

        out_df[col] = vil_depth
        out_df[f"H_{t}"] = [depth_to_level(x) for x in vil_depth]

    # 3-6. 存檔
    out_df.to_excel(VILLAGE_OUTPUT_XLSX, index=False)
    print("✅ 已輸出到：", VILLAGE_OUTPUT_XLSX)


# # -----------------------------
# # 執行
# # -----------------------------
# build_village_risk_table()
