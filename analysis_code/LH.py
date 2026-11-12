import geopandas as gpd
import pandas as pd
import numpy as np
import libpysal
from esda.moran import Moran_Local_BV
import os

# ==========================================================

# 参数设置

# ==========================================================

shade_col = "DST"          # 用于LISA分析的总遮荫时长
elderly_col = "age_65"    # 老年人口

cities = [
'hangzhou',
'ningbo',
'wenzhou',
'jinhua',
'taizhou',
'jiaxing',
'shaoxing',
'huzhou',
'lishui',
'quzhou'
]

input_dir = r"E:\research\elderly_data\cities"

output_excel = (
r"E:\research\Painting\Fig5\LH_priority_summary.xlsx"
)

results = []

for city in cities:

    print(f"Processing {city}...")

    shp_path = os.path.join(
        input_dir,
        f"{city}_fullfishnet.shp"
    )

    if not os.path.exists(shp_path):
        print(f"Missing: {shp_path}")
        continue

    gdf = gpd.read_file(shp_path)

    # ------------------------------------------------------
    # 数据清洗
    # ------------------------------------------------------

    gdf[shade_col] = pd.to_numeric(
        gdf[shade_col],
        errors="coerce"
    ).fillna(0)

    gdf[elderly_col] = pd.to_numeric(
        gdf[elderly_col],
        errors="coerce"
    ).fillna(0)

    gdf = gdf[gdf.geometry.notnull()].copy()

    # 只保留有老年人口的网格
    valid = gdf[gdf[elderly_col] > 0].copy()

    if len(valid) < 20:
        continue

    # ------------------------------------------------------
    # KNN权重矩阵
    # ------------------------------------------------------

    w = libpysal.weights.KNN.from_dataframe(
        valid,
        k=8
    )

    w.transform = "r"

    # ------------------------------------------------------
    # Bivariate LISA
    # ------------------------------------------------------

    lisa = Moran_Local_BV(
        valid[shade_col].values,
        valid[elderly_col].values,
        w,
        permutations=999
    )

    sig = lisa.p_sim < 0.05

    valid["lisa_cat"] = 0

    valid.loc[
        sig & (lisa.q == 1),
        "lisa_cat"
    ] = 1

    valid.loc[
        sig & (lisa.q == 2),
        "lisa_cat"
    ] = 2

    valid.loc[
        sig & (lisa.q == 3),
        "lisa_cat"
    ] = 3

    valid.loc[
        sig & (lisa.q == 4),
        "lisa_cat"
    ] = 4

    # ------------------------------------------------------
    # 统计指标
    # ------------------------------------------------------

    total_grids = len(valid)

    significant_grids = np.sum(
        valid["lisa_cat"] > 0
    )

    lh_grids = np.sum(
        valid["lisa_cat"] == 2
    )

    lh_grid_pct = (
        lh_grids / significant_grids * 100
        if significant_grids > 0
        else np.nan
    )

    total_elderly = valid[
        elderly_col
    ].sum()

    lh_elderly = valid.loc[
        valid["lisa_cat"] == 2,
        elderly_col
    ].sum()

    lh_elderly_pct = (
        lh_elderly / total_elderly * 100
        if total_elderly > 0
        else np.nan
    )

    results.append({
        "City": city.capitalize(),
        "Total_Grids": total_grids,
        "Significant_Grids": significant_grids,
        "LH_Grids": lh_grids,
        "LH_Grid_Pct": round(lh_grid_pct, 2),
        "Total_Elderly": round(total_elderly, 0),
        "LH_Elderly": round(lh_elderly, 0),
        "LH_Elderly_Pct": round(lh_elderly_pct, 2)
    })
# ==========================================================

# 导出

# ==========================================================

df = pd.DataFrame(results)

df.to_excel(
output_excel,
index=False
)

print("\nFinished.")
print(df)

print(f"\nSaved to:\n{output_excel}")
