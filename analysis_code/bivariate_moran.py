import geopandas as gpd
import pandas as pd
import numpy as np
import libpysal
from esda.moran import Moran_Local_BV
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from esda.lee import Spatial_Pearson
import os

# =================================================================
# 1. 配置变量与基准城市（杭州）范围计算
# =================================================================
shade_col = "DST"  
elderly_col = "age_65" 
input_dir = r"E:\research\elderly_data\cities"
output_dir = r"E:\research\Painting\Fig5"
# 确保包含杭州以获取基准尺寸
cities = ['hangzhou', 'lishui'] 

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("正在计算杭州基准尺寸以统一比例尺...")
hz_path = rf"{input_dir}\hangzhou_fullfishnet.shp"
if not os.path.exists(hz_path):
    raise FileNotFoundError("未找到杭州(hangzhou)数据，无法确立统一比例尺基准。")

# 加载杭州并计算其最大边长
hz_temp = gpd.read_file(hz_path).to_crs(epsg=3857)
hz_bounds = hz_temp.total_bounds # [minx, miny, maxx, maxy]
hz_max_dim = max(hz_bounds[2] - hz_bounds[0], hz_bounds[3] - hz_bounds[1])

# 统一视口半长：在杭州最大边长基础上预留 10% 的地图内边距
# 图例将通过 bbox_to_anchor 放在这个范围之外
global_view_half_size = (hz_max_dim * 1.1) / 2
print(f"✅ 已锁定统一视口跨度（基于杭州）: {global_view_half_size * 2 / 1000:.2f} km")

# =================================================================
# 2. 批量处理循环
# =================================================================
# 如果只想输出特定城市，可以在此处修改列表，但 global_view_half_size 已由杭州决定
process_cities = ['hangzhou', 'wenzhou', 'jinhua', 'taizhou', 'jiaxing', 'huzhou', 'shaoxing', 'lishui', 'ningbo', 'quzhou'] 

for city in process_cities:
    print(f"\n处理城市: {city.upper()}")
    file_path = rf"{input_dir}\{city}_fullfishnet.shp"
    if not os.path.exists(file_path): continue

    data = gpd.read_file(file_path)
    data[elderly_col] = pd.to_numeric(data[elderly_col], errors='coerce').fillna(0)
    data[shade_col] = pd.to_numeric(data[shade_col], errors='coerce').fillna(0)
    data = data[data.geometry.notnull()].copy().reset_index(drop=True)

    # 空间权重与统计计算 (KNN-8)
    w = libpysal.weights.KNN.from_dataframe(data, k=8)
    w.transform = 'r' 
    moran_loc_bv = Moran_Local_BV(data[shade_col].values, data[elderly_col].values, w, permutations=999)
    
    # Lee's L 指标
    lee = Spatial_Pearson(connectivity=w.sparse, permutations=999)
    lee.fit(data[shade_col].values.reshape(-1, 1), data[elderly_col].values.reshape(-1, 1))
    
    # LISA 分类映射
    sig_level = 0.05
    significant = moran_loc_bv.p_sim < sig_level
    data['lisa_cat'] = 0
    data.loc[significant & (moran_loc_bv.q == 1), 'lisa_cat'] = 1 # HH
    data.loc[significant & (moran_loc_bv.q == 2), 'lisa_cat'] = 2 # LH
    data.loc[significant & (moran_loc_bv.q == 3), 'lisa_cat'] = 3 # LL
    data.loc[significant & (moran_loc_bv.q == 4), 'lisa_cat'] = 4 # HL

    # =================================================================
    # 3. 绘图：强制统一比例尺 + 外部图例
    # =================================================================
    lisa_colors = {0: '#eeeeee', 1: '#e76254', 2: '#376795', 3: '#72bcd5', 4: '#ffd06f'}
    lisa_labels = {0: 'Not Significant', 1: 'High-High', 2: 'Low-High (Priority)', 3: 'Low-Low', 4: 'High-Low'}

    data_plot = data.to_crs(epsg=3857)
    
    # 创建画布：宽 15 英寸，高 10 英寸（宽出部分用于放图例）
    fig, ax = plt.subplots(1, 1, figsize=(15, 10), dpi=300)
    
    # 绘制研究区背景轮廓
    study_area_union = gpd.GeoSeries([data_plot.union_all()], crs=data_plot.crs)
    study_area_union.plot(ax=ax, color='#f8f8f8', edgecolor='#333333', linewidth=0.8, zorder=1)

    # 循环绘制 LISA 各类别的格网
    for cat, color in lisa_colors.items():
        subset = data_plot[data_plot['lisa_cat'] == cat]
        if not subset.empty:
            subset.plot(ax=ax, color=color, edgecolor='none', alpha=0.9, zorder=2)

    # --- 核心：统一比例尺控制 ---
    # 以当前城市中心为原点，展开与杭州相同大小的视口
    bounds = data_plot.total_bounds
    cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
    
    ax.set_xlim(cx - global_view_half_size, cx + global_view_half_size)
    ax.set_ylim(cy - global_view_half_size, cy + global_view_half_size)
    ax.set_aspect('equal')
    ax.axis('off')

    # --- 核心：外部图例控制 ---
    patches = [mpatches.Patch(color=lisa_colors[i], label=f"{lisa_labels[i]} (n={np.sum(data['lisa_cat']==i)})") for i in range(1, 5)]
    stats_text = f"Global Lee's L: {lee.association_[0, 1]:.4f}\nP-value: {lee.significance_[0, 1]:.3f}"
    patches.append(mpatches.Patch(color='none', label=stats_text))
    
    # bbox_to_anchor(1.05, 0.5) 将图例放在轴右侧（x=1.05），垂直居中（y=0.5）
    ax.legend(handles=patches, title=f'Bivariate LISA: {city.upper()}', 
              loc='center left', bbox_to_anchor=(1.0, 0.5), 
              frameon=True, fontsize=12, title_fontsize=14, borderpad=1.5)

    # 4. 保存
    save_path = fr"{output_dir}\{city}_bivariate_moran.png"
    # bbox_inches='tight' 会自动包含右侧延伸出的图例部分
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    plt.close() 

    print(f"✅ {city.upper()} 处理完成并已保存至 Fig5 目录。")