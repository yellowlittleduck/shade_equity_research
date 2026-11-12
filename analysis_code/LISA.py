'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2026-01-18 16:20:37
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2026-01-18 16:20:40
FilePath: \analysis\LISA.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import geopandas as gpd
import pandas as pd
import numpy as np
import libpysal
from esda.moran import Moran_Local_BV
import os

'''
此脚本用于统计双变量 LISA 分类（遮荫 vs 老年人口）下的树木遮荫净贡献（DST）。
通过计算各分类（HH, LH, HL, LL）的平均树木遮荫时长，分析树木对空间公平性的实际贡献。
'''

# 1. 配置参数
shade_total_col = "DSM"    # 总遮荫时长（用于计算LISA现状）
elderly_pop_col = "age_65" # 老年人口（需求指标）
tree_contribution_col = "DST" # 树木净遮荫贡献时长（你想分析的贡献因子）

cities = ['hangzhou', 'wenzhou', 'jinhua', 'taizhou', 'jiaxing', 
          'huzhou', 'shaoxing', 'lishui', 'ningbo', 'quzhou']
input_dir = r"E:\research\elderly_data\cities"
output_csv = r"E:\research\Painting\Fig5\lisa_tree_contribution_stats.csv"

all_city_stats = []

# 2. 批量处理
for city in cities:
    print(f"正在处理城市: {city.upper()}...")
    file_path = os.path.join(input_dir, f"{city}_fullfishnet.shp")
    
    if not os.path.exists(file_path):
        continue

    try:
        data = gpd.read_file(file_path)
        
        # 数据清洗
        for col in [shade_total_col, elderly_pop_col, tree_contribution_col]:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        
        # 仅在有老年人的区域计算 LISA
        valid_data = data[data[elderly_pop_col] > 0].copy().reset_index(drop=True)
        
        if len(valid_data) < 10:
            continue

        # 3. 计算双变量 LISA (DSM vs age_65)
        w = libpysal.weights.KNN.from_dataframe(valid_data, k=8)
        w.transform = 'r'
        
        moran_loc_bv = Moran_Local_BV(valid_data[shade_total_col].values, 
                                      valid_data[elderly_pop_col].values, w)
        
        # 4. 分类映射
        # q=1:HH, q=2:LH, q=3:LL, q=4:HL
        sig = moran_loc_bv.p_sim < 0.05
        valid_data['lisa_cat'] = 0  # 默认为不显著
        valid_data.loc[sig & (moran_loc_bv.q == 1), 'lisa_cat'] = 1 # HH
        valid_data.loc[sig & (moran_loc_bv.q == 2), 'lisa_cat'] = 2 # LH
        valid_data.loc[sig & (moran_loc_bv.q == 3), 'lisa_cat'] = 3 # LL
        valid_data.loc[sig & (moran_loc_bv.q == 4), 'lisa_cat'] = 4 # HL

        # 5. 统计各分类下的平均树木贡献 (DST)
        cat_mapping = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
        city_result = {'City': city.capitalize()}
        
        for cat_id, cat_name in cat_mapping.items():
            subset = valid_data[valid_data['lisa_cat'] == cat_id]
            if not subset.empty:
                avg_tree_shade = subset[tree_contribution_col].mean()
                city_result[f'Avg_DST_{cat_name}'] = round(avg_tree_shade, 4)
            else:
                city_result[f'Avg_DST_{cat_name}'] = np.nan
        
        # 计算该城市所有有效格网的平均树木时长作为基准
        city_result['City_Avg_DST'] = round(valid_data[tree_contribution_col].mean(), 4)
        
        all_city_stats.append(city_result)
        print(f"✅ {city.upper()} 统计完成")

    except Exception as e:
        print(f"❌ 处理 {city} 时出错: {e}")

# 6. 导出结果
if all_city_stats:
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n任务结束。结果已保存至: {output_csv}")