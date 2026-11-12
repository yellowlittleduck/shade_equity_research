import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =================================================================
# 1. 数据读取与预处理
# =================================================================
file_path_provincial = r"E:\research\elderly_data\cities_xlsx\provincial_shade.xlsx"
file_path_builtup = r"E:\research\elderly_data\cities_builtup_shade\provincial_builtup_shade.xlsx"
file_path_original = r"E:\research\elderly_data\cities_builtup_shade\provincial_original.xlsx"
file_path_20 = r"E:\research\elderly_data\cities_builtup_shade\provincial_builtup20.xlsx"
file_path_30 = r"E:\research\elderly_data\cities_builtup_shade\provincial_builtup20.xlsx"
file_path_p50 = r"E:\research\elderly_data\cities_builtup_shade\provincial_builtupp50.xlsx"
file_path_p70 = r"E:\research\elderly_data\cities_builtup_shade\provincial_builtupp70.xlsx"
file_path = file_path_provincial
df = pd.read_excel(file_path)

# 基本清洗：确保列名包含 DB, DSM, 以及代表老年人口的列 (例如 age_65)
# 如果您的列名是 p_elderly，请相应修改
elderly_col = "age_65" 
df = df.dropna(subset=[elderly_col, "DB", "DSM"])
df = df[df[elderly_col] > 0].copy()

# =================================================================
# 2. 定义坐标计算函数 (专门针对 Lorenz Curve)
# =================================================================
def get_lorenz_coords(shade_vals, elderly_pop_vals):
    """
    计算洛伦兹曲线坐标：
    X轴：按遮阳时长排序后的累积老年人口比例
    Y轴：按遮阳时长排序后的累积老年人总遮阳资源比例
    """
    # 构造临时记录
    temp_df = pd.DataFrame({
        'shade': shade_vals,
        'p_elderly': elderly_pop_vals
    })
    
    # 核心逻辑：按遮阳时长 (shade) 从低到高排序
    temp_df = temp_df.sort_values('shade').reset_index(drop=True)
    
    # 计算 X 轴：累计老年人口比例
    total_e_pop = temp_df['p_elderly'].sum()
    x_coords = temp_df['p_elderly'].cumsum() / total_e_pop
    
    # 计算 Y 轴：累计老年人享有的总遮阳资源 (时长 * 对应人口)
    temp_df['total_resource'] = temp_df['shade'] * temp_df['p_elderly']
    total_resource = temp_df['total_resource'].sum()
    y_coords = temp_df['total_resource'].cumsum() / total_resource
    
    # 插入起点 (0,0)
    x = np.insert(x_coords.values, 0, 0)
    y = np.insert(y_coords.values, 0, 0)
    
    # 计算 Gini 值 (1 - 2 * AUC)
    auc = np.trapz(y, x)
    gini = 1 - 2 * auc
    
    return x, y, gini

# =================================================================
# 3. 计算坐标与 Gini 值
# =================================================================
x_db, y_db, gini_db = get_lorenz_coords(df["DB"], df[elderly_col])
x_dsm, y_dsm, gini_dsm = get_lorenz_coords(df["DSM"], df[elderly_col])

# =================================================================
# 4. 绘图
# =================================================================
plt.figure(figsize=(8, 8), dpi=100)

# 对角线：绝对公平线 (Line of Equity)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5, label="Line of equity")

# LCN – Lorenz Curve Without trees (DB)
plt.plot(
    x_db, y_db,
    color="#3b82f6", linewidth=2.5, 
    label=f"L$S_N$ (Without trees, Gini={gini_db:.3f})"
)

# LCT – Lorenz Curve With trees (DSM)
plt.plot(
    x_dsm, y_dsm,
    color="#10b981", linewidth=2.5, 
    label=f"L$S_T$ (With trees, Gini={gini_dsm:.3f})"
)

# =================================================================
# 5. 图形格式优化
# =================================================================
plt.xlabel("Cumulative percentage of elderly population (ranked by $D_k$)", fontsize=16)
plt.ylabel("Cumulative percentage of shade duration of the elderly", fontsize=16)
# plt.title("Lorenz Curves for Urban Shade Intragroup Equity", fontsize=14, pad=20)

plt.xlim(0, 1)
plt.ylim(0, 1)

# 设置坐标轴刻度标签字体大小
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(loc="upper left", frameon=True, fontsize=16)
# plt.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

# 装饰性填充
plt.fill_between(x_db, x_db, y_db, color="#3b82f6", alpha=0.1)
plt.fill_between(x_dsm, x_dsm, y_dsm, color="#10b981", alpha=0.1)

plt.tight_layout()
# plt.show()

# 保存图像 (可选)
plt.savefig(r"E:\research\Painting\Fig4\provincial_lorenz_curve1.png", dpi=300, bbox_inches="tight")
# plt.savefig(rf"E:\research\Painting\builtup\provincial_lorenz_curve_ori1.png", dpi=300, bbox_inches="tight")