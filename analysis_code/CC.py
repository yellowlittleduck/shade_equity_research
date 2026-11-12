'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-12-16 19:32:48
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2026-06-02 20:20:36
FilePath: \analysis\CC.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================
# 1. 读取省级汇总数据
# =============================
# 假设文件路径不变
file_path_provincial = r"E:\research\elderly_data\cities_xlsx\provincial_shade.xlsx"
file_path_builtup = r"E:\research\elderly_data\cities_builtup_shade\provincial_builtup_shade.xlsx"
file_path_original = r"E:\research\elderly_data\cities_builtup_shade\provincial_original.xlsx"
file_path_20 = r"E:\research\elderly_data\cities_builtup_shade\provincial_builtup20.xlsx"
file_path_30 = r"E:\research\elderly_data\cities_builtup_shade\provincial_builtup30.xlsx"
file_path_p25 = r"E:\research\elderly_data\cities_builtup_shade\provincial_builtupp25.xlsx"
file_path_p50 = r"E:\research\elderly_data\cities_builtup_shade\provincial_builtupp50.xlsx"
file_path_p75 = r"E:\research\elderly_data\cities_builtup_shade\provincial_builtupp75.xlsx"
file_path = file_path_provincial
df = pd.read_excel(file_path)

# 基本清洗
# 确保列名一致：population(总人口), elderly_ratio(老年人比例), DB(无树遮荫), DSM(有树遮荫)
df = df.dropna(subset=["population", "elderly_ratio", "DB", "DSM"])
df = df[df["population"] > 0].copy()

# =============================
# 2. 按老年人口比例 Ri 排序 (这是 CI 的核心：Ranking Variable)
# =============================
df = df.sort_values("elderly_ratio", ascending=True).reset_index(drop=True)

# =============================
# 3. 计算累积人口比例（X-axis）
# =============================
total_pop = df["population"].sum()
df["cum_pop"] = df["population"].cumsum() / total_pop

# =============================
# 4. 计算累计资源比例（Y-axis） - 核心修正点
# =============================
# 必须计算“人均时长 * 人口数”，得到该网格的总遮荫资源量
df["total_resource_DB"] = df["DB"] * df["population"]
df["total_resource_DSM"] = df["DSM"] * df["population"]

df["cum_DB_weighted"] = df["total_resource_DB"].cumsum() / df["total_resource_DB"].sum()
df["cum_DSM_weighted"] = df["total_resource_DSM"].cumsum() / df["total_resource_DSM"].sum()

# =============================
# 5. 验证 CI 计算 (梯形法则)
# =============================
def get_ci_from_curve(x, y):
    # 计算曲线下面积 AUC
    auc = np.trapz(y, x)
    return 1 - 2 * auc

ci_db = get_ci_from_curve(df["cum_pop"].values, df["cum_DB_weighted"].values)
ci_dsm = get_ci_from_curve(df["cum_pop"].values, df["cum_DSM_weighted"].values)

print(f"Calculated CI (Without Trees): {ci_db:.4f}")
print(f"Calculated CI (With Trees):    {ci_dsm:.4f}")
print(f"Improvement (Delta CI):       {ci_dsm - ci_db:.4f}")

# =============================
# 6. 绘图
# =============================
plt.figure(figsize=(8, 8), dpi=100)

# 对角线：绝对公平线
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5, label="Line of equity")

# CCN – Without trees
plt.plot(
    df["cum_pop"], df["cum_DB_weighted"],
    color="#3b82f6", linewidth=2.5, label=f"C$S_N$ (Without trees, CI={ci_db:.3f})"
)

# CCT – With trees
plt.plot(
    df["cum_pop"], df["cum_DSM_weighted"],
    color="#10b981", linewidth=2.5, label=f"C$S_T$ (With trees, CI={ci_dsm:.3f})"
)

# =============================
# 7. 优化图形格式
# =============================
plt.xlabel("Cumulative percentage of population (ranked by $R_{E,k}$)", fontsize=16)
plt.ylabel("Cumulative percentage of shade duration", fontsize=16)
# plt.title("Concentration Curves for Urban Shade Intergroup Equity", fontsize=16, pad=20)

plt.xlim(0, 1)
plt.ylim(0, 1)

# 设置坐标轴刻度标签字体大小
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# 让图例显示在合适的位置
plt.legend(loc="upper left", frameon=True, fontsize=16)
# plt.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

# 装饰性：填充曲线与对角线之间的面积（可选，能直观看到 CI）
plt.fill_between(df["cum_pop"], df["cum_pop"], df["cum_DB_weighted"], color="#3b82f6", alpha=0.1)
plt.fill_between(df["cum_pop"], df["cum_pop"], df["cum_DSM_weighted"], color="#10b981", alpha=0.1)

plt.tight_layout()
# plt.show()

plt.savefig( r"E:\research\Painting\Fig4\provincial_concentration_curve1.png", dpi=300, bbox_inches="tight")
# plt.savefig( rf"E:\research\Painting\builtup\provincial_concentration_curve_ori1.png", dpi=300, bbox_inches="tight")