import pandas as pd
import os

# ==================================================
# 文件路径
# ==================================================

shade_file = (
    r"E:\research\elderly_data\cities_xlsx\provincial_shade.xlsx"
)

zonal_file = (
    r"E:\research\elderly_data\cities_builtup_shade\zonal_elderly.csv"
)

output_dir = (
    r"E:\research\elderly_data\cities_builtup_shade"
)

os.makedirs(output_dir, exist_ok=True)

# ==================================================
# Built-up Scenarios
# ==================================================

SCENARIOS = {
    "builtupp25": 10900,
    "builtupp50": 57211,
}
    # "original": None,
    # "builtupp50": 57211,
    # "builtupp75": 141949,
# ==================================================
# 读取数据
# ==================================================

print("读取 provincial_shade ...")

df_shade = pd.read_excel(shade_file)

df_shade["id"] = (
    df_shade["id"]
    .astype(str)
    .str.split(".")
    .str[0]
    .str.strip()
)

print("读取 zonal_elderly ...")

df_zonal = pd.read_csv(zonal_file)

df_zonal["id"] = (
    df_zonal["id1"]
    .astype(str)
    .str.split(".")
    .str[0]
    .str.strip()
)

df_zonal["MEAN"] = pd.to_numeric(
    df_zonal["MEAN"],
    errors="coerce"
)

# ==================================================
# 开始生成各Scenario
# ==================================================

for scenario, threshold in SCENARIOS.items():

    print("\n" + "=" * 60)
    print(f"Scenario: {scenario}")
    print("=" * 60)

    # ----------------------------------------------
    # Original
    # ----------------------------------------------

    if threshold is None:

        selected_ids = set(
            df_zonal["id"]
        )

    # ----------------------------------------------
    # Built-up Filter
    # ----------------------------------------------

    else:

        df_selected = df_zonal[
            df_zonal["MEAN"] > threshold
        ]

        selected_ids = set(
            df_selected["id"]
        )

        print(
            f"Threshold = {threshold:,}"
        )

        print(
            f"Selected grids = "
            f"{len(selected_ids):,}"
        )

    # ----------------------------------------------
    # 去 provincial_shade 调取记录
    # ----------------------------------------------

    df_result = df_shade[
        df_shade["id"].isin(selected_ids)
    ].copy()

    print(
        f"Matched records = "
        f"{len(df_result):,}"
    )

    # ----------------------------------------------
    # 输出
    # ----------------------------------------------

    output_file = os.path.join(
        output_dir,
        f"provincial_{scenario}.xlsx"
    )

    df_result.to_excel(
        output_file,
        index=False
    )

    print(
        f"Saved: {output_file}"
    )

print("\n🎉 全部完成")