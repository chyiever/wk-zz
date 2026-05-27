#!/usr/bin/env python3
"""修改 feature_comparison notebook：添加标题 + 特征分布对比代码"""
import json
from pathlib import Path

NOTEBOOK_PATH = Path(r"E:\codes\ZZ-BK\notebooks\2026-05-27-feature_comparison_build3_vs_batch.ipynb")

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# ── 1. 给已有单元格添加一级/二级标题 ──

# cell-0 已有总标题，改为 H1
cells[0]["source"] = [
    "# 五特征对比：build-3 自定义函数 vs batch compute_all_features\n",
    "\n",
    "使用同一 TDMS 数据文件、相同滑窗与预处理方式，分别用两种方法计算五个特征并对比差异。\n",
    "\n",
    "- **方法A**：来自 `2026-05-19-realdata_feature_dataset_build-3.ipynb` 的自定义函数（`_spectral_centroid_mean` 等）\n",
    "- **方法B**：来自 `2026-05-06-batch_feature_extract_12folders_gpu.ipynb` 的 `compute_all_features` 完整函数\n",
]

# 在 cell-1 前插入 H2
h2_import = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 1. 模块导入与配置\n"],
}
# 在 cell-2 前插入 H2
h2_config = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 2. 参数与特征配置\n"],
}
# 在 cell-3 前插入 H2
h2_tdms_helper = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 3. TDMS 加载辅助函数\n"],
}
# 在 cell-4 前插入 H2
h2_load = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 4. 加载 TDMS 数据并预处理\n"],
}
# 在 cell-5 前插入 H2
h2_band_params = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 5. 频带参数构建\n"],
}
# 在 cell-6 前插入 H2
h2_window = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 6. 滑窗生成\n"],
}
# 在 cell-7 前插入 H2
h2_method_a = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 7. 方法A：build-3 自定义特征函数\n"],
}
# 在 cell-8 前插入 H2
h2_method_b = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 8. 方法B：batch compute_all_features\n"],
}
# 在 cell-9 前插入 H2
h2_compute = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 9. 逐窗口计算两种方法的特征\n"],
}
# 在 cell-10 前插入 H2
h2_compare = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 10. 对比两种方法的结果\n"],
}
# 在 cell-11 前插入 H2
h2_detail = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 11. 详细逐窗口差值\n"],
}
# 在 cell-12 前插入 H2
h2_summary = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 12. 最终汇总\n"],
}

# 插入 H2 标题（按索引从后往前插，避免索引偏移）
inserts = [
    (1, h2_import),
    (2, h2_config),
    (3, h2_tdms_helper),
    (4, h2_load),
    (5, h2_band_params),
    (6, h2_window),
    (7, h2_method_a),
    (8, h2_method_b),
    (9, h2_compute),
    (10, h2_compare),
    (11, h2_detail),
    (12, h2_summary),
]
for idx, cell in reversed(inserts):
    cells.insert(idx, cell)

# ── 2. 追加特征分布对比单元格 ──

h2_dist = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 13. 特征概率分布对比：方法A（无标签） vs 方法B（按标签分组）\n",
        "\n",
        "对比两种计算方式产出的五种特征的概率分布：\n",
        "- **方法A**：`realdata_feature_dataset_20260519_v3` + `realdata_feature_dataset_20260523` 中所有样本合并，无标签区分\n",
        "- **方法B**：`dataset_build_260518_features_gpu` 中按 label 列分组（0=噪声，1=断丝），分别绘制分布\n",
        "\n",
        "每个特征一个子图，横轴为特征值，纵轴为概率密度（KDE）。\n",
    ],
}

code_dist_load = {
    "cell_type": "code",
    "metadata": {},
    "source": [
        "# =========================\n",
        "# 13.1 读取方法A的特征数据\n",
        "# =========================\n",
        "import glob\n",
        "from scipy.stats import gaussian_kde\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib\n",
        "\n",
        "matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']\n",
        "matplotlib.rcParams['axes.unicode_minus'] = False\n",
        "\n",
        "FEATURE_COLS = [\n",
        "    'b_1k_10k__SC_mean',\n",
        "    'b_1k_10k__C_f',\n",
        "    'b_1k_100k__epsilon_2x',\n",
        "    'b_1k_100k__SC_res_mean',\n",
        "    'b_1k_100k__I_burst',\n",
        "]\n",
        "\n",
        "# 方法A：读取 v3 和 0523 目录下的特征文件\n",
        "dir_A_v3 = workspace / 'outputs' / 'realdata_feature_dataset_20260519_v3'\n",
        "dir_A_0523 = workspace / 'outputs' / 'realdata_feature_dataset_20260523'\n",
        "\n",
        "dfs_A = []\n",
        "for d in [dir_A_v3, dir_A_0523]:\n",
        "    pattern = str(d / '*_features_*_part_*.csv')\n",
        "    for fp in sorted(glob.glob(pattern)):\n",
        "        dfs_A.append(pd.read_csv(fp, usecols=FEATURE_COLS, encoding='utf-8'))\n",
        "\n",
        "df_A_all = pd.concat(dfs_A, ignore_index=True)\n",
        "print(f'方法A 样本总数: {len(df_A_all)}')\n",
        "print(f'方法A 特征列: {list(df_A_all.columns)}')\n",
    ],
    "outputs": [],
    "execution_count": None,
}

code_dist_load_B = {
    "cell_type": "code",
    "metadata": {},
    "source": [
        "# =========================\n",
        "# 13.2 读取方法B的特征数据并按标签分组\n",
        "# =========================\n",
        "\n",
        "dir_B = workspace / 'outputs' / 'dataset_build_260518_features_gpu'\n",
        "cols_B = FEATURE_COLS + ['label']\n",
        "\n",
        "dfs_B = []\n",
        "for fp in sorted(glob.glob(str(dir_B / '*_features.csv'))):\n",
        "    dfs_B.append(pd.read_csv(fp, usecols=cols_B, encoding='utf-8'))\n",
        "\n",
        "df_B_all = pd.concat(dfs_B, ignore_index=True)\n",
        "df_B_noise = df_B_all[df_B_all['label'] == 0].reset_index(drop=True)\n",
        "df_B_break = df_B_all[df_B_all['label'] == 1].reset_index(drop=True)\n",
        "\n",
        "print(f'方法B 样本总数: {len(df_B_all)}')\n",
        "print(f'  噪声(label=0): {len(df_B_noise)}')\n",
        "print(f'  断丝(label=1): {len(df_B_break)}')\n",
    ],
    "outputs": [],
    "execution_count": None,
}

code_dist_plot = {
    "cell_type": "code",
    "metadata": {},
    "source": [
        "# =========================\n",
        "# 13.3 绘制五种特征概率分布对比图\n",
        "# =========================\n",
        "\n",
        "fig, axes = plt.subplots(1, 5, figsize=(28, 5))\n",
        "\n",
        "for idx, feat in enumerate(FEATURE_COLS):\n",
        "    ax = axes[idx]\n",
        "\n",
        "    # 方法A：所有样本 KDE\n",
        "    vals_A = df_A_all[feat].dropna().values\n",
        "    if vals_A.size > 1:\n",
        "        kde_A = gaussian_kde(vals_A)\n",
        "        x_min = vals_A.min()\n",
        "        x_max = vals_A.max()\n",
        "        x_range = x_max - x_min\n",
        "        if x_range < 1e-12:\n",
        "            x_range = 1.0\n",
        "        xs_A = np.linspace(x_min - 0.05 * x_range, x_max + 0.05 * x_range, 500)\n",
        "        ax.plot(xs_A, kde_A(xs_A), color='#1f77b4', linewidth=2, label='方法A (全样本)')\n",
        "\n",
        "    # 方法B-噪声\n",
        "    vals_B0 = df_B_noise[feat].dropna().values\n",
        "    if vals_B0.size > 1:\n",
        "        kde_B0 = gaussian_kde(vals_B0)\n",
        "        x_min0 = vals_B0.min()\n",
        "        x_max0 = vals_B0.max()\n",
        "        x_range0 = x_max0 - x_min0\n",
        "        if x_range0 < 1e-12:\n",
        "            x_range0 = 1.0\n",
        "        xs_B0 = np.linspace(x_min0 - 0.05 * x_range0, x_max0 + 0.05 * x_range0, 500)\n",
        "        ax.plot(xs_B0, kde_B0(xs_B0), color='#ff7f0e', linewidth=2, linestyle='--', label='方法B-噪声')\n",
        "\n",
        "    # 方法B-断丝\n",
        "    vals_B1 = df_B_break[feat].dropna().values\n",
        "    if vals_B1.size > 1:\n",
        "        kde_B1 = gaussian_kde(vals_B1)\n",
        "        x_min1 = vals_B1.min()\n",
        "        x_max1 = vals_B1.max()\n",
        "        x_range1 = x_max1 - x_min1\n",
        "        if x_range1 < 1e-12:\n",
        "            x_range1 = 1.0\n",
        "        xs_B1 = np.linspace(x_min1 - 0.05 * x_range1, x_max1 + 0.05 * x_range1, 500)\n",
        "        ax.plot(xs_B1, kde_B1(xs_B1), color='#d62728', linewidth=2, linestyle=':', label='方法B-断丝')\n",
        "\n",
        "    ax.set_xlabel(feat, fontsize=10)\n",
        "    ax.set_ylabel('概率密度', fontsize=10)\n",
        "    ax.legend(fontsize=8, loc='best')\n",
        "    ax.tick_params(labelsize=8)\n",
        "\n",
        "fig.suptitle('五种特征概率分布对比：方法A(无标签) vs 方法B(按标签分组)', fontsize=14, y=1.02)\n",
        "plt.tight_layout()\n",
        "plt.savefig(workspace / 'outputs' / 'feature_distribution_comparison_A_vs_B.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print('分布对比图已保存至 outputs/feature_distribution_comparison_A_vs_B.png')\n",
    ],
    "outputs": [],
    "execution_count": None,
}

# 追加到 cells 末尾
cells.append(h2_dist)
cells.append(code_dist_load)
cells.append(code_dist_load_B)
cells.append(code_dist_plot)

# ── 3. 写回 notebook ──
with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook 已更新: {NOTEBOOK_PATH}")
print(f"总单元格数: {len(cells)}")
