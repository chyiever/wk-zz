"""Insert section 2.5 into 2026-05-17_cross_condition_experiment.ipynb"""
import json

NB_PATH = r'E:\codes\ZZ-BK\notebooks\2026-05-17_cross_condition_experiment.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Insert after cell index 25 (id=a57d1771, last cell of 2.4)
INSERT_AFTER = 25

markdown_source = (
    "## 2.5 最不重要特征排序（Least Important Feature Ranking）\n"
    "\n"
    "在 2.2 RFECV 和 2.4 跨数据集筛选的基础上，将**全量特征**按重要性**升序**排列，"
    "依次给出最不重要的 10 / 50 / 100 / 200 个特征组。\n"
    "\n"
    "**目的（Purpose）**\n"
    "- 识别对分类贡献最小的特征，为后续特征剪枝（feature pruning）或降维提供依据。\n"
    "- 将 RFECV 排名（`rfecv_ranking`，越大越不重要）与 Stage-A 分数（`stage_a_score`，越小越不重要）融合，"
    "生成综合重要性排名。\n"
    "\n"
    "**输入（Inputs）**\n"
    "- `rfecv_results`：2.2 节各 Top-K 下的 RFECV 结果（含 `.summary` DataFrame）。\n"
    "- `stage_a_scores`：2.1 节 Stage-A 全量评分。\n"
    "- `dataset_feature_results`：2.4 节各数据集的筛选结果（可选，用于跨数据集聚合）。\n"
    "\n"
    "**输出（Outputs）**\n"
    "- `least_important_df`：全量特征按重要性升序排列的 DataFrame。\n"
    "- 各阈值（10 / 50 / 100 / 200）下最不重要特征列表及可视化。\n"
)

code_source = (
    "# 目的（Purpose）: 在 2.2/2.4 筛选基础上，将全量特征按重要性升序排序，给出最不重要的特征组。\n"
    "# 输入（Inputs）: `rfecv_results`（2.2）、`stage_a_scores`（2.1）、`dataset_feature_results`（2.4，可选）\n"
    "# 输出（Outputs）: `least_important_df`（升序排列），各阈值特征列表，可视化图。\n"
    "# 说明（Notes）: RFECV ranking 越大 = 越不重要；Stage-A score 越小 = 越不重要；两者归一化后取均值得综合排名。\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "\n"
    "# --------------------------------------------------------------------------\n"
    "# Step 1: 聚合 Stage-A 分数（全量特征，取 base_train_mask 对应的那次评分）\n"
    "# --------------------------------------------------------------------------\n"
    "stage_a_df = stage_a_scores.copy() if hasattr(stage_a_scores, 'copy') else pd.DataFrame(stage_a_scores)\n"
    "\n"
    "score_col = next((c for c in ['score', 'stage_a_score', 'importance'] if c in stage_a_df.columns), None)\n"
    "feat_col  = next((c for c in ['feature', 'feature_name', 'name'] if c in stage_a_df.columns), None)\n"
    "if score_col is None or feat_col is None:\n"
    "    print('stage_a_scores columns:', list(stage_a_df.columns))\n"
    "    raise ValueError('Cannot locate feature/score columns in stage_a_scores')\n"
    "\n"
    "stage_a_info = stage_a_df[[feat_col, score_col]].rename(\n"
    "    columns={feat_col: 'feature', score_col: 'stage_a_score'}\n"
    ")\n"
    "\n"
    "# --------------------------------------------------------------------------\n"
    "# Step 2: 聚合 RFECV ranking（取 top_k_candidates 中最大 Top-K 的结果以覆盖最多特征）\n"
    "# --------------------------------------------------------------------------\n"
    "if rfecv_results:\n"
    "    best_topk = max(rfecv_results.keys())\n"
    "    rfecv_summary = rfecv_results[best_topk].summary.copy()\n"
    "    rfecv_feat_col = next((c for c in ['feature', 'feature_name'] if c in rfecv_summary.columns), None)\n"
    "    rfecv_rank_col = next((c for c in ['rfecv_ranking', 'ranking', 'rank'] if c in rfecv_summary.columns), None)\n"
    "    if rfecv_feat_col and rfecv_rank_col:\n"
    "        rfecv_info = rfecv_summary[[rfecv_feat_col, rfecv_rank_col]].rename(\n"
    "            columns={rfecv_feat_col: 'feature', rfecv_rank_col: 'rfecv_ranking'}\n"
    "        )\n"
    "    else:\n"
    "        print('rfecv_summary columns:', list(rfecv_summary.columns))\n"
    "        rfecv_info = pd.DataFrame(columns=['feature', 'rfecv_ranking'])\n"
    "else:\n"
    "    rfecv_info = pd.DataFrame(columns=['feature', 'rfecv_ranking'])\n"
    "\n"
    "# --------------------------------------------------------------------------\n"
    "# Step 3: 合并，归一化，计算综合重要性（越小 = 越不重要）\n"
    "# --------------------------------------------------------------------------\n"
    "merged = stage_a_info.merge(rfecv_info, on='feature', how='left')\n"
    "\n"
    "def minmax_norm(s):\n"
    "    lo, hi = s.min(), s.max()\n"
    "    return (s - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=s.index)\n"
    "\n"
    "merged['stage_a_score_norm'] = minmax_norm(merged['stage_a_score'])\n"
    "if merged['rfecv_ranking'].notna().any():\n"
    "    merged['rfecv_rank_norm'] = 1.0 - minmax_norm(\n"
    "        merged['rfecv_ranking'].fillna(merged['rfecv_ranking'].max())\n"
    "    )\n"
    "    merged['combined_importance'] = 0.5 * merged['stage_a_score_norm'] + 0.5 * merged['rfecv_rank_norm']\n"
    "else:\n"
    "    merged['rfecv_rank_norm'] = float('nan')\n"
    "    merged['combined_importance'] = merged['stage_a_score_norm']\n"
    "\n"
    "least_important_df = merged.sort_values('combined_importance', ascending=True).reset_index(drop=True)\n"
    "least_important_df.index = least_important_df.index + 1\n"
    "least_important_df.index.name = 'importance_rank'\n"
    "\n"
    "print(f'全量特征数（Total features）: {len(least_important_df)}')\n"
    "print()\n"
    "\n"
    "# --------------------------------------------------------------------------\n"
    "# Step 4: 按阈值输出最不重要特征组\n"
    "# --------------------------------------------------------------------------\n"
    "thresholds = [10, 50, 100, 200]\n"
    "least_important_groups = {}\n"
    "for n in thresholds:\n"
    "    group = least_important_df.head(n)['feature'].tolist()\n"
    "    least_important_groups[n] = group\n"
    "    print(f'=== 最不重要 Top-{n} 特征（Least Important Top-{n}）===')\n"
    "    for rank, feat in enumerate(group, start=1):\n"
    "        print(f'  {rank:3d}. {feat}')\n"
    "    print()\n"
    "\n"
    "display(least_important_df[['feature', 'stage_a_score', 'rfecv_ranking', 'combined_importance']].head(30))\n"
)

viz_source = (
    "# 目的（Purpose）: 可视化最不重要特征的综合重要性分布及各阈值边界。\n"
    "# 输入（Inputs）: `least_important_df`、`thresholds`\n"
    "# 输出（Outputs）: 两张图：(1) 综合重要性曲线 + 阈值竖线；(2) 最不重要 Top-20 横条形图。\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
    "\n"
    "ax = axes[0]\n"
    "ax.plot(\n"
    "    range(1, len(least_important_df) + 1),\n"
    "    least_important_df['combined_importance'].values,\n"
    "    color='#4472c4', linewidth=1.2, label='Combined Importance',\n"
    ")\n"
    "colors_v = ['#ed7d31', '#a9d18e', '#ffc000', '#ff0000']\n"
    "for n, cv in zip(thresholds, colors_v):\n"
    "    if n <= len(least_important_df):\n"
    "        ax.axvline(x=n, color=cv, linestyle='--', linewidth=1, label=f'Top-{n}')\n"
    "ax.set_xlabel('Importance Rank（升序，rank 1 = 最不重要）')\n"
    "ax.set_ylabel('Combined Importance Score')\n"
    "ax.set_title('Feature Importance Distribution（Ascending）')\n"
    "ax.legend(fontsize=8)\n"
    "\n"
    "ax2 = axes[1]\n"
    "show_n = min(20, len(least_important_df))\n"
    "top20 = least_important_df.head(show_n)[['feature', 'combined_importance']]\n"
    "ax2.barh(top20['feature'][::-1], top20['combined_importance'][::-1], color='#ed7d31')\n"
    "ax2.set_xlabel('Combined Importance Score（越小越不重要）')\n"
    "ax2.set_title(f'Least Important Top-{show_n} Features')\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
)

new_cells = [
    {
        "cell_type": "markdown",
        "id": "sec25_least_important_md",
        "metadata": {},
        "source": markdown_source,
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "sec25_least_important_code",
        "metadata": {},
        "outputs": [],
        "source": code_source,
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "sec25_least_important_viz",
        "metadata": {},
        "outputs": [],
        "source": viz_source,
    },
]

nb['cells'] = nb['cells'][:INSERT_AFTER + 1] + new_cells + nb['cells'][INSERT_AFTER + 1:]

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done. Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells'][26:29], start=26):
    print(f"  Cell {i}: id={c.get('id')} [{c['cell_type']}]")
