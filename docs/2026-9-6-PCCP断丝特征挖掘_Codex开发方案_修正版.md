# PCCP断丝特征挖掘与筛选系统开发方案（Codex版）

## 1. 项目目标

当前阶段的任务不是训练最终断丝分类模型，而是从数百个人工提取特征中挖掘能够表征PCCP断丝响应的核心特征组合。

目标：

1.  找出区分断丝（BK）、水流噪声（FL）、敲击干扰（QJ）的关键特征；
2.  分析特征对流速变化的敏感性；
3.  去除高度冗余特征；
4.  输出后续分类模型使用的少量稳定特征集合。

------------------------------------------------------------------------

## 2. 数据特点与不均衡问题重新定义

当前数据：

  类别     样本数   source_files
  ------ -------- --------------
  BK00        102             17
  BK05         60             10
  FL00      17066             87
  FL05      20000            100
  QJ00        648            108
  QJ05        360             60

数据特点：

FL窗口为独立水流背景样本，因此不能简单删除大量FL数据。

BK/QJ窗口来自少量物理事件，因此需要考虑事件相关性。

当前问题不是"水流窗口重复"，而是：

-   背景样本数量远大于断丝样本；
-   断丝物理事件数量有限；
-   统计方法容易被多数类影响。

------------------------------------------------------------------------

## 3. 类别不均衡处理原则

本阶段不是分类训练，因此不采用：

-   SMOTE；
-   大量复制BK样本；
-   class weight优化分类器。

采用：

**Bootstrap平衡统计评价方法。**

每次特征评价时：

1.  保留全部BK样本；
2.  从FL、QJ中随机抽取与BK规模相近的样本；
3.  计算特征判别能力；
4.  重复100\~1000次。

最终得到：

-   平均重要性；
-   标准差；
-   置信区间；
-   Top-K出现概率。

------------------------------------------------------------------------

## 4. 数据管理原则

BK/QJ：

同一个物理事件产生的多个窗口必须作为一个整体处理。

不能让同一个断丝事件的窗口同时进入不同验证集合。

FL：

由于窗口独立，可以按照窗口进行随机采样。

------------------------------------------------------------------------

# Notebook开发框架

## Notebook01 数据读取与质量检查

内容：

-   六类数据读取；
-   字段统一；
-   feature列识别；
-   缺失值检查；
-   行级有效特征数检查；
-   异常值检查；
-   特征尺度统计。

补充规则：

-   文件级检查使用 `min_feature_csv_columns`，默认 100。列数低于阈值的 CSV 视为异常分片，不进入后续分析。
-   行级检查使用 `min_valid_feature_values_per_row`，默认 100。若某个样本行虽然所在 CSV 表头完整，但有效频带特征值过少，则该行会被写入读取审计并从后续分析剔除。
-   `load_report.csv` 新增 `retained_rows`、`dropped_low_valid_feature_rows`、`feature_columns_detected`、`min/median/max_valid_feature_values`，用于回答“哪些文件有大量缺失特征行、后续是否继续使用”。
-   示例 `features_20260905_173711_part_0001.csv` 属于表头完整但行级特征缺失严重的情况：旧版只查列数和空表，因此没有检测出来；新版按行检查后会剔除低有效特征行，缺失特征样本不再进入 Notebook02-Notebook09。

输出：

-   dataset_summary.csv
-   feature_list.csv
-   quality_report.csv
-   load_report.csv

------------------------------------------------------------------------

## Notebook02 六类特征分布分析

分析：

-   BK00
-   BK05
-   FL00
-   FL05
-   QJ00
-   QJ05

方法：

-   箱线图；
-   KDE分布；
-   PCA；
-   UMAP。

说明：

-   “六类特征分布”指在 `BK00/BK05/FL00/FL05/QJ00/QJ05` 六个来源标签上观察特征分布，不是指只有六个特征。
-   CSV 中有数百个候选特征，本节先对全部特征计算 `six_class_f_ratio = 类间均值方差 / 类内方差均值`，再选择差异最大的前 `distribution_top_n` 个特征用于图示。
-   所有含 `feature`、`selected_features`、`recommended_features` 等字段的表格，都追加中文含义列，便于人工审查特征物理意义。

------------------------------------------------------------------------

## Notebook03 单特征判别能力分析

针对每个特征计算：

### 3.1 BK00 vs NONBK00

同流速零流速下，断丝与 `FL00+QJ00` 比较。

### 3.2 BK00 vs QJ00

零流速下，断丝与敲击干扰比较。

### 3.3 BK05 vs QJ05

v0.5 流速下，断丝与敲击干扰比较。

### 3.4 BK05 vs NONBK05

v0.5 流速下，断丝与 `FL05+QJ05` 比较。

### 3.5 BK vs NONBK

合并两种流速后，全部断丝与全部非断丝比较，作为最终排序主任务。

指标：

-   Wasserstein distance；
-   Cliff delta；
-   AUC；
-   Mutual Information。

### BK vs QJ

分析断丝特异性。

### BK00 vs BK05

分析流速敏感性。

输出：

feature_discrimination.csv

------------------------------------------------------------------------

## Notebook04 特征冗余分析

方法：

### 4.1 使用 Pearson

Pearson 相关偏重线性关系，适合发现近似成比例变化的重复特征。

### 4.2 使用 Spearman

Spearman 相关偏重单调关系，先转为秩再计算相关，更适合尺度不同或非线性单调变化的特征。本流程后续冗余惩罚和 mRMR 默认使用 Spearman。

### 4.3 对比两种方法结果

合并 Pearson 和 Spearman 命中的高相关特征对，构建冗余组：

-   2 个特征高度相关时，推荐保留其中 `BK vs NONBK` 判别分最高的 1 个；
-   3 个或更多特征形成同一高相关连通组时，同样推荐保留 1 个代表，其余列入冗余删除候选；
-   输出 `correlation_method_comparison.csv` 和 `redundancy_recommendations.csv`，后者包含 `features_in_group`、`recommended_keep`、`recommended_drop`、Pearson/Spearman 高相关对数和中文含义。

输出：

-   correlation_cluster.csv
-   pearson_high_correlation_pairs.csv
-   spearman_high_correlation_pairs.csv
-   correlation_method_comparison.csv
-   redundancy_recommendations.csv
-   mrmr_rank.csv

------------------------------------------------------------------------

## Notebook05 多特征组合搜索

目标：

寻找最佳特征组合。

分节方法：

### 5.1 mRMR前缀组合

按 `mrmr_score = relevance_score - redundancy_weight * mean_abs_corr_to_selected` 搜索低冗余高判别候选，并评估前 K 个特征组合。

### 5.2 近似 ReliefF 排序

通过最近同类和最近异类比较，奖励能拉近同类、拉远异类的特征。

### 5.3 顺序前向选择 SFS

每轮加入使 LDA 交叉验证 AUC 提升最大的特征。

### 5.4 多方法结果对比

统一比较 mRMR 前缀组合和 SFS 路径，关注 AUC 高、LDA separation 高且特征数较少的组合。

评价：

-   类间距离；
-   LDA separation；
-   AUC。

------------------------------------------------------------------------

## Notebook06 特征稳定性分析

采用Bootstrap：

每次重新抽样并重新排序。默认 `bootstrap_rounds=200`，每轮保留全部 BK 样本，再从 Other 中抽取与 BK 行数相同的样本，样本量约为 `2 * BK行数`。Top-K 频率默认统计 `(10, 20, 30)`。

原理：

-   Bootstrap 通过重复抽样估计统计排名对样本扰动的敏感性；
-   若一个特征在多轮平衡抽样中稳定进入 Top-K，说明它不是由某一次多数类样本组成偶然造成；
-   `mean_rank` 越小越好，`rank_std` 越小越稳定，`top10/top20/top30_frequency` 越高越稳定。

参数调整：

-   快速调试可设置 `bootstrap_rounds=20-50`；
-   正式报告建议 `bootstrap_rounds=200-1000`；
-   若需要更宽候选池，可把 `bootstrap_top_ks` 扩展为 `(10, 20, 30, 50)`。

输出：

feature_stability.csv

包含：

-   平均排名；
-   排名标准差；
-   Top10频率；
-   Top20频率。

------------------------------------------------------------------------

## Notebook07 跨流速鲁棒性分析

比较：

BK00 vs FL00

BK05 vs FL05

寻找：

在不同流速下保持稳定的特征。

指标公式：

-   `auc_lift = max(AUC, 1-AUC) - 0.5`；
-   `cross_condition_consistency = min(lift00, lift05) / max(lift00, lift05)`；
-   `median_shift_norm = |median(a)-median(b)| / scale`，`scale` 优先取两组数据合并后的 IQR；
-   `cross_flow_score = direction_same * (0.55*(lift00+lift05)+0.45*consistency) / (1+0.5*bk_shift+0.25*fl_shift)`。

表格列意义：

-   AUC 列衡量两个流速下断丝与流噪的判别强度；
-   `direction_same` 检查特征方向是否跨流速一致；
-   `bk00_bk05_median_shift_norm` 和 `fl00_fl05_median_shift_norm` 衡量特征本身是否受流速影响；
-   `cross_flow_score` 用于排序，所以表格优先列出该分数最高的特征。

------------------------------------------------------------------------

## Notebook08 最终特征评价

融合：

-   判别能力；
-   流速敏感性；
-   冗余；
-   Bootstrap稳定性；
-   跨流速一致性。

输出：

final_feature_ranking.csv

------------------------------------------------------------------------

# Python代码结构

    PCCP_feature_selection/

    ├── run_all.py
    ├── config.py

    ├── src/
    │   ├── data_loader.py
    │   ├── feature_schema.py
    │   ├── event_parser.py
    │   ├── quality_control.py
    │   ├── distribution_analysis.py
    │   ├── feature_discrimination.py
    │   ├── feature_redundancy.py
    │   ├── feature_selection.py
    │   ├── bootstrap_stability.py
    │   ├── cross_flow_analysis.py
    │   ├── visualization.py
    │   └── report_generator.py

------------------------------------------------------------------------

# 最终特征评价

每个特征计算：

-   BK_NONBK判别能力；
-   BK_QJ判别能力；
-   流速敏感性；
-   冗余程度；
-   稳定性；
-   跨流速一致性。

综合形成：

    Final Score =
    (断丝判别能力 + 敲击区分能力)
    /
    (1 + 流速敏感性)

------------------------------------------------------------------------

# 最终输出

必须生成：

1.  feature_discrimination.csv
2.  feature_stability.csv
3.  cross_flow_feature.csv
4.  final_feature_ranking.csv

最终得到：

-   A级：稳定断丝核心特征；
-   B级：候选特征；
-   C级：工况相关特征。

后续分类阶段再使用筛选后的几十个特征建立分类模型。
