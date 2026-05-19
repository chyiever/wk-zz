# Wrapper Selection（包装式特征选择）与 RFECV 原理报告（中文）

> 面向：notebooks/2026-05-06_cross_condition_experiment.ipynb 的 **单元格17（Wrapper/RFECV）**。
>
> 结论先行：本项目的 Wrapper Selection 使用 `sklearn.feature_selection.RFECV` 实现；
> - **线性模型（estimator）**：`LinearSVC`（线性 SVM 分类器）
> - **交叉验证（CV）**：`StratifiedKFold`（分层 K 折，shuffle=True）
> - **模型性能评价（scoring）**：默认 `f1`（可通过参数 `scoring` 改）

---

## 1. 单元格17：Wrapper 方法执行过程中，“模型性能评价函数”是什么？

在本仓库的 Stage-B Wrapper（RFECV）实现中，评价方式来自 `sklearn` 的 `RFECV(..., scoring=...)`。

- 对应代码位置：src/fea_sel_clasf/feature_wrapper.py 中的 `run_rfecv_selection(..., scoring: str = "f1")`
- 默认 `scoring="f1"`，表示 RFECV 在每一个候选特征子集规模上，使用 **F1 分数** 来衡量模型在交叉验证中的平均表现。

因此：
- **Wrapper/RFECV 阶段的“模型性能评价函数”= F1（默认）**
- 与后续“模型搜索/阈值选择/最终评估”阶段（可能更关注 recall_pos、balanced_accuracy）是两套不同目的的评价。

> 说明：`sklearn` 的 `scoring="f1"` 默认是二分类 F1（以标签 `1` 作为正类）。

---

## 2. 最佳特征组合时，模型的分类效果如何？

### 2.1 在 RFECV 语境下，“最佳特征组合”是什么意思？

RFECV 的目标是：在一系列特征子集大小 $m$（例如从 $p$ 个特征逐步删到最少特征数）中，找到使交叉验证平均得分最大的子集大小：

$$
\hat{m} = \arg\max_m \; \frac{1}{K}\sum_{k=1}^K S\big( f^{(k)}_m \big)
$$

- $K$：交叉验证折数（本项目用分层 K 折）
- $f^{(k)}_m$：在第 $k$ 折训练得到、使用 $m$ 个特征的模型
- $S(\cdot)$：评分函数（本项目默认是 F1）

因此：
- **“最佳特征组合”= RFECV 最终 `support_==True` 的那组特征**
- **“分类效果如何”= 该子集规模下的交叉验证平均 F1（默认）**

### 2.2 为什么你在单元格17看不到“最佳 F1 数值”？

当前 `run_rfecv_selection` 仅返回：
- `selected_features`（最终入选特征名列表）
- `summary`（每个特征的 support/ranking）

它没有把 `RFECV` 内部的 `cv_results_`（包含每个特征数对应的平均 CV 得分）一并返回。

### 2.3 想量化“分类效果”（F1 等），应如何做？（建议写法）

你可以在 notebook 里对 RFECV 选出的特征，额外做一次交叉验证评估（不改变既有逻辑，仅用于报告）：

1) 复用 RFECV 的基学习器（LinearSVC + 预处理）并对 `rfecv.selected_features` 做 CV：

```python
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score

# 取训练数据
frame = loaded_table.frame.loc[base_train_mask].copy()
y = frame[loaded_table.label_column].to_numpy(dtype=int)
X = frame.loc[:, list(rfecv.selected_features)].apply(pd.to_numeric, errors='coerce')

# 复用 run_rfecv_selection 里同款 estimator（imputer+scaler+LinearSVC）
# 注意：若你不想复制代码，可在 src/fea_sel_clasf/feature_wrapper.py 里抽一个 builder 返回 estimator。

scores = cross_validate(
    estimator=estimator,
    X=X,
    y=y,
    cv=cv,
    scoring={
        'f1': 'f1',
        'recall_pos': make_scorer(recall_score, zero_division=0),
        'precision_pos': make_scorer(precision_score, zero_division=0),
    },
    n_jobs=1,
)
print({k: float(scores[k].mean()) for k in scores if k.startswith('test_')})
```

2) 或者：进入下一阶段 `search_best_model(...)`（项目既有流程），它会输出更全面的 CV 指标（recall_pos、balanced_accuracy、f1_pos 等），用于最终模型选择。

---

## 3. Wrapper Selection（包装式特征选择）原理

### 3.1 定义

包装式特征选择（Wrapper Feature Selection）把“特征子集好不好”这件事，交给一个可训练的模型去回答。

- 过滤式（Filter）：不训练模型或训练很轻量的统计量（如 MI、相关系数）
- 包装式（Wrapper）：**每次选择/比较特征子集都要训练模型**，用模型表现来决定取舍

其一般形式是：

$$
\hat{\mathcal{F}} = \arg\max_{\mathcal{F} \subseteq \{1,\dots,p\}}\; \mathrm{Score}(\mathcal{F})
$$

- $\mathcal{F}$：特征索引集合
- $\mathrm{Score}(\mathcal{F})$：用特征子集 $\mathcal{F}$ 训练模型并评估得到的分数（通常来自交叉验证）

### 3.2 优缺点

- 优点：直接围绕最终任务（分类/回归）的性能优化，更贴近业务目标
- 缺点：计算开销更大；若评估方式不严谨，容易过拟合

因此：Wrapper 通常要配合 **CV（交叉验证）** 才能把过拟合风险控制住。

---

## 4. 交叉验证（Cross Validation, CV）原理

### 4.1 为什么需要 CV

如果你只在同一份训练数据上评估，会高估模型表现。
CV 的思想是：把训练集拆成多个“训练/验证”子任务，反复训练与评估，用平均表现估计模型的泛化能力。

### 4.2 K 折交叉验证（K-Fold CV）

把训练集划分为 $K$ 份（folds）。第 $k$ 次：
- 用第 $k$ 份作为验证集
- 其余 $K-1$ 份作为训练集

CV 得分估计：

$$
\widehat{S}_{CV} = \frac{1}{K}\sum_{k=1}^{K} S_k
$$

其中 $S_k$ 是第 $k$ 折的验证得分。

### 4.3 分层 K 折（StratifiedKFold）

二分类里如果正样本很少，普通 K 折可能出现某些折“没有正样本”，导致指标（如 F1、Recall）不可用或不稳定。

分层 K 折会尽量让每一折的类别比例与整体接近。

### 4.4 本项目的 CV 策略（非常重要）

本项目在 `run_rfecv_selection` 内部调用 `_build_cv(y, n_splits=5)`，并做了一个保护：

- 先统计各类样本数：`counts = np.bincount(y)`
- 令 `min_class = 最少的非零类别样本数`
- 折数取：

$$
K = \max\big(2, \min(5, \mathrm{min\_class})\big)
$$

含义：
- 最少 2 折
- 默认最多 5 折
- 但不会超过最小类样本数（避免某折拿不到该类样本）

---

## 5. RFE（Recursive Feature Elimination，递归特征消除）原理

### 5.1 RFE 做什么

RFE 是一种典型的 Wrapper 子类：
1) 先训练一个模型
2) 根据模型给出的“特征重要性”对特征排序
3) 删除最不重要的一批特征
4) 重复 1)–3)，直到达到目标特征数

### 5.2 关键：特征重要性从哪里来？

RFE 需要一个能提供“特征权重/重要性”的模型。
- 线性模型：权重向量 $w$（系数 `coef_`）
- 树模型：分裂增益、Gini 重要性等

以线性模型为例，决策函数：

$$
 f(x) = w^\top x + b
$$

常见的“重要性”度量是 $|w_j|$（第 $j$ 个特征的绝对权重）：
- $|w_j|$ 越大，特征对决策边界影响越强
- 在每次迭代中删掉 $|w_j|$ 最小的一批

> 注意：系数重要性依赖特征尺度，因此 RFE/RFECV 通常会搭配标准化（StandardScaler）。

---

## 6. RFECV（Recursive Feature Elimination with Cross Validation）原理

### 6.1 RFECV 比 RFE 多了什么？

RFE 需要你手工指定“最终保留多少特征”。
RFECV 会：
- 在 RFE 的递归删特征过程中，
- 对每一个候选的特征数量 $m$ 做一次交叉验证评分，
- 自动选择得分最高的那个 $m$。

### 6.2 机制概述

- 外层：RFE 逐步减少特征数
- 内层：对每个特征数对应的子集，做 CV 得到平均分

最终选择：

$$
\hat{m} = \arg\max_m \widehat{S}_{CV}(m)
$$

并输出：
- `support_`：每个特征是否被最终选中
- `ranking_`：特征的排名（1 表示最重要/最终保留）

---

## 7. 本项目实现细节对照（从原理到代码）

对应文件：src/fea_sel_clasf/feature_wrapper.py

### 7.1 线性模型是什么？

RFECV 的 estimator 是一个 `Pipeline`：
1) `SimpleImputer(strategy="median")`：用中位数填补缺失
2) `StandardScaler()`：标准化
3) `LinearSVC(...)`：线性 SVM 分类器

`LinearSVC` 的关键参数：
- `C=1.0`：正则强度（越大越不正则化）
- `class_weight="balanced"`：类别权重按频率反比自动调整
- `dual=False`：更适合样本数 > 特征数的情形（通常更快）

### 7.2 LinearSVC 的数学形式（核心）

以二分类 $y_i \in \{-1, +1\}$ 为例，线性 SVM 的优化目标（示意）是：

$$
\min_{w,b} \; \frac{1}{2}\lVert w \rVert_2^2 + C\sum_{i=1}^{n} \max\big(0, 1 - y_i(w^\top x_i + b)\big)
$$

- 第一项：$\lVert w \rVert_2^2$ 控制模型复杂度（间隔最大化的对偶体现）
- 第二项：合页损失（hinge loss），惩罚分错或间隔不足的样本
- $C$：在“间隔/正则化”与“训练误差”之间权衡

预测时：
- 先计算 $f(x)=w^\top x + b$
- 再用阈值（通常 0）决定类别

### 7.3 RFECV 的 scoring 是什么？

在本项目中：
- `run_rfecv_selection(..., scoring="f1")` 默认用 F1

F1 基于混淆矩阵定义。令：
- TP：预测为 1 且真实为 1
- FP：预测为 1 但真实为 0
- TN：预测为 0 且真实为 0
- FN：预测为 0 但真实为 1

则：

$$
\mathrm{Precision} = \frac{TP}{TP+FP}
$$

$$
\mathrm{Recall} = \frac{TP}{TP+FN}
$$

$$
\mathrm{F1} = \frac{2\cdot \mathrm{Precision}\cdot \mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}
$$

> 解释：F1 同时惩罚“漏检（FN）”与“误报（FP）”，适合类别不平衡且你同时关注精确率与召回率的场景。

---

## 8. 补充：项目后续阶段的“模型性能评价函数”是什么？（避免混淆）

你在 notebook 里还会看到另一个评价函数：src/fea_sel_clasf/evaluator.py 中的 `evaluate_binary_predictions(...)`。

它是在“最终模型 + 阈值”确定后，用测试集概率 `y_prob` + 阈值 `threshold` 生成 `y_pred` 并计算多种指标：
- accuracy
- balanced_accuracy
- recall_pos
- precision_pos
- specificity_neg
- f1_pos
- auroc
- fpr
- 以及 TP/FP/TN/FN

其中 **Balanced Accuracy** 的典型形式是：

$$
\mathrm{BalancedAcc} = \frac{1}{2}(\mathrm{Recall}_{pos} + \mathrm{Specificity}_{neg})
$$

这套指标更偏“最终协议评估（per test domain）”，而 RFECV 的 `scoring` 更偏“特征子集选择（wrapper selection）”。

---

## 9. 单元格17输出该如何解读？（对应你看到的打印）

在单元格17里每个 `top_k` 你会看到三类信息：

1) `top_k=60: candidate_features=60`
- 含义：Stage-A 从全量特征排序后截取了 Top-60 作为 RFECV 的候选输入

2) `rfecv selected_features=5`
- 含义：RFECV 最终选择了 5 个特征
- 注意：`run_rfecv_selection` 默认 `min_features_to_select=5`，因此在很多情况下你会看到最终数量“稳定为 5”

3) `rfecv selected_feature_names=[...]`
- 含义：最终入选的 5 个特征名

如果你要回答“最佳特征组合时分类效果如何”：
- 在 RFECV 语境下，它对应的是“选出这 5 个特征时的 CV 平均 F1（默认）”
- 若你要在报告中给出明确数值，建议按第 2.3 节的方法追加一次 CV 评估，把 mean/std 的 F1、Recall、Precision 打出来。
