# 项目工作日志

## 2026-05-19

- 目标：将本项目指定目录上传至 `https://github.com/chyiever/wk-zz.git`。
- 范围：`notebooks/`、`docs/`、`src/`、`tools/`。
- 执行事项：
  - 在 `docs/` 下新增本日志文件 `dev.md`，用于记录本次上传操作；
  - 初始化本地 Git 仓库并配置远程地址；
  - 仅将上述目录纳入版本管理并提交；
  - 推送到远程仓库对应分支。
- 结果：已完成本地准备，待推送结果以终端执行状态为准。

## 2026-05-21

- 本次更新范围：`notebooks/2026-05-06_cross_condition_experiment.ipynb`、`notebooks/2026-05-18-das-loc-notebook1-dev.ipynb`、`docs/dev.md`。
- 程序更新日志：
  - `2026-05-06_cross_condition_experiment.ipynb`：补充实验组特征打印与对比展示，新增 3.4 特征分布按标签统计与可视化，更新部分单元格执行结果。
  - `2026-05-18-das-loc-notebook1-dev.ipynb`：调整图表配色、图例与刻度展示，清理部分输出内容。
- GitHub 上传日志：
  - 已提交到本地 `master` 分支。
  - 已推送到 `origin/master`。

## 2026-05-25

- 本次更新范围：`tools/build_realdata_prediction_notebook.py`、`notebooks/2026-05-25-realdata_label_prediction_cross_condition.ipynb`、`docs/dev.md`。
- 程序更新日志：
  - 预测 notebook 新增“训练阈值优先”机制：增加 `USE_TRAINED_THRESHOLD` 开关，默认 `True`。
  - 对每个实验组（`exp_*`）按当前 `MODEL_TYPE` 自动读取训练阶段保存阈值：
    - 优先从 `summary.csv` 的 `model_name + threshold` 读取；
    - 若缺失则回退读取 `metrics_<model_type>_*.json` 的 `threshold`。
  - 预测输出新增阈值追踪字段：`threshold_in_use`、`threshold_source`，便于回溯每条预测所用阈值来源（`trained/manual`）。
  - 概率曲线图阈值线改为使用分组实际阈值（而不是固定全局阈值）。
  - 保持特征向量顺序强约束：`x = df.loc[:, selected_features]`，与模型输入顺序一致。
- 自检记录：
  - notebook 由 Python 脚本以 `encoding='utf-8'` 重新生成；
  - 关键配置与函数存在：`USE_TRAINED_THRESHOLD`、`load_trained_threshold`、`threshold_in_use`；
  - 中文文本自检未发现 `'?'` 乱码占位。
- GitHub 上传日志：
  - 已提交到本地 `master` 分支；
  - 已推送到 `origin/master`（以终端 push 结果为准）。

- 补充修复（同日）：
  - 问题：当 `EXPERIMENTS` 配置为单个字符串（如 `exp_07_train_F130_F130A_F130C`）时，旧逻辑会按字符迭代，导致误访问 `...\\e\\selected_features.csv` 并报 `FileNotFoundError`。
  - 修复：
    - 更新 `list_experiments()` 入参兼容：`None / str / list[str] / tuple[str, ...]`；
    - 当传入单字符串时自动包装为单元素列表，不再逐字符迭代；
    - 配置注释补充“单实验可直接写字符串”的示例。
  - 影响文件：`tools/build_realdata_prediction_notebook.py`、`notebooks/2026-05-25-realdata_label_prediction_cross_condition.ipynb`。

## 2026-05-27

- 本次更新范围：`notebooks/2026-05-27-feature_comparison_build3_vs_batch.ipynb`、`tools/create_comparison_notebook.py`、`docs/2026-05-27-五特征计算方法对比报告.md`、`docs/dev.md`。
- 任务背景：
  - 对 `b_1k_10k__SC_mean`、`b_1k_10k__C_f`、`b_1k_100k__epsilon_2x`、`b_1k_100k__SC_res_mean`、`b_1k_100k__I_burst` 五个特征，验证 `2026-05-19-realdata_feature_dataset_build-3.ipynb` 与 `2026-05-06-batch_feature_extract_12folders_gpu.ipynb` 两个 notebook 的计算结果是否完全一致。
- 程序更新日志：
  - 新增对比验证 notebook `2026-05-27-feature_comparison_build3_vs_batch.ipynb`：
    - 使用同一 TDMS 数据文件（`0002341-500K-20260324T201052.395.tdms`，500 kHz，10 s）；
    - 相同预处理（去均值 + (1k, 95k) Hz 4 阶 Butterworth 带通滤波）与滑窗参数（20 ms，50% 重叠，共 999 窗口）；
    - 方法 A：复刻 build-3 的内联自定义函数（`_spectral_centroid_mean`、`_c_f_from_context` 等）；
    - 方法 B：调用 `fea_cpt_gpu.features.compute_all_features` 后取对应键值；
    - 两种方法各自独立构建 `FeatureRecord` 与 `build_context`，逐窗口计算并对比。
  - 新增生成脚本 `tools/create_comparison_notebook.py`，以 `encoding='utf-8'` 写入 notebook，避免终端重定向乱码。
- 分析结论：
  - 两个 notebook 均为**滑窗级别**特征计算（非整段信号），数据处理流程完全一致。
  - 五个特征在 999 个窗口上绝对差均为 0，两种实现方法数学上完全等价。
  - build-3 的自定义函数是对 `compute_all_features` 中对应逻辑的轻量提取，主要差异在工程结构（5特征专用 vs 全量特征集）。
  - 详细对比见 `docs/2026-05-27-五特征计算方法对比报告.md`。
- GitHub 上传日志：
  - 已提交到本地 `master` 分支；
  - 已推送到 `origin/master`（以终端 push 结果为准）。

  - 补充更新（同日）：
    - 更新范围：`notebooks/2026-05-27-feature_comparison_build3_vs_batch.ipynb`、`tools/modify_comparison_notebook.py`、`docs/dev.md`。
    - 为 notebook 各节添加一级标题（H1）与二级标题（H2，共 13 节），提升可读性。
    - 新增第 13 节"特征概率分布对比：方法A（无标签）vs 方法B（按标签分组）"：
      - 方法A：读取 `outputs/realdata_feature_dataset_20260519_v3` + `outputs/realdata_feature_dataset_20260523` 中的特征 CSV，所有样本合并（719550 条），无标签区分，绘制 KDE 概率密度曲线。
      - 方法B：读取 `outputs/dataset_build_260518_features_gpu` 中的特征 CSV，按 `label` 列分组（0=噪声 5293 条，1=断丝 5293 条），分别绘制 KDE 曲线。
      - 每个特征一个子图，横轴为特征值，纵轴为概率密度，共 5 个子图。
      - 图表保存至 `outputs/feature_distribution_comparison_A_vs_B.png`。
    - 修改脚本 `tools/modify_comparison_notebook.py`：以 `encoding='utf-8'` 读写 `.ipynb`，插入 H2 标题单元格并追加分布对比代码单元格。
    - 自检记录：
      - notebook 由 Python 脚本以 `encoding='utf-8'` 读写，未使用终端重定向；
      - 中文文本自检未发现 `'?'` 乱码占位；
      - 分布对比代码独立验证通过，图片生成正常。

## 2026-05-28

- 本次更新范围：`notebooks/2026-05-27-feature_comparison_build3_vs_batch.ipynb`、
  `docs/2026-05-27-五特征计算方法对比报告.md`、`tools/update_dist_comparison.py`、`docs/dev.md`。
- 程序更新日志：
  - 将第 13 节重命名为"真实数据 vs 训练集+测试集 特征分布对比"，更新所有标签和描述：
    - 方法A 改称"真实数据"，方法B 改称"训练集+测试集"。
    - 图例更新为"真实数据"、"训练集+测试集: 噪声"、"训练集+测试集: 断丝"。
  - 新增第 13.4 节"定量统计指标"：对每个特征打印均值、标准差、偏度、峰度、P25/P50/P75 分位数，
    以及三组 KS 检验（真实 vs 噪声、真实 vs 断丝、噪声 vs 断丝）。
  - 更新对比报告 `docs/2026-05-27-五特征计算方法对比报告.md`：
    - 新增第 7 节"真实数据 vs 训练集+测试集 特征分布对比"，包含数据集对比表、
      5个特征的定量统计汇总表、分布差异原因分析（5条）、下一步建议（5条）。
    - 原第 6 节"结论"重编号为第 8 节，新增第 5 条结论总结分布对比结果。
  - 定量统计结果验证通过：
    - 真实数据 719,550 条，训练集+测试集 10,586 条（噪声/断丝各 5,293）。
    - 所有特征的 KS 检验均显著（p<0.001），分布差异极大。
    - I_burst 偏度 12.78、峰度 276.59，呈现极端右偏长尾分布。
    - C_f 和 I_burst 在噪声与断丝两组间 KS>0.95，区分能力最强。
- 自检记录：
  - notebook 由 Python 脚本以 `encoding='utf-8'` 读写，未使用终端重定向；
  - 中文文本自检未发现 `'?'` 乱码占位；
  - 定量统计代码独立验证通过，输出正确。
- GitHub 上传日志：
  - 已提交到本地 `master` 分支；
  - 已推送到 `origin/master`（以终端 push 结果为准）。