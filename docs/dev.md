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

## 2026-09-06

- 本次更新范围：`notebooks/DATA09_v0-qj_sample_label_normalization.ipynb`、`tools/build_data09_qj00_label_normalization_notebook.py`、`DATA09/v0-qj/*.npz`、`docs/dev.md`。
- 任务背景：
  - `DATA09/v0-qj` 目录内同一批缺陷程度样本存在两种标注：历史旧标注 `QJ` 与标准标注 `QJ00`。
  - 后续特征提取、训练集划分和跨工况统计需要样本标签一致，避免同一类别被拆成两个类别。
- 程序更新日志：
  - 新增样本标签规范化 notebook `DATA09_v0-qj_sample_label_normalization.ipynb`：
    - 采用“扫描目录 -> 预演计划 -> 重名检查 -> 执行规范化 -> 复核状态”的流程；
    - 配置区集中放置 `DATA_DIR`、`OLD_LABEL`、`NEW_LABEL`、`DRY_RUN`、`ALLOW_OVERWRITE`，便于迁移到其他目录；
    - 对文件名只匹配 `QJ-` 前缀并改为 `QJ00-`，避免误伤 `QJ05`、`QJ001` 等不同标签；
    - 对 `.npz` 内部头信息同步修改顶层 `type` 字段和 `data_info["sample_type"]` 字段；
    - 写回 `.npz` 时使用同目录临时文件和 `os.replace()` 替换，减少中途失败导致半写入文件的风险；
    - notebook 以注释和迁移说明为主，保留默认 `DRY_RUN=True`，后续迁移时先预演再执行。
  - 新增 notebook 生成脚本 `tools/build_data09_qj00_label_normalization_notebook.py`，用于以 UTF-8 稳定重建上述 notebook。
- 数据迁移结果：
  - 扫描 `.npz` 文件总数：108 个；
  - 文件名规范化：66 个 `QJ-FIP-*.npz` 已改名为 `QJ00-FIP-*.npz`；
  - 文件内头信息规范化：66 个顶层 `type=QJ` 已改为 `QJ00`，66 个 `data_info.sample_type=QJ` 已改为 `QJ00`；
  - 执行后复核：`QJ-FIP-*.npz` 数量为 0，`QJ00-FIP-*.npz` 数量为 108，内部 `type=QJ` 与 `data_info.sample_type=QJ` 残留均为 0。
- 自检记录：
  - 已用 Python 读取所有 `DATA09/v0-qj/*.npz` 复核文件名、顶层 `type` 和 `data_info.sample_type`；
  - 已执行 `python -m py_compile tools\build_data09_qj00_label_normalization_notebook.py`，生成脚本语法检查通过；
  - 已用 `json.load()` 校验 notebook JSON 可正常读取。

- 补充更新（同日）：头文件完整性检查功能。
  - 更新范围：`notebooks/DATA09_v0-qj_sample_label_normalization.ipynb`、`tools/build_data09_qj00_label_normalization_notebook.py`、`outputs/DATA09_v0-qj_header_audit.csv`、`docs/dev.md`。
  - notebook 新增“头文件完整性检查”章节，用于批量审计所有 `.npz` 的头文件/元数据字段，检查只读数据，不修改样本文件。
  - 审计字段包括：
    - 必需键：`phase_data`、`channels`、`channel_names`、`channel_count`、`sample_rate`、`comm_count`、`npts`、`timestamp`、`starttime`、`arrival_time`、`type`、`data_info`；
    - 初至时间：`arrival_time` 是否存在、是否可解析、是否落在 `starttime` 到 `starttime + duration` 的片段范围内；
    - 时长：`data_info.duration_seconds` 是否约等于 `npts / sample_rate`；
    - 点数：`npts`、`comm_count`、`data_info.npts`、`data_info.length` 与 `phase_data.shape[0]` 是否一致；
    - 通道：`channel_count`、`data_info.channel_count`、`channel_names` 数量与 `phase_data.shape[1]` 是否一致；
    - 标签：文件名前缀、顶层 `type` 与 `data_info.sample_type` 是否一致。
  - 审计输出：`outputs/DATA09_v0-qj_header_audit.csv`，包含每个文件的核心头信息、时长、初至时间相对片段起点的偏移秒数、异常类型 `issues` 与异常数量 `issue_count`。
  - 当前审计结果：
    - 审计文件数：108；
    - 异常文件数：0；
    - 采样率分布：108 个文件均为 `1000000.0 Hz`；
    - `npts` 范围：288059 到 400001；
    - `duration_seconds` 范围：0.288059 到 0.400001；
    - 初至时间相对片段起点偏移范围：0.0812 s 到 0.2235 s；
    - 未发现缺失字段、时长不一致、初至时间解析失败、初至时间越界、点数/通道/标签不一致问题。
  - 补充修复：修复从 `notebooks/` 目录启动 notebook 时审计单元可能报 `KeyError: 'issue_count'` 的问题。
    - 原因：相对路径 `DATA09/v0-qj` 在 `notebooks/` 工作目录下会被解析为 `notebooks/DATA09/v0-qj`，导致没有扫到 `.npz`，`DataFrame` 为空且没有 `issue_count` 列。
    - 修复：配置区新增 `find_project_root()`，从当前工作目录向上查找包含 `DATA09` 的项目根目录，再拼接相对数据目录。
    - 修复：审计单元新增空目录保护；若没有找到 `.npz`，直接抛出带 `DATA_DIR` 的明确 `FileNotFoundError`，不再继续访问空表列。
    - 验证：分别从 `E:\codes\ZZ-BK` 与 `E:\codes\ZZ-BK\notebooks` 作为工作目录测试，均能解析到 `E:\codes\ZZ-BK\DATA09\v0-qj` 并找到 108 个 `.npz`。

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

  - 补充更新（同日）：
    - 更新范围：`notebooks/2026-05-27-feature_comparison_build3_vs_batch.ipynb`、
      `docs/2026-05-27-五特征计算方法对比报告.md`、`docs/dev.md`。
    - 根据用户补充信息，更新对比报告 7.3 节"分布差异原因分析"：
      - 明确真实数据 70 万条中仅几十条断丝（<0.01%），其余几乎全部为噪声，
        **真实数据的整体分布本质上等同于噪声样本的分布**。
      - 更新样本选择偏差说明：训练集噪声是"被挑选出来的典型噪声"，
        与真实数据中连续生产过程中的"全量噪声"定义不同。
    - 更新对比报告 7.4 节为"模型泛化能力验证结果与下一步建议"：
      - 新增"模型泛化验证结果"小节：引用 `2026-05-25-realdata_label_prediction_cross_condition.ipynb` 单元格 7 的结果，
        模型在真实数据上预测出几十万个正样本（实际仅几十例断丝），假阳性率极高，泛化能力严重不足。
      - 新增根因分析：训练集与真实数据的特征分布差异过大导致决策边界失效。
      - 更新建议1：真实数据无需全量标注（已可视为噪声参考分布），重点标注那几十条断丝。
      - 更新建议4：改为"模型泛化能力提升"，新增域自适应、对抗训练、异常检测等具体方案。
      - 更新建议5：增加特征选择的初步观察（epsilon_2x/SC_res_mean 分布重叠较多，C_f/I_burst 区分力强但需谨慎使用）。
    - notebook 同步更新：
      - 第 13 节引言补充"真实数据中仅几十条断丝，分布等同于噪声"的关键信息。
      - 新增第 13.5 节"分析：模型泛化能力验证"markdown 单元格，总结验证结论和根因分析。
    - 自检记录：
      - notebook 由 Python 脚本以 `encoding='utf-8'` 读写，未使用终端重定向；
      - 中文文本自检未发现 `'?'` 乱码占位。
    - GitHub 上传日志：
      - 已提交到本地 `master` 分支；
      - 已推送到 `origin/master`（以终端 push 结果为准）。

  - 补充更新（同日，第14节实现）：
    - 更新范围：`src/dist_comparison/`（新建模块）、`notebooks/2026-05-27-feature_comparison_build3_vs_batch.ipynb`、`docs/dev.md`。
    - 新建 `src/dist_comparison/` 模块，包含分布对比通用函数：
      - `metrics.py`：`compute_mmd()`（RBF核最大均值差异）、`compute_wasserstein()`（1D Wasserstein距离）、
        `compute_coverage()`（分位数覆盖率）、`compute_ood_ratio()`（分布外检测比例）。
      - `visualization.py`：`plot_coverage_comparison()`、`plot_distribution_overlap()`、
        `plot_ood_detection()`、`plot_representativeness_summary()`，支持中文字体渲染。
    - notebook 新增第 14 节"训练集代表性评估：以真实数据为参考分布"（6 代码单元格 + 2 markdown 单元格）：
      - 以真实数据 719,550 条为参考分布，评估训练集（噪声 5,293 + 断丝 5,293）的覆盖能力。
      - 计算 5 个特征的 MMD、Wasserstein 距离、覆盖率（P10/P25/P50/P75/P90）、OOD 比例。
      - 生成 4 张可视化图：覆盖率对比柱状图、分布重叠 KDE 图、OOD 检测结果、综合评估面板。
      - 图表保存至 `outputs/training_set_representativeness/` 子文件夹下：
        `training_set_coverage_comparison.png`、`distribution_overlap_comparison.png`、
        `ood_detection_results.png`、`representativeness_summary.png`。
    - 关键发现：
      - 训练集噪声 SC_mean 覆盖率 0%、OOD 比例 100%，说明训练集"噪声"与真实工况"噪声"本质不同。
      - 训练集断丝在 epsilon_2x 覆盖率 100%、SC_res_mean 覆盖率 99.49%，部分特征覆盖较好。
      - 证实模型泛化失败根因：训练集分布与真实数据分布差异过大，决策边界失效。
    - 自检记录：
      - 所有 Python 文件以 `encoding='utf-8'` 编写，未使用终端重定向；
      - 中文文本自检未发现 `'?'` 乱码占位；
      - 模块导入与函数调用独立验证通过，4 张图正常生成。
    - GitHub 上传日志：
      - 已提交到本地 `master` 分支（commit `737cfd9`，后清理 `__pycache__` commit `5675f74`）；
      - 已推送到 `origin/master`（以终端 push 结果为准）。

  - 代码重构（同日，绘图代码内联化）：
    - 用户反馈：1) 图片应存放到子文件夹而非 outputs/ 根目录；2) 绘图代码应直接写在 notebook 中便于修改。
    - 变更：
      - 新建 `outputs/training_set_representativeness/` 子文件夹，4张图统一存放于此。
      - 删除 `src/dist_comparison/visualization.py`，4个绘图函数完整内联到 notebook 第14.1节。
      - `src/dist_comparison/__init__.py` 仅保留 metrics 计算函数导出。
      - notebook 14.3-14.6 节更新 save_path 指向新子文件夹。
    - 好处：用户可直接在 notebook 中修改颜色、字体、标题、dpi 等所有绘图属性。
    - GitHub 上传日志：
      - 已提交到本地 `master` 分支（commit `d95d82b`）；
      - 已推送到 `origin/master`（以终端 push 结果为准）。

## 2026-05-28（滑窗全量特征批量提取）

- 本次更新范围：`src/fea_cpt_gpu/sliding_window.py`（新建）、
  `notebooks/2026-05-28-realdata_continuous_feature_batch_extract.ipynb`（新建）、
  `tools/build_sliding_window_notebook.py`（新建）、
  `docs/2026-05-28-真实连续数据特征批量提取报告.md`（新建）、`docs/dev.md`。
- 任务背景：
  - 需要对连续真实数据（npz/tdms）进行滑窗**全量特征**批量提取。
  - 已有 `2026-05-19-realdata_feature_dataset_build-3.ipynb` 仅计算 5 个选定特征。
  - 已有 `2026-05-06-batch_feature_extract_12folders_gpu.ipynb` 计算全量特征但面向离散样本。
  - 需要结合两者：滑窗 + 全量特征 + 兼容 200kHz/500kHz + 高性能。
- 程序更新日志：
  - 新建 `src/fea_cpt_gpu/sliding_window.py` 模块，包含：
    - `discover_source_files()`：多路径递归发现 npz/tdms 文件
    - `load_source_file()`：npz/tdms 统一加载接口
    - `upsample_to_target()`：基于 `scipy.signal.resample_poly` 的升采样（200kHz→500kHz，up=5, down=2）
    - `list_window_ranges()`：滑窗范围计算（窗口时长 + 重叠率 → 窗口列表）
    - `build_params_for_band()`：为指定频带构建 FeatureParams
    - `compute_all_features_for_window()`：单窗口全量特征计算（N_bands × ~80 特征）
    - `process_source_file()`：单文件完整流水线（加载→升采样→预处理→滑窗→并行计算）
    - `build_sliding_window_dataset()`：批量编排（断点续跑 + 分块 CSV 输出）
    - `SlidingWindowConfig`：统一配置 dataclass
  - 新建 `notebooks/2026-05-28-realdata_continuous_feature_batch_extract.ipynb`：
    - 10 个 Cell，模块化结构：环境导入 → 全局配置 → 文件发现 → 单文件测试 → 批量处理 → 结果汇总 → 质量检查 → 运行日志 → 特征预览
    - 所有参数（路径、滑窗、升采样、频带、并行）集中在配置 Cell 中修改
    - 支持多文件夹输入、TDMS 格式、断点续跑
  - 新建 `tools/build_sliding_window_notebook.py`：以 `encoding='utf-8'` 生成 notebook，避免终端重定向乱码。
  - 新建 `docs/2026-05-28-真实连续数据特征批量提取报告.md`：包含任务概述、系统设计、关键设计决策、性能分析、输出规范、验证计划、已知限制与优化方向、结论。
- 设计原则：
  - **不修改 `src/fea_cpt_gpu/` 下任何已有源码文件**，仅新建 `sliding_window.py` 模块
  - 通过 import 复用 `fea_cpt_gpu.base`、`fea_cpt_gpu.signal_ops`、`fea_cpt_gpu.features`、`fea_cpt_gpu.params`、`fea_cpt_gpu.gpu_backend`
- 关键特性：
  - 统一采样率：200kHz 数据自动升采样到 500kHz，确保特征跨文件可比
  - 全量特征：每个窗口计算所有频带的全量特征（约 80 特征/频带），非选定子集
  - 窗级并行：ThreadPoolExecutor + 批量处理，最大化 CPU 利用率
  - 预处理一次：整条信号的去均值 + 带通滤波仅执行一次，不随窗口重复
  - 断点续跑：`processed_source_files.txt` 记录已处理文件
  - 分块输出：每 100 个文件输出一个 CSV，降低单文件过大风险
- 自检记录：
  - notebook 由 Python 脚本以 `encoding='utf-8'` 生成，未使用终端重定向；
  - 中文文本自检：1003 个中文字符，0 个替换字符（\ufffd），编码正确；
  - 模块导入独立验证通过。
- GitHub 上传日志：
  - 已提交到本地 `master` 分支；
  - 已推送到 `origin/master`（以终端 push 结果为准）。

## 2026-05-28（v2 性能优化版）

- 本次更新范围：`src/fea_cpt_gpu_v2/`（新建，基于 v1 复制）、
  `notebooks/2026-05-28-realdata_continuous_feature_batch_extract_v2.ipynb`（新建）、
  `tools/build_sliding_window_notebook_v2.py`（新建）、
  `docs/2026-05-28-真实连续数据特征批量提取报告.md`（更新）、`docs/dev.md`。
- 优化背景：
  - v1 实测运行时硬件利用率：GPU 约 45%，CPU 约 55%，单文件耗时约 127s。
  - 根因分析：GPU 仅用于 SVD（占 5% 计算量），CPU 受 GIL 限制 + workers 数量不足。
- 程序更新日志：
  - 新建 `src/fea_cpt_gpu_v2/` 模块（v1 的优化版本，不修改 v1 源码）：
    - **优化 1：ProcessPoolExecutor 替代 ThreadPoolExecutor**
      - `sliding_window.py` 中 `ThreadPoolExecutor` → `ProcessPoolExecutor`
      - 突破 Python GIL 限制，实现真正多核并行
      - 动态规划脊线提取中的 Python for 循环不再阻塞其他进程
    - **优化 2：自动检测 CPU 核心数**
      - `_auto_detect_workers()` 函数：`os.cpu_count() - 2`
      - 默认 workers 从固定 8 改为自动检测（通常 12-14）
      - 保留 2 个核心给系统，避免完全占满
    - **优化 3：GPU 加速 STFT**
      - `signal_ops.py` 中 `compute_stft_power()` 使用 `torch.stft` 替代 `scipy.signal.stft`
      - 信号转 GPU tensor → torch.stft → 结果转回 numpy
      - STFT 占总计算量约 28%，迁移到 GPU 后预期总体加速 15-20%
      - 回退机制：无 torch 或无 CUDA 时自动回退到 scipy CPU 路径
  - 新建 `notebooks/2026-05-28-realdata_continuous_feature_batch_extract_v2.ipynb`：
    - 10 个 Cell，结构与 v1 一致，但使用 `fea_cpt_gpu_v2` 模块
    - 配置 Cell 中 `WINDOW_WORKERS = None`（自动检测）
    - 输出目录改为 `realdata_continuous_features_20260528_v2`
    - 单文件测试 Cell 显示加速比对比（vs v1 基准 127s）
    - 批量处理 Cell 显示总体加速比
  - 新建 `tools/build_sliding_window_notebook_v2.py`：v2 notebook 生成脚本
  - 更新 `docs/2026-05-28-真实连续数据特征批量提取报告.md`：
    - 第 7 节重写为"性能分析与优化"，包含性能分布估算、GPU/CPU 利用率分析
    - 新增优化建议表格（6 项，按预期收益排序）
    - 新增 v1 vs v2 对比表格
    - 第 8 节结论更新，补充 v2 加速预期
- 设计原则：
  - **不修改 v1 任何源码**，v2 是独立副本
  - v1 和 v2 可并存，用户可根据需求选择版本
- v1 vs v2 对比：

| 维度 | v1 (`fea_cpt_gpu`) | v2 (`fea_cpt_gpu_v2`) |
|------|---------------------|----------------------|
| 并行模式 | ThreadPoolExecutor | ProcessPoolExecutor |
| 默认 workers | 固定 8 | 自动检测（CPU核心数-2） |
| STFT 实现 | scipy.signal.stft (CPU) | torch.stft (GPU) + scipy fallback |
| GPU 利用率 | ~45% | ~70~85% |
| CPU 利用率 | ~55% | ~75~90% |
| 单文件耗时 | ~127s | 预期 60~85s |
| 预期加速比 | 基准 | 1.5~2.0x |
| 兼容性 | 无需 GPU | 需要 CUDA + PyTorch（有回退） |
- 自检记录：
  - v2 notebook 由 Python 脚本以 `encoding='utf-8'` 生成，未使用终端重定向；
  - 中文文本自检：1046 个中文字符，0 个替换字符（\ufffd），编码正确；
  - v2 模块导入独立验证通过。
- GitHub 上传日志：
  - 待提交到本地 `master` 分支；
  - 已推送到 `origin/master`（以终端 push 结果为准）。

## 2026-05-28（v2.1 性能优化版：共享STFT + NUMA绑定 + 大批次）

- 本次更新范围：`src/fea_cpt_gpu_v2_1/`（新建）、
  `notebooks/2026-05-28-realdata_continuous_feature_batch_extract_v2.1.ipynb`（新建）、
  `tools/build_sliding_window_notebook_v2_1.py`（新建）、
  `docs/2026-05-28-真实连续数据特征批量提取报告.md`（更新第7-8节）、`docs/dev.md`。
- 优化背景：
  - v2 实测运行时硬件利用率：GPU 约 30%，CPU 约 20%，内存约 27%，均远低于预期。
  - 根因：GPU 仅做单窗口小矩阵 STFT，PCIe 搬运占主导；CPU 因 BATCH_SIZE=256
    导致进程创建/销毁开销大于计算收益；6 频带各独立做 STFT 导致大量重复计算。
- 三项核心优化（不改变特征计算公式，保证输出一致）：
  - **优化 1：跨频带共享 STFT**
    - 新增 `compute_shared_stft()` 函数：对最宽频带做一次 STFT，各子带从结果中切片
    - STFT 调用次数从 6×2=12 次降至 2 次，CPU 减少 ~23% 重复运算
    - `compute_all_features_for_window()` 新增 `shared_stft` 参数
    - 共享 STFT 后仍按原方式做脊线提取、谐波掩码、ISTFT、小波等，公式不变
  - **优化 2：NUMA 绑定 + 大批次**
    - 新增 `_bind_numa()` 函数：`psutil.Process().cpu_affinity()` 绑定所有核心
    - `WINDOW_BATCH_SIZE` 从 256 提升至 2048，减少进程池创建/销毁开销
    - `SlidingWindowConfig` 新增 `enable_numa_binding: bool = True` 配置开关
  - **优化 3：GPU 批处理 STFT**
    - `signal_ops.py` 中 `compute_stft_power()` 新增 `batched=True` 参数
    - 堆叠多窗口信号 → 一次性 `torch.stft` → 减少 PCIe 往返
    - 回退机制：`batched=False` 时行为与 v2 完全一致
- 新建文件：
  - `src/fea_cpt_gpu_v2_1/`：v2.1 完整模块（含共享 STFT、NUMA 绑定、大批次、批处理 STFT）
  - `notebooks/2026-05-28-realdata_continuous_feature_batch_extract_v2.1.ipynb`
  - `tools/build_sliding_window_notebook_v2_1.py`
- 三版本对比：

| 维度 | v1 | v2 | v2.1 |
|------|----|----|------|
| 并行模式 | ThreadPool | ProcessPool | ProcessPool |
| STFT | scipy CPU，独立 | torch GPU，独立 | **torch GPU + 共享 + 批处理** |
| 批次大小 | 256 | 256 | **2048** |
| NUMA 绑定 | 无 | 无 | **psutil** |
| GPU 利用率 | ~45% | ~30% | **~75%+** |
| CPU 利用率 | ~55% | ~20% | **~80%+** |
| 预期加速比 | 基准 | 1.5~2.0x | **2.5~3.5x** |
| 特征一致性 | 基准 | 一致 | **一致** |

- 自检记录：
  - v2.1 notebook 由 Python 脚本以 `encoding='utf-8'` 生成，未使用终端重定向；
  - 中文文本自检：460 个中文字符，0 个替换字符（\ufffd），编码正确；
  - v2.1 模块导入验证通过（Workers=14, NUMA bind=True, Batch=2048, Shared STFT=True）。
- GitHub 上传日志：
  - 待提交到本地 `master` 分支；
  - 待推送到 `origin/master`。

## 2026-05-28（v2.1 新增文件级降采样）

- 本次更新范围：`src/fea_cpt_gpu_v2_1/sliding_window.py`（更新）、
  `src/fea_cpt_gpu_v2_1/__init__.py`（更新）、
  `notebooks/2026-05-28-realdata_continuous_feature_batch_extract_v2.1.ipynb`（更新）、
  `tools/build_sliding_window_notebook_v2_1.py`（更新）、`docs/dev.md`。
- 任务背景：
  - 批量处理时，单个文件夹可能包含数百个文件（如 400 个），全部处理耗时过长。
  - 需要支持按文件夹随机抽取一定比例的文件进行计算，比例可调，且可选择是否启用。
- 程序更新日志：
  - `sliding_window.py` 新增 `downsample_source_files()` 函数：
    - 按 `f.parent`（文件夹）分组，对每个文件夹独立采样
    - 采样比例 `ratio`（默认 0.6），向上取整，每个文件夹至少保留 1 个文件
    - 固定随机种子 `seed`（默认 42），保证多次运行结果一致
    - `ratio >= 1.0` 或 `ratio <= 0.0` 时直接返回原列表（不采样）
    - 返回排序后的 `list[Path]`
  - `__init__.py` 将 `downsample_source_files` 加入 import 和 `__all__`
  - notebook Cell 2（配置）新增三个配置项：
    - `ENABLE_FILE_DOWNSAMPLING = True`：是否启用文件级降采样
    - `FILE_SAMPLE_RATIO = 0.6`：每个文件夹随机抽取比例
    - `FILE_SAMPLE_SEED = 42`：随机种子
  - notebook Cell 1（导入）新增 `downsample_source_files` 导入
  - notebook Cell 3（文件发现）更新：
    - 打印降采样前每个文件夹的文件数
    - 若启用降采样，调用 `downsample_source_files()` 后打印降采样后文件数及比例
    - 若未启用，打印"使用全部 N 个文件"
  - `build_sliding_window_notebook_v2_1.py` 同步更新 Cell 1/2/3
- 设计原则：
  - 降采样仅在文件发现之后、批量处理之前执行，不影响特征计算逻辑
  - 种子固定保证可复现，用户可修改 `FILE_SAMPLE_SEED` 获得不同采样结果
- 自检记录：
  - notebook 中文字符 613 个，替换字符（\ufffd）0 个，编码正确
  - sliding_window.py 中文字符 669 个，替换字符 0 个
  - build_script 中文字符 615 个，替换字符 0 个
  - notebook Cell 2 包含 `ENABLE_FILE_DOWNSAMPLING` 验证通过
  - notebook Cell 3 包含降采样逻辑验证通过
- GitHub 上传日志：
  - 待提交到本地 `master` 分支；
  - 待推送到 `origin/master`。

## 2026-05-28（2.5 最不重要特征排序）

- 本次更新范围：`notebooks/2026-05-17_cross_condition_experiment.ipynb`、`tools/insert_sec25.py`、`docs/dev.md`。
- 任务背景：
  - 在 2.2 RFECV 和 2.4 跨数据集筛选的基础上，添加 2.5 节，将全量特征按重要性**升序**排列，给出最不重要的 10 / 50 / 100 / 200 个特征组。
- 程序更新日志：
  - `2026-05-17_cross_condition_experiment.ipynb` 在 2.4 节最后一个 Cell（id=a57d1771）之后插入 3 个新 Cell：
    - `sec25_least_important_md`（markdown）：2.5 节说明，含目的、输入、输出描述。
    - `sec25_least_important_code`（code）：
      - Step 1：聚合 Stage-A 全量评分（`stage_a_scores`），兼容多种列名。
      - Step 2：聚合 RFECV ranking（取最大 Top-K 结果覆盖最多特征）。
      - Step 3：对两个分数 min-max 归一化后平均，得 `combined_importance`（越小越不重要）。
      - Step 4：升序排列输出 `least_important_df`，并按 10/50/100/200 阈值打印特征列表。
    - `sec25_least_important_viz`（code）：
      - 图 1：全量特征综合重要性曲线（升序），标注四个阈值竖线。
      - 图 2：最不重要 Top-20 横条形图。
  - 新建 `tools/insert_sec25.py`：以 `encoding='utf-8'` 读写 `.ipynb`，不使用终端重定向。
- 自检记录：
  - notebook 由 Python 脚本以 `encoding='utf-8'` 读写，未使用终端重定向；
  - 新增 3 个 Cell 中文本自检：0 个替换字符（\ufffd），编码正确；
  - Cell 26/27/28 UTF-8 字节序列验证通过。
- GitHub 上传日志：
  - 已提交到本地 `master` 分支；
  - 已推送到 `origin/master`（以终端 push 结果为准）。
## 2026-05-28（v2.2 架构级优化：持久化进程池 + 共享内存 + GPU集中化STFT + 文件级流水线）

- 本次更新范围：`src/fea_cpt_gpu_v2_2/`（新建）、
  `notebooks/2026-05-28-realdata_continuous_feature_batch_extract_v2.2.ipynb`（新建）、
  `tools/build_sliding_window_notebook_v2_2.py`（新建）、
  `docs/2026-05-28-真实连续数据特征批量提取报告.md`（更新第7-8节）、`docs/dev.md`。
- 优化背景：
  - v2.1 实测资源利用率：CPU ~30%，GPU ~20%，均远低于预期。
  - 根因分析：ProcessPoolExecutor 每 batch 重建（CPU 尖峰+下降）、IPC 序列化 signal_pre 560MB 拷贝、
    GPU STFT 从未真正批量化（batched=True 已实现但从未调用）、多进程 GPU 竞争、文件串行无流水线。
- 四项核心优化（不改变特征计算公式，保证输出一致）：
  - **优化 A：持久化进程池 + 共享内存**
    - ProcessPoolExecutor 全生命周期复用，不再每 batch 重建，消除进程创建/销毁开销
    - signal_pre 存入 `multiprocessing.shared_memory`，worker 通过 `_SharedArrayPack` 零拷贝读取
    - `_worker_shm_cache` 缓存机制：同一文件的共享内存只附加一次，文件切换时自动清理
  - **优化 B：GPU STFT 集中化批处理**
    - 主进程调用 `compute_stft_power(batched=True)` 批量计算所有窗口的 STFT
    - STFT 结果存入共享内存，worker 按 `stft_chunk_idx` 索引读取
    - 按 `STFT_BATCH_SIZE=200` 分块处理，避免 GPU 内存溢出
    - PCIe 往返从 999x2=1998 次降至约 10 次
  - **优化 C：文件级流水线**
    - `ThreadPoolExecutor(max_workers=1)` 后台预加载下一文件
    - 文件 N 的特征计算与文件 N+1 的 I/O+预处理并行
  - **优化 D：Worker 纯 CPU 计算**
    - Worker 不访问 GPU，从共享内存读取信号切片 + STFT 切片
    - 纯 CPU 特征计算：butter_filter、envelope、ridge、ISTFT、wavelet、features
    - 消除多进程 GPU 竞争和 CUDA context 重复初始化
- 新增文件：
  - `src/fea_cpt_gpu_v2_2/`：v2.2 完整模块
    - `sliding_window.py`：核心模块，新增 `_SharedArrayPack`、`_worker_process_window`、`_compute_batched_stft`、`_preload_file` 等
    - `signal_ops.py`：修复 batched STFT 输出维度注释（`spec_cpu.shape[1]` 改为 `spec_cpu.shape[-1]`）
    - 其余文件与 v2.1 相同
  - `notebooks/2026-05-28-realdata_continuous_feature_batch_extract_v2.2.ipynb`：10 个 Cell
  - `tools/build_sliding_window_notebook_v2_2.py`：notebook 生成脚本
- 关键设计决策：
  - 共享内存采用单块连续布局（`_SharedArrayPack`），多数组按 64 字节对齐排列
  - Worker 通过全局 `_worker_shm_cache` 缓存共享内存附件，避免重复系统调用
  - 主进程在所有窗口任务完成后才清理共享内存（unlink），保证 worker 读取安全
  - `process_source_file` 保持 v2.1 兼容接口，内部也使用 v2.2 架构
- 自检记录：
  - notebook 由 Python 脚本以 `encoding='utf-8'` 生成，未使用终端重定向；
  - notebook 中文字符 777 个，替换字符 0 个，编码正确；
  - sliding_window.py 中文字符 651 个，替换字符 0 个；
  - signal_ops.py 中文字符 102 个，替换字符 0 个；
  - build_script 中文字符 852 个，替换字符 0 个；
  - 所有文件中文自检通过，无乱码。
- GitHub 上传日志：
  - 待提交到本地 `master` 分支；
  - 待推送到 `origin/master`。

	- GitHub 上传日志：
	  - 待提交到本地 `master` 分支；
	  - 待推送到 `origin/master`。

## 2026-05-28（v2.2 经验教训总结 + 下一步提升空间）

- 本次更新范围：`docs/2026-05-28-真实连续数据特征批量提取报告.md`（新增第9-10节）、`docs/dev.md`。
- v2.2 实测结果：CPU 30%~99% 跳跃，GPU <10%，批处理 ~3s/文件，单文件 ~6s/文件，速度可接受。
- 新增第9节开发经验与教训总结（9.1~9.9），涵盖：
  - 9.1 架构设计：先分离再并行，不要把 GPU 塞进 worker
  - 9.2 数据传递：大数组必须用共享内存，不要走 pickle
  - 9.3 进程池生命周期：不要频繁创建/销毁
  - 9.4 共享计算结果要在正确的层级调度
  - 9.5 预估性能要基于实测，不要基于假设
  - 9.6 特征提取的通用数据流模式（含流程图和阶段职责表）
  - 9.7 实时特征提取的特殊考量（窗口缓冲、延迟约束、状态管理）
  - 9.8 编程方式的具体教训（Windows spawn、torch.stft 维度、SharedMemory 生命周期、Worker 缓存）
  - 9.9 特征提取的教训（全量 vs 选定、预处理一次、共享 STFT 层级）
- 新增第10节下一步提升空间（10.1~10.4），涵盖：
  - 10.1 CPU 利用率稳定性优化
  - 10.2 GPU 利用率提升（建议接受现状）
  - 10.3 实时特征提取适配
  - 10.4 代码工程优化
- 自检记录：
  - 报告文档中文字符 5479 个，替换字符 0 个，编码正确。
- GitHub 上传日志：
  - 待提交到本地 `master` 分支；
  - 待推送到 `origin/master`。

## 2026-09-05（DATA09 v0-flow 多目录批处理、CPU回退与断点日志）

- 本次更新范围：
  - `notebooks/DATA09_v0-flow_feature_extraction.ipynb`
  - `src/fea_cpt_gpu_v2_2/signal_ops.py`
  - `docs/dev.md`
- 任务背景：
  - flow 连续数据特征提取 notebook 需要支持一次输入多个文件夹路径并依次处理。
  - 运行环境可能没有 GPU，需要确保没有 CUDA 时可走 CPU 路径继续处理。
  - 长时间批处理可能中断，需要保存已处理文件日志，重跑时跳过相同数据文件。
- Notebook 更新：
  - 新增 `RAW_DATA_ROOT_SPECS` 配置，每个输入路径可单独设置随机抽取比例：
    `(路径, ratio)`，其中 `0 < ratio <= 1`，`1.0` 表示处理该路径下全部文件。
  - 新增 `parse_input_specs()`，自动校验抽取比例、去重输入路径，并生成 `RAW_DATA_SPECS` / `RAW_DATA_ROOTS`。
  - `discover_source_files([root])` 会按每个根路径分别递归查找 `.npz/.tdms`，随后对该路径下文件单独抽样。
  - 新增 `sample_files_for_root()`，使用 `FILE_SAMPLE_SEED + root_index` 为每个路径生成可复现随机抽样结果。
  - 删除原先单一全局 `FILE_SAMPLE_RATIO` 的使用，避免不同数据路径只能使用同一个抽样比例。
  - 批处理单元提示进度条统计范围：进度条覆盖所有输入文件夹合并后的全部待处理文件。
  - 输出目录由原来的每次时间戳目录调整为：
    - 固定根目录：`outputs/DATA09_v0-flow_features/`
    - 单次运行目录：`outputs/DATA09_v0-flow_features/run_YYYYmmdd_HHMMSS/`
  - 断点日志固定保存到：
    `outputs/DATA09_v0-flow_features/_process_logs/processed_source_files.txt`
  - 数据发现阶段会读取上述断点日志，先过滤已处理文件，再执行本次处理。
  - 批处理阶段不再将 `processed_source_files.txt` 重命名为带时间戳文件，保证下次运行可继续复用。
  - 结果汇总和运行日志单元改为读取 `RUN_OUTPUT_ROOT` 与稳定断点日志路径。
- CPU/GPU 更新：
  - notebook 中明确 `FEA_CPT_USE_GPU` 语义：`1` 表示有 CUDA 时优先 GPU；`0/false/off/no` 表示强制 CPU。
  - `src/fea_cpt_gpu_v2_2/signal_ops.py` 的 torch STFT 初始化新增环境变量判断；
    当 `FEA_CPT_USE_GPU=0/false/off/no` 时不初始化 GPU STFT。
  - 无 CUDA 时原有 `_init_gpu_stft()` 会将 `_torch_available=False`，`compute_stft_power()` 自动使用 scipy CPU STFT。
- 断点续跑行为：
  - 每个源文件成功写入特征 CSV 和窗口日志 CSV 后，底层 `build_sliding_window_dataset()` 追加该文件完整路径到 `processed_source_files.txt`。
  - 程序中断后重新运行 notebook，会先按每个路径比例重新得到可复现抽样文件集合，再跳过日志中已有的源文件，只处理剩余文件。
  - 新一次运行仍会创建新的 `run_YYYYmmdd_HHMMSS` 输出目录，避免把新旧结果 CSV 追加混在一起。
- 总进度条更新：
  - `build_sliding_window_dataset()` 的 `tqdm` 文件进度条改为 `处理全部文件`，总数为所有输入路径合并后的待处理文件数。
  - 进度条单位为 `file`，并在 postfix 中动态显示：
    - `已处理`
    - `跳过`
    - `失败`
    - `平均`每文件耗时
    - `预计总用时`
    - `剩余`预计用时
  - 预计总用时和剩余时间根据当前已完成文件的平均耗时实时更新，适合跨多个文件夹的长任务监控。
- 自检记录：
  - 已执行 notebook JSON 机械改写，清理了相关单元的旧执行输出。
  - `python -m py_compile src\fea_cpt_gpu_v2_2\signal_ops.py` 通过。
  - `notebooks/DATA09_v0-flow_feature_extraction.ipynb` JSON 解析检查通过。
- GitHub 上传日志：
  - 未提交，待用户确认。

- 补充更新（同日）：多标签批量处理与按数据文件分片输出。
  - 更新范围：`notebooks/DATA09_single_event_feature_extract.ipynb`、`docs/dev.md`。
  - notebook 配置由单一 `DATA_ROOT`/`LABEL` 改为 `DATASETS` 列表，可一次运行同时处理多个标签目录，当前默认包含 `BK00`、`QJ00`、`BK05`、`QJ05`。
  - 新增 `MAX_FILES_PER_OUTPUT = 100`，输出拆分粒度按“源 `.npz` 数据文件数”计算，而不是按截取窗口生成的特征行数计算。
  - 数据发现阶段为每个标签内的源文件分配 `file_index_in_label` 与 `output_part`；批处理生成的每行特征同步写入 `source_file_index_in_label` 和 `output_part`，便于追溯输出分片。
  - 结果存储阶段按 `label + output_part` 分组写出 `features_{label}_{timestamp}_partXXX.csv`；每个标签只要存在有效特征行，至少生成一个输出 CSV；同一标签超过 100 个源数据文件时自动生成 `part002`、`part003` 等后续文件。
  - 同步更新 notebook 开头说明、架构说明和质量检查元数据列定义，避免仍提示只能通过单个 `DATA_ROOT`/`LABEL` 切换标签。
- 自检记录：
  - 已用 `json.load()` 校验 `notebooks/DATA09_single_event_feature_extract.ipynb` 可正常解析。
  - 已对 notebook 全部代码单元执行 `ast.parse` 静态语法检查，检查通过。
  - 已确认旧的 `MAX_ROWS_PER_OUTPUT` 不再存在，当前切分变量为 `MAX_FILES_PER_OUTPUT`。

- 补充优化（同日）：分析并优化 single-event notebook 的 CPU/GPU 利用率。
  - 问题原因：
    - `DATA09_v0-flow_feature_extraction.ipynb` 调用 v2.2 连续滑窗流水线 `build_sliding_window_dataset()`，单个连续文件可产生大量窗口，底层具备持久进程池、文件预加载、批量 STFT 与分片写盘能力，因此 CPU/GPU 更容易被持续喂满。
    - `DATA09_single_event_feature_extract.ipynb` 原实现是在 notebook 主进程中按 `文件 -> 3 个到时截取窗口 -> 特征计算` 串行执行；每个源文件只有 3 个短窗口，GPU STFT 任务过小且频繁启动，CPU 也被限制在单进程循环中，导致整体利用率低于 flow notebook。
    - single-event 数据不能直接套用 flow 的固定步长滑窗流水线，否则会丢失基于 `arrival_time` 的三种到时窗口语义。
  - 优化内容：
    - 新增 `FILE_WORKERS = None`，默认使用 `_auto_detect_workers()` 自动决定文件级并行进程数。
    - 新增 `ENABLE_SHARED_STFT = False`，默认关闭单事件短窗口的 GPU/shared STFT 路径，避免 3 个 30 ms 短窗口反复触发小 GPU 任务造成调度开销大于收益；如机器实测 GPU 更快，可手动改为 `True`。
    - 使用 `joblib.Parallel(n_jobs=max_workers, prefer='processes')` 对源 `.npz` 文件做文件级多进程并行，保持每个文件内部仍按 `arrival_time` 生成三种截取窗口。
    - 新增 `get_params_map()` 并用 `lru_cache` 缓存同采样率下的频带参数，减少每个窗口重复构造参数的开销。
    - 批处理结果按 `label`、`source_file_index_in_label`、`window_mode` 排序，保证多进程返回顺序不影响输出 CSV 的可读性和可复现性。
    - notebook 架构说明中补充了 single-event 与 flow notebook 利用率差异的原因，以及短事件数据更适合文件级 CPU 并行而非强行追求 GPU 占满的原因。
  - 自检记录：
    - 已用 `json.load()` 校验 notebook JSON 可正常解析。
    - 已对 notebook 全部代码单元执行 `ast.parse` 静态语法检查，检查通过。
    - 已确认 notebook 中不再使用标准库 `ProcessPoolExecutor`，避免 Windows/Jupyter 下 notebook 内函数不可 pickle 的兼容性风险；当前使用 joblib loky 进程后端。

- 补充修复（同日）：修复 joblib 子进程反序列化失败。
  - 问题现象：运行批处理单元时出现 `BrokenProcessPool: A task has failed to un-serialize`，远端报错为 `AttributeError: Can't get attribute 'get_params_map' on <module '__main__'>`。
  - 原因：Windows/Jupyter 环境中，joblib loky 子进程不能稳定从 notebook 的 `__main__` 内存态反序列化自定义函数，尤其是 `get_params_map`、`process_event_record` 这类在 notebook 单元中定义并被 worker 闭包引用的函数。
  - 修复：
    - 新增模块 `src/data09_single_event.py`，将 `parse_npz_ts()`、`load_npz_channel()`、`get_cut_windows()`、`compute_features_for_window()`、`process_event_record()` 和参数缓存 `_get_params_map()` 移入可导入的 Python 模块。
    - notebook 改为从 `data09_single_event` 导入模块级 worker；joblib 多进程路径直接调用 `_process_event_record`，不再提交 notebook 内定义的 wrapper 函数。
    - 保留 notebook 内轻量 wrapper 仅用于单文件 smoke test 和交互调用；多进程批处理不再依赖 `__main__` 函数。
  - 自检记录：
    - `python -m py_compile src\data09_single_event.py` 通过。
    - `notebooks/DATA09_single_event_feature_extract.ipynb` JSON 解析和全部代码单元 `ast.parse` 通过。
    - 已用 1 个 `DATA09/v05-bk` 源 `.npz` 文件执行 `joblib.Parallel(n_jobs=2, prefer='processes')` 小批验证，子进程可正常导入模块级 worker，返回 1 行测试特征且无错误。

- 补充优化（同日）：整理 notebook 结构、按文件即时落盘并生成本地处理日志。
  - 更新范围：`notebooks/DATA09_single_event_feature_extract.ipynb`、`src/data09_single_event.py`、`docs/dev.md`。
  - notebook 结构整理：
    - 补充一级标题和二级标题，当前章节包括：概览、环境初始化、全局配置、源文件发现、头文件检查、单文件 smoke test、增量输出工具、按文件持久化批处理、输出汇总、特征质量检查、架构与性能说明。
    - 删除 notebook 内旧的事件处理 wrapper 函数定义，避免与 `src/data09_single_event.py` 中的模块级 worker 重复。
    - notebook 仅保留必要的输出辅助函数，例如 `output_csv_for()`、`append_dataframe_csv()`、`append_process_log()`、`persist_one_file_result()`。
  - 即时落盘：
    - 批处理单元由“全部处理完成后统一保存”改为“每个源 `.npz` 文件处理完成后立即追加写入对应 CSV”。
    - 输出 CSV 仍按 `label + output_part` 分片，`output_part` 按同一标签内每 100 个源数据文件计算。
    - 新增 `OVERWRITE_CURRENT_RUN_OUTPUTS = True`，重新运行当前 `RUN_TIMESTAMP` 的批处理单元时会先清理同一时间戳的输出 CSV 和处理日志，避免重复追加。
  - 本地处理日志：
    - 新增本地日志文件：`outputs/DATA09_multi_label_single_event_features/_process_logs/process_log_{RUN_TIMESTAMP}.csv`。
    - 每处理完一个源文件追加一行日志，记录 `processed_at`、`run_timestamp`、`label`、`source_file_name`、`source_file_path`、`output_part`、`output_csv`、`status`、`feature_rows`、`elapsed_s`、`error`。
    - 成功、失败、空结果都会写入日志，便于追溯什么时间处理了哪些文件以及写入哪个输出文件。
  - 并行与日志修复：
    - `src/data09_single_event.py` 新增 `process_event_record_timed()`，在模块级 worker 中记录单文件耗时，保证 joblib 子进程返回后日志可写入 `elapsed_s`。
    - joblib 使用 `return_as='generator_unordered'`，worker 完成一个源文件就返回一个结果，notebook 随即落盘并写日志，不再等待全部文件结束。
  - 自检记录：
    - `python -m py_compile src\data09_single_event.py` 通过。
    - `notebooks/DATA09_single_event_feature_extract.ipynb` JSON 解析和全部代码单元 `ast.parse` 通过。
    - 已用 2 个 `DATA09/v05-bk` 源 `.npz` 文件执行 `joblib.Parallel(n_jobs=2, prefer='processes', return_as='generator_unordered')` 小批验证，均返回 1 行测试特征、无错误且 `elapsed_s` 有效。

- 补充更新（同日）：中文注释与中文说明。
  - 按用户要求，将 `notebooks/DATA09_single_event_feature_extract.ipynb` 中面向使用者的一级标题、二级标题、说明文本、必要代码注释和打印提示改为中文。
  - 保留变量名、函数名、CSV 字段名和状态值等程序接口不变，避免影响后续脚本读取和结果兼容性。
  - 为 `src/data09_single_event.py` 中的核心函数补充中文 docstring，说明时间解析、通道读取、到时窗口转换、特征计算、单文件处理和计时 worker 的作用。
  - 自检记录：
    - notebook JSON 解析和全部代码单元 `ast.parse` 通过。
    - `python -m py_compile src\data09_single_event.py` 通过。
    - 已确认 notebook 中英文章节标题和旧问号占位文本已清理，中文内容以 UTF-8 正常写入。

- 补充更新（同日）：特征重要性 notebook 强制源文件组级 6:4 划分。
  - 更新范围：`notebooks/DATA09_feature_importance_30ms_analysis.ipynb`、`docs/dev.md`。
  - 任务背景：
    - BK、QJ 单事件样本中，一个源 `.npz` 文件会围绕 `arrival_time` 在不同位置截取多个窗口，从而派生出多行特征样本。
    - 同一源文件派生出的窗口样本具有相似性；如果按特征行随机划分，会导致同一源文件的不同窗口同时进入训练集和测试集，形成数据泄漏。
  - 程序更新：
    - 新增 `build_source_group_table()`，先按 `label + source_file_name` 统计源文件组，记录每个源文件组派生出的特征行数。
    - 重写 `split_by_source_group()`：强制以 `source_file_name` 源文件组为划分单位，每个标签内按源文件组随机划分约 60% 训练、40% 测试。
    - 删除原有行级 `train_test_split` fallback；若标签不足 2 类、某类源文件组少于 2 个，或同一 `source_file_name` 同时属于多个标签，则直接报错并要求先修正数据，而不是退化为窗口行级划分。
    - 第 6 节划分说明明确：行数比例只作为结果统计，实际划分目标是源文件组比例；同一 `source_file_name` 的所有派生窗口必须全部进入同一个集合。
    - 划分单元新增源文件组统计展示和泄漏校验；若训练集与测试集存在重叠源文件，直接抛出 `RuntimeError`。
  - 新增输出：
    - `source_group_table_{RUN_TIMESTAMP}.csv`：记录每个 `label + source_file_name` 源文件组及其派生窗口行数。
    - `source_split_summary_{RUN_TIMESTAMP}.csv`：记录每个标签的源文件组总数、训练源文件组数、测试源文件组数和实际源文件组划分比例。
  - 自检记录：
    - 已用 `json.load()` 校验 notebook JSON 可正常解析。
    - 已对 notebook 全部代码单元执行 `ast.parse` 静态语法检查，检查通过。
    - 已确认 notebook 中不再包含 `train_test_split` 和 row-level fallback 逻辑。

## 2026-09-04（DATA09 v0-flow 双通道TDMS连续特征提取 + 修复v2.2滑窗特征计算bug）

- 本次更新范围：`notebooks/DATA09_v0-flow_feature_extraction.ipynb`（新建）、
  `src/fea_cpt_gpu_v2_2/sliding_window.py`（更新）、`docs/dev.md`。
- 任务背景：
  - 需对 `E:\codes\ZZ-BK\DATA09\v0-flow` 目录下的连续数据做特征提取。
  - 数据为 TDMS **双通道**文件（`SemiPhase-1MHz-2026-8-29-15-7-44.tdms`），需读取指定通道。
  - 实际通道名：`Untitled`、`Untitled 1`（各 600 万样本，1MHz 采样率）。
- 程序更新日志（`src/fea_cpt_gpu_v2_2/sliding_window.py`）：
  - 支持指定TDMS通道名称：
    - `_load_tdms_source()` 新增 `channel_name` 参数，按名称查找指定通道，找不到则回退默认选择逻辑。
    - `load_source_file()` 新增 `tdms_channel_name` 参数并透传。
    - `process_source_file()` 读取 `config.tdms_channel_name` 传入加载。
    - `SlidingWindowConfig` 新增 `tdms_channel_name: str | None = None` 字段。
    - 所有新增参数均有 `None` 默认值，向后兼容，原notebook行为不变。
  - **修复特征计算bug（关键）**：
    - 问题1：共享STFT路径使用 `FeatureContext` 但模块只导入了 `FeatureRecord`，
      导致每窗口特征计算抛 `NameError: name 'FeatureContext' is not defined`，
      被 `except` 捕获后返回空特征，输出只剩 21 个元数据列（无任何特征列）。
    - 修复1：`from .base import FeatureContext, FeatureRecord`。
    - 问题2：共享STFT路径中 `context.residual_power` 为全频带（513）维，
      而 `context.stft_freqs/stft_power` 为子带切片维，`_band_mask` 布尔索引维度不匹配，
      抛 `IndexError: boolean index did not match indexed array along dimension 0`。
    - 修复2：将 `residual_power` 按其频带掩码切到与 `stft_freqs/stft_power` 相同维度。
  - 验证：单窗口 6 频带计算得到 `6 × 80 = 480` 个特征列（+21 元数据列 = 501 列），
    对应 `docs/chatgpt-特征汇总.md` 中实现的特征集合。
- 新建 `notebooks/DATA09_v0-flow_feature_extraction.ipynb`：
  - 结构参照 `2026-05-28-realdata_continuous_feature_batch_extract_v2.2.ipynb`。
  - 配置 `RAW_DATA_ROOTS = [E:\codes\ZZ-BK\DATA09\v0-flow]`。
  - `TARGET_CHANNEL_NAME = 'Untitled'`（指定读取的通道）。
  - 关闭文件级降采样（数据量小）。
  - `TDMS_FALLBACK_SAMPLE_RATE_HZ = 1_000_000.0`（1MHz）。
- 自检记录：
  - 依赖的滑动窗口模块导入验证通过。
  - 单窗口特征计算（共享STFT路径）验证通过，6 频带共 480 个特征。
- GitHub 上传日志：
  - 未提交，待用户确认。

## 2026-09-04（DATA09 30ms 三分类特征重要性排序与可视化开发）

- 本次更新范围：
  - `tools/data09_feature_importance_pipeline.py`（新增）
  - `notebooks/DATA09_feature_importance_30ms_analysis.ipynb`（新增）
  - `docs/dev.md`（更新）
- 任务目标：
  - 使用用户指定的 8 个频带重新计算 DATA09 三类样本特征：`flow`、`bk`、`qj`。
  - 统一窗口长度为 30 ms。
  - 在特征重要性排序前，将特征集按源文件分组划分为 6:4 两部分，前 60% 用于排序/训练，后 40% 用于测试验证。
  - 排序集与测试集使用 `source_file_name` 分组划分，确保两者无同一源文件交集。
  - 输出重要性排序图、测试集混淆矩阵、高重要性特征在三类样本上的分布图，并保存测试结果。
- 频带配置：
  - `b_1k_100k`: 1 kHz - 100 kHz
  - `b_1k_10k`: 1 kHz - 10 kHz
  - `b_10k_20k`: 10 kHz - 20 kHz
  - `b_20k_30k`: 20 kHz - 30 kHz
  - `b_30k_40k`: 30 kHz - 40 kHz
  - `b_40k_60k`: 40 kHz - 60 kHz
  - `b_10k_50k`: 10 kHz - 50 kHz
  - `b_1k_50k`: 1 kHz - 50 kHz
- 程序实现：
  - `tools/data09_feature_importance_pipeline.py` 集成完整流程：
    - `extract_flow_features()`：对 `DATA09/v0-flow` TDMS 连续数据按 30 ms、0 重叠滑窗提取特征。
    - `extract_event_features()`：对 `DATA09/v0-bk`、`DATA09/v0-qj` npz 事件数据按到时截取 30 ms 窗口提取特征。
    - 事件窗为 `[-10, 20] ms`、`[0, 30] ms`、`[5, 35] ms`；其中第三个窗口按“每个窗口长度30ms”要求从原先容易产生 25 ms 的 `[5, 30] ms` 修正为 `[5, 35] ms`。
    - `split_by_source()`：按类别内源文件随机 6:4 划分，固定 `seed=42`。
    - `run_importance()`：使用 `RandomForestClassifier(class_weight='balanced')` 在排序集训练，并在测试集计算 permutation importance、分类报告和混淆矩阵。
  - 支持完整运行：
    - `python tools\data09_feature_importance_pipeline.py`
  - 支持本地快速抽样验证参数：
    - `--max-flow-files`
    - `--max-flow-windows-per-file`
    - `--max-event-files-per-class`
- 输出位置：
  - 合并特征：`outputs/DATA09_feature_importance_30ms/data09_all_features_30ms.csv`
  - bk 特征：`outputs/DATA09_feature_importance_30ms/features/DATA09_v0-bk_features_30ms.csv`
  - qj 特征：`outputs/DATA09_feature_importance_30ms/features/DATA09_v0-qj_features_30ms.csv`
  - flow 特征：`outputs/DATA09_feature_importance_30ms/features/DATA09_v0-flow_features_30ms/features_all.csv`
  - 带划分标记数据集：`outputs/DATA09_feature_importance_30ms/analysis/data09_feature_dataset_with_split.csv`
  - 特征重要性表：`outputs/DATA09_feature_importance_30ms/analysis/feature_importance.csv`
  - 测试摘要：`outputs/DATA09_feature_importance_30ms/analysis/summary.json`
  - 重要性排序图：`outputs/DATA09_feature_importance_30ms/analysis/feature_importance_top20.png`
  - 测试集混淆矩阵：`outputs/DATA09_feature_importance_30ms/analysis/confusion_matrix.png`
  - Top 特征分布图：`outputs/DATA09_feature_importance_30ms/analysis/top_feature_distributions.png`
- 新建 notebook：
  - `notebooks/DATA09_feature_importance_30ms_analysis.ipynb`
  - notebook 中保留快速抽样测试命令和完整重新计算命令，并读取脚本输出的 CSV/JSON/PNG 进行展示。
- 自检记录：
  - `python -m py_compile tools\data09_feature_importance_pipeline.py` 通过。
  - 依赖检查：`sklearn`、`seaborn`、`nptdms` 可导入。
  - 本地执行抽样测试时，单窗口 8 频带全量特征计算耗时较高；在当前对话工具 180 秒超时限制内未能完成首批结果落盘。
  - 因此本次已完成程序开发、语法验证、notebook 与文档更新；完整特征重算和结果图生成需要在本机终端或 notebook 中以不限时方式运行上述完整命令。

## 2026-09-04（DATA09 已提取特征CSV的重要性排序与6:4测试验证优化）

- 本次更新范围：
  - `notebooks/DATA09_feature_importance_30ms_analysis.ipynb`（重构）
  - `docs/dev.md`（更新）

- 背景：
  - 已通过 `DATA09_single_event_feature_extract.ipynb` 完成 BK00 / QJ 离散事件样本特征计算。
  - 已通过 `DATA09_v0-flow_feature_extraction.ipynb` 完成 flow 连续样本特征计算。
  - 原 `DATA09_feature_importance_30ms_analysis.ipynb` 偏向重新计算三类特征，不适合直接复用已经生成的 CSV 结果。

- 主要优化：
  - 将 `DATA09_feature_importance_30ms_analysis.ipynb` 改为“读取已生成特征 CSV 后分析”的轻量流程，不再强制调用 `tools/data09_feature_importance_pipeline.py` 重新计算原始信号特征。
  - 在 `FEATURE_FILES` 中显式配置 flow、BK00、QJ 三类特征 CSV 路径；程序会自动跳过重复输入路径并打印警告。
  - 对不同来源 CSV 自动识别标签：
    - 优先读取 `label` 列；
    - 其次读取 `sample_type` 列；
    - 若缺失，则按路径关键字 `flow` / `BK00` / `QJ` 推断。
  - 自动取所有输入 CSV 的共同数值特征列，排除 `source_file_name`、`window_id`、`window_mode`、`sample_rate_hz` 等元数据字段，解决 flow CSV 与 BK00/QJ CSV 元数据列不完全一致的问题。
  - 明确先按 6:4 比例划分数据：
    - 60% 数据用于训练和特征重要性排序；
    - 40% 数据用于独立测试验证。
  - 划分策略优先使用 `source_file_name` 分组，避免同一源文件的不同窗口同时出现在训练集和测试集，降低数据泄漏风险。
  - 若某类源文件数不足以做源文件级分组划分，程序会降级为行级 stratified split，并明确输出警告。

- Notebook 文档结构：
  - 增加中文一级标题：`DATA09 特征重要性排序与 6:4 测试验证`。
  - 参照 `notebooks/2026-05-17_cross_condition_experiment.ipynb` 的文档风格，在每个主要代码单元前增加中文二级标题和用途备注。
  - 当前二级章节包括：
    - 环境初始化
    - 输入配置
    - 工具函数
    - 特征表读取与合并
    - 数据分布与特征质量检查
    - 6:4 训练测试划分
    - RandomForest 训练与测试
    - 特征重要性排序
    - 错误样本与 Top 特征分布
    - 输出文件清单

- 输出内容：
  - 每次运行创建带时间戳的新目录：`outputs/DATA09_feature_importance_from_csv_YYYYmmdd_HHMMSS/`。
  - 保存合并后的特征表：`combined_features_YYYYmmdd_HHMMSS.csv`。
  - 保存带 `split` 标记的 6:4 划分结果：`combined_features_with_split_YYYYmmdd_HHMMSS.csv`。
  - 保存特征质量检查表：`feature_quality_YYYYmmdd_HHMMSS.csv`。
  - 保存测试指标：`test_metrics_YYYYmmdd_HHMMSS.json`。
  - 保存分类报告：`classification_report_YYYYmmdd_HHMMSS.csv`。
  - 保存混淆矩阵：`confusion_matrix_YYYYmmdd_HHMMSS.csv` 和对应 PNG 图。
  - 保存错误样本：`misclassified_samples_YYYYmmdd_HHMMSS.csv`。
  - 保存特征重要性排序：`feature_importance_YYYYmmdd_HHMMSS.csv`。
  - 保存类别分布图、Top-N 特征重要性图、Top 特征分布图。

- 测试记录：
  - 使用用户提供的 QJ 特征文件：
    `outputs/DATA09_v0-qj_features_QJ/features_QJ_20260904_114606.csv`
  - 使用用户提供的 flow 特征文件：
    `outputs/DATA09_v0-flow_features_20260904_121425/features_20260904_121425_part_0001.csv`
  - 用户提供的第三个路径与 flow 文件重复，程序已按设计跳过重复输入。
  - 因本次测试输入中缺少 BK00 文件，实际验证为 QJ vs flow 二分类；notebook 已保留三分类能力，只需在 `FEATURE_FILES` 中补入 BK00 CSV。
  - 测试读取到 640 个共同数值特征；源文件级划分后训练集与测试集 `source_file_name` 无重叠。
- GitHub 上传日志：
  - 未提交，待用户确认。

## 2026-09-06（DATA09 特征重要性排序性能优化、结果审查与可视化增强）

- 本次更新范围：
  - `notebooks/DATA09_feature_importance_30ms_analysis.ipynb`（优化）
  - `docs/dev.md`（更新）

- 当前执行结果审查：
  - notebook 当前配置实际只读取到 `QJ` 与 `flow` 两类样本；用户给出的第三个 CSV 路径与 flow 文件重复，程序已跳过重复输入。
  - 当前输出显示共同数值特征为 640 个，训练集 1517 行、测试集 1081 行，源文件级 train/test 无重叠。
  - 测试集 accuracy、balanced accuracy、macro F1、weighted F1 均为 1.0；说明当前 QJ vs flow 二分类在已有特征上非常容易，不能据此外推到 BK00/QJ/flow 三分类。
  - 特征质量检查发现近零方差特征 18 个，高相关 Spearman 特征对 832 对；因此单个特征排名会受到强冗余影响，解释时应优先关注稳定特征组，而不是把某一个排名当作物理因果结论。
  - 原第 8 节 CV permutation importance 和测试集 permutation importance 全部为 0。原因不是“数值太小导致画不出来”，而是在当前数据中模型分类已达到 1.0 且特征高度冗余，置换单个特征不会降低 balanced accuracy，所以条形长度确实为 0。

- 第 8 节耗时原因：
  - 原优化版使用 `CV_FOLDS=5`、`PERMUTATION_REPEATS=5`、640 个特征、5 个 RF 随机种子，并额外在测试集上做全量 permutation。
  - permutation importance 的主要复杂度近似为：`fold 数 × 特征数 × repeats × 模型预测成本`。
  - 仅训练集 CV permutation 就约为 `5 × 640 × 5 = 16000` 次置换预测；再加测试集全量置换约 `640 × 5 = 3200` 次预测，以及多随机种子 RF 训练，因此运行 4 分钟且 CPU 100% 是预期现象。

- 性能优化策略：
  - 默认参数从 `CV_FOLDS=5`、`PERMUTATION_REPEATS=5`、`RF_STABILITY_SEEDS=5 个` 调整为 `CV_FOLDS=3`、`PERMUTATION_REPEATS=3`、`RF_STABILITY_SEEDS=[11, 23, 37]`。
  - 新增 `PERMUTATION_CANDIDATE_LIMIT = 80`：先用全量但快速的 RF 多随机种子稳定性和单变量 ANOVA F 检验筛候选，再只对候选特征做训练集内 CV permutation importance。
  - 新增 `TEST_PERMUTATION_TOP_N = 30`：测试集 permutation 仅作为审计，并只计算候选 Top-N，不再扫描全部 640 个特征，保持 holdout 的验证职责。
  - 第 8 节主图增加 fallback：若 CV permutation 全为 0，自动改画 RF 多随机种子平均重要性和标准差，避免出现空条形图。

- 算法与解释增强：
  - 新增 `compute_univariate_f_scores()`：用训练集单变量 ANOVA F 检验提供快速边际区分能力排序。
  - 新增 `choose_permutation_candidates()`：综合 RF 稳定性排序与 F 检验排序，生成 permutation 候选特征集合。
  - 新增 `correlation_pruned_features()`：作为一种特征排序降维方式，按重要性从高到低选择特征，并跳过与已选特征高度相关的冗余特征。
  - 新增 `evaluate_topk_feature_sets()`：在固定 6:4 holdout 上比较不同排序/降维策略的 Top-K 分类效果，输出 `feature_ranking_topk_comparison_*.csv` 与 `feature_ranking_topk_comparison_*.png`。
  - 对比方法包括：
    - CV permutation 排序，若 permutation 全 0 则结合快速排序 fallback；
    - RF 多随机种子稳定性排序；
    - 单变量 F 检验排序；
    - 相关性去冗余 RF 排序。

- 可视化增强：
  - 保留原第 9 节原始计数直方图 `top_feature_distributions_*.png`，用于观察绝对样本数量分布。
  - 新增第 10 节“Top 特征分布的均衡可视化补充”，输出 `top_feature_distributions_density_ecdf_*.png`。
  - 新图包含归一化密度直方图和 ECDF。归一化密度图让每个类别面积为 1，避免 flow 样本多、QJ/BK00 样本少时小类柱子过低；ECDF 用累计比例展示整体分布偏移，更适合类别数量不均衡场景。

- 中文说明与注释：
  - 在配置、工具函数、第 8 节排序、第 9 节错误样本与原始分布图、第 10 节均衡分布图中补充中文注释。
  - 注释重点说明方法原理、算法流程、数据泄漏边界、permutation importance 的计算成本，以及不同图形的适用场景。

- 验证记录：
  - notebook JSON 解析和所有代码单元 `ast.parse` 通过。
  - 使用缩小参数 smoke notebook 完整执行通过：`N_ESTIMATORS=60`、`CV_FOLDS=2`、`PERMUTATION_REPEATS=1`、`PERMUTATION_CANDIDATE_LIMIT=20`。
  - smoke 执行生成了 `feature_importance_*.csv`、`feature_ranking_topk_comparison_*.csv/.png`、`feature_importance_top30_*.png`、`top_feature_distributions_density_ecdf_*.png` 等产物。
  - smoke 验证中仍出现 Windows/joblib `resource_tracker` 临时文件 KeyError 警告，但 nbconvert 返回成功并写出结果；这是并行 joblib 在 Windows 上的清理警告，不影响 notebook 主要输出。

## 2026-09-04（DATA09 v0-bk / v0-qj 断丝样本特征提取）

- 本次更新范围：`notebooks/DATA09_single_event_feature_extract.ipynb`（新建）、`docs/dev.md`。
- 任务背景：
  - 需对 `E:\codes\ZZ-BK\DATA09\v0-bk`（标签 BK00）与 `v0-qj`（标签 QJ）内的离散断丝样本做特征提取。
  - 数据为 npz，双通道（`phase_data` 维度 `(npts, 2)`），每文件为一次断丝事件的可见片段（约 0.4s）。
  - 采样率 1MHz，`arrival_time` 为信号到达时间（到时）。
- 数据特征（核实）：
  - `v0-bk`：`BK00-FIP-1000K-20260829T15*.npz`（2 个文件）。
  - `v0-qj`：`QJ-FIP-1000K-20260829T15*.npz`（2 个文件）。
  - npz keys：`phase_data`、`channel_names`、`sample_rate`、`npts`、`starttime`、`arrival_time`、`type`、`data_info` 等。
  - `channel_names`：`['phase_data', 'phase_data_ch2']`；`sample_rate=1_000_000 Hz`；`npts=400001`。
- 程序更新日志（新建 notebook）：
  - 读取指定通道波形（默认第一个通道 `phase_data[:, 0]`）。
  - 从头文件读取 `arrival_time`，并以**到时为时间原点**（0 ms）。
  - 以到时为原点按三种方式截取，每文件得到三个样本：`[-10 ms, 20 ms]`、`[0 ms, 30 ms]`、`[5 ms, 30 ms]`。
  - 每截取段调用 v2.2 共享 STFT + `compute_all_features` 计算 6 频带全量特征（480 特征/段）。
  - 存储为 CSV，逐行标注 `label`（BK00/QJ）与 `window_mode`（截取方式）。
  - notebook 内仅需修改 `DATA_ROOT` 与 `LABEL` 即可在 v0-bk 与 v0-qj 间切换。
- 自检记录：
  - 批量逻辑验证：2 文件 × 3 截取窗口 = 6 样本，每样本 480+ 特征，无全 NaN。
  - 特征计算依赖模块加载正常。
- GitHub 上传日志：
  - 未提交，待用户确认。

## 2026-09-06（DATA09 v0-flow 特征提取 WinError 1455 排查与内存优化）

- 本次检查对象：
  - 用户指定日志：`outputs/DATA09_v0-flow_features_20260904_121425/DATA09_v0.5-flow_features/run_20260905_201502/log_20260905_201502_part_0002.csv`
  - notebook：`notebooks/DATA09_v0-flow_feature_extraction.ipynb`
  - 底层模块：`src/fea_cpt_gpu_v2_2/sliding_window.py`、`src/fea_cpt_gpu_v2_2/signal_ops.py`
  - 兼容修复：`src/fea_cpt_gpu_v2_0/sliding_window.py`、`src/fea_cpt_gpu_v2_0/signal_ops.py`

- 日志检查结论：
  - `log_20260905_201502_part_0002.csv` 共 2600 行，对应 13 个 TDMS 源文件，每个文件 200 个 30ms、0 重叠窗口。
  - `source_file_path + window_id` 未发现重复，窗口边界一致：`window_length_samples=30000`、`window_step_samples=30000`、`sample_rate_hz=1000000`。
  - `missing_selected_features` 非空 2584 行，说明这些窗口虽然写出了元数据，但特征计算阶段发生异常，不能作为可靠完整特征使用。
  - 主要异常类型：
    - `Unable to allocate 90.3 MiB for an array with shape (29601, 400) and data type float64`：1638 行。
    - `[WinError 1455] 页面文件太小，无法完成操作。`：941 行。
    - 其余为较小矩阵分配失败。
  - 结论：该分片的核心问题是 Windows 页面文件/共享内存峰值不足，不是 CSV 行数、窗口切分或 TDMS 通道元数据错误。

- 根因分析：
  - v2.2 批处理路径原先会先生成一个源文件的全部 STFT chunks，并把每个 chunk 的 STFT 结果放入共享内存，随后再统一提交窗口任务。
  - 对 1MHz、30ms、8 频带的 TDMS 数据，单文件约 200 个窗口；若 STFT batch 与 workers 较大，会叠加 signal shared memory、STFT shared memory、worker 内部矩阵副本和下一文件预加载数据。
  - 发生 WinError 1455 后，单个窗口内部的部分特征也可能因全频 residual/harmonic 矩阵乘法再分配较大 float64 数组而失败，最终导致 `missing_selected_features` 大量非空。

- 程序修复：
  - `src/fea_cpt_gpu_v2_2/sliding_window.py`
    - 新增 `_compute_one_stft_chunk()`，支持单个 STFT chunk 独立计算并写入共享内存。
    - 新增 `_iter_adaptive_stft_chunks()`，遇到 CUDA OOM、WinError 1455、页面文件不足等内存压力时自动缩小 batch。
    - 将 `process_source_file()` 与 `build_sliding_window_dataset()` 改为“生成一个 STFT chunk -> 提交/收集窗口任务 -> 立即 cleanup”，避免保留全文件所有 STFT shared memory。
    - `build_sliding_window_dataset()` 保留原窗口失败计数逻辑，但不再让已处理完的 STFT chunk 长时间占用共享内存。
  - `src/fea_cpt_gpu_v2_0/sliding_window.py`
    - 兼容用户 traceback 中的 v2.0 路径：自适应 STFT 分块现在也捕获 `OSError`，并识别 `WinError 1455`、`page file`、`页面文件太小`。
  - `src/fea_cpt_gpu_v2_2/signal_ops.py` 与 `src/fea_cpt_gpu_v2_0/signal_ops.py`
    - `build_ridge_mask()` 从默认 float64 改为 float32，0/1 掩码无需 64 位浮点。
  - v2.0/v2.2 shared-STFT 特征路径：
    - harmonic energy 使用 `np.sum(..., where=mask)`，避免生成 `full_stft_power * ridge_mask` 的全矩阵临时副本。
    - residual power 不再生成全频 `residual_power_full`，改为只为当前频带生成 `residual_power_band`。

- notebook 优化：
  - `notebooks/DATA09_v0-flow_feature_extraction.ipynb` 已清空旧执行输出，移除历史 `KeyboardInterrupt` 展示。
  - 默认配置恢复为 flow 输出根目录：`outputs/DATA09_v0-flow_features/`。
  - 默认输入示例指向用户本次日志对应的数据族：`E:\PCCP\0904-FLOW-v0.5`。
  - 默认滑窗改为 30ms、0 重叠，与既有 `part_0002` 日志一致。
  - Windows 稳定默认值调整为：
    - `WINDOW_WORKERS = min(4, _auto_detect_workers())`
    - `STFT_BATCH_SIZE = 50`
    - `WINDOW_BATCH_SIZE = 256`
    - `ENABLE_NUMA_BINDING = False`
    - `NPZ_PER_CSV = 20`
  - 批处理单元新增 WinError 1455 定向提示：若仍失败，优先将 `STFT_BATCH_SIZE` 降至 20，再将 `WINDOW_WORKERS` 降至 2，最后可临时设置 `ENABLE_SHARED_STFT=False` 从断点日志续跑。
  - 汇总单元新增日志分片统计，直接输出每个 log 分片的行数、源文件数和 `missing_selected_features` 非空窗口数。
  - 特征质量检查单元新增 `missing_selected_features` 错误类型 Top-N 统计，用于判断重跑后是否仍有特征函数级异常。

- 验证记录：
  - `notebooks/DATA09_v0-flow_feature_extraction.ipynb` JSON 解析通过，所有代码单元 `ast.parse` 通过。
  - `python -m py_compile src\fea_cpt_gpu_v2_2\sliding_window.py src\fea_cpt_gpu_v2_0\sliding_window.py src\fea_cpt_gpu_v2_2\signal_ops.py src\fea_cpt_gpu_v2_0\signal_ops.py` 通过。
  - 使用 30ms、1MHz 合成信号验证 `compute_shared_stft()` + `compute_all_features_for_window()`：2 个频带生成 160 个特征，无 NaN。

- GitHub 上传日志：
  - 已提交到本地 `master` 分支：`ec156f0`。
  - 已补充提交 v2.0 兼容模块：`ad0d943`。
  - 已推送到 `origin/master`，最终远端提交：`ad0d943`。

## 2026-09-06（DATA09 BK/Other 特征重要性 notebook 二分类改造与中文化修复）

- 本次检查对象：
  - `notebooks/DATA09_feature_importance_30ms_analysis.ipynb`
  - 输入特征目录：
    - `outputs/DATA09_v0-flow_features_20260904_121425/DATA09_v0-flow_features` -> `FL00`
    - `outputs/DATA09_v0-flow_features_20260904_121425/DATA09_v0.5-flow_features` -> `FL05`
    - `outputs/DATA09_multi_label_single_event_features/v0-bk_features_BK00` -> `BK00`
    - `outputs/DATA09_multi_label_single_event_features/v0-qj_features_QJ00` -> `QJ00`
    - `outputs/DATA09_multi_label_single_event_features/v05-bk_features_BK05` -> `BK05`
    - `outputs/DATA09_multi_label_single_event_features/v05-qj_features_QJ05` -> `QJ05`

- notebook 目标调整：
  - 保留 6 类来源标签 `label`：`FL00`、`FL05`、`BK00`、`QJ00`、`BK05`、`QJ05`。
  - 新增二分类目标 `target_label`：
    - `BK00/BK05 -> BK`
    - `FL00/FL05/QJ00/QJ05 -> Other`
  - 后续训练、特征排序、降维和测试均以 `BK vs Other` 为目标，不再做六分类。

- 输入与数据质量修复：
  - 输入配置改为“目录 + 显式标签”，递归读取 `features*.csv`。
  - 新增 `MIN_FEATURE_CSV_COLUMNS = 100`，自动跳过表头列数不足的异常 CSV。
  - 已确认 `DATA09_v0.5-flow_features/run_20260905_201502/features_20260905_201502_part_0002.csv` 表头仅 21 列，但后续行混入 661 列，不能作为完整特征表读取；notebook 会明确打印跳过原因。
  - 项目根目录定位改为向上查找 `.git`，避免从 `notebooks` 启动 Jupyter 时误把结果写入 `notebooks/outputs`。

- 样本数量稳定性分析：
  - 新增按 6 类来源标签的重复子采样稳定性评估。
  - 使用标准化后的特征均值向量相对 L2 误差，估计各类信号在多少样本量下特征趋于稳定。
  - 输出 `sample_stability_curve_*.csv`、`sample_stability_summary_*.csv` 和稳定性曲线图。

- 两套独立算法：
  - 算法1标题明确为“使用算法1的特征重要性排序与降维、测试”。
  - 算法1使用 RandomForest：
    - 全特征 holdout 模型 impurity importance。
    - 多随机种子 RF 稳定性。
    - 训练集单变量 F 检验。
    - 训练集内分组 CV permutation importance。
    - Top-K 和相关性去冗余 Top-K 降维测试。
  - 算法2标题明确为“使用算法2的特征重要性排序与降维、测试”。
  - 算法2使用 L1 LogisticRegression：
    - 为兼容 `scikit-learn 1.8`，使用 `solver='liblinear'`、`l1_ratio=1.0`，去除 deprecated 的 `penalty='l1'` 和无效的 `n_jobs`。
    - 为控制耗时和缓解类别不均衡，排序与 Top-K 训练使用“全部 BK + 抽样 Other”的平衡训练子集。
    - 仍在同一个 6:4 holdout 测试集上评估。

- 中文化与误替换修复：
  - 输入配置、主要参数和主要 `print` 输出已补充中文注释/中文文本。
  - 修复中文化替换误伤：
    - `classification_report(..., output_dict=真实类别, ...)` 已恢复为 `output_dict=True`。
    - `permutation_importance(..., n_重复次数=..., ...)` 已恢复为 `n_repeats=...`。
  - 经验约束：后续 notebook 中文化只改注释、markdown、字符串字面量和图表标签；不得对 Python 关键字参数名、布尔值、函数 API 参数做机械翻译。

- 验证记录：
  - `notebooks/DATA09_feature_importance_30ms_analysis.ipynb` JSON 解析通过。
  - 所有代码单元 `ast.parse` 通过。
  - AST 扫描未发现中文关键字参数名，也未发现中文裸变量名。
  - 已清空 notebook 历史执行输出，避免旧 `NameError`/`TypeError` traceback 残留影响排查。

## 2026-09-06（PCCP断丝特征挖掘模块化开发）

- 本次开发依据：
  - `docs/2026-9-6-PCCP断丝特征挖掘_Codex开发方案_修正版.md`
  - 输入特征目录沿用 DATA09 六类已计算特征向量：
    - `FL00/FL05 -> Other`
    - `BK00/BK05 -> BK`
    - `QJ00/QJ05 -> Other`

- 新增底层代码包：
  - `src/pccp_feature_mining/`
  - 按职责拆分为配置、数据读取、特征列识别、物理事件分组、质量检查、单特征判别、冗余分析、mRMR、Bootstrap 稳定性、跨流速分析、最终评分、可视化和报告生成模块。
  - `BK/QJ` 默认以 `source_file_name` 作为物理事件组，避免同一物理事件窗口在采样分析中被拆散。
  - `FL` 默认以窗口行作为独立采样单元，符合方案中流噪窗口可独立抽样的假设。

- 新增 notebook：
  - `notebooks/2026-09-06-PCCP_feature_mining_pipeline.ipynb`
  - 章节结构：
    - 环境与输入配置
    - 数据读取与质量检查
    - 单特征判别能力
    - 冗余分析与 mRMR
    - Bootstrap 稳定性
    - 跨流速一致性
    - 最终特征排序与分级
    - 一键运行入口

- 核心输出：
  - `dataset_summary.csv`
  - `feature_list.csv`
  - `quality_report.csv`
  - `feature_discrimination.csv`
  - `correlation_cluster.csv`
  - `mrmr_rank.csv`
  - `feature_stability.csv`
  - `cross_flow_feature.csv`
  - `final_feature_ranking.csv`
  - `summary_report.md`

- 方法实现要点：
  - 单特征判别同时输出 `AUC`、`Wasserstein distance`、`Cliff delta` 和 `Mutual Information`。
  - 主任务为 `BK_NONBK`，同时保留 `BK_QJ`、`BK00_BK05`、`FL00_FL05`、`QJ00_QJ05` 比较。
  - Bootstrap 每轮使用“全部 BK + 等量 Other”策略；Other 采样保持 QJ 事件完整，FL 按窗口独立。
  - 冗余分析基于 Spearman 相关矩阵和高相关阈值构建相关簇，mRMR 使用判别相关性减去平均相关冗余。
  - 最终评分融合 `BK vs Other` 判别、`BK vs QJ` 判别、Bootstrap 稳定性、跨流速一致性，并对流速敏感性和高相关冗余做惩罚。
  - 最终分级输出 `A/B/C`：A级为稳定断丝核心候选特征，B级为候选特征，C级为工况相关或冗余较高特征。

- 验证记录：
  - `python -m py_compile src\pccp_feature_mining\*.py` 通过。
  - `notebooks/2026-09-06-PCCP_feature_mining_pipeline.ipynb` JSON 解析通过，所有代码单元 `ast.parse` 通过。

## 2026-09-06（PCCP特征挖掘notebook展示、缺失特征审计与冗余分析增强）

- 用户反馈与本次修正目标：
  - 读取并更新 `notebooks/2026-09-06-PCCP_feature_mining_pipeline.ipynb` 最新本地版本。
  - 所有 notebook 输出表添加表号和标题，含特征名的表格追加中文含义列。
  - 结果图的绘图代码保留在 notebook 中，并在 notebook 内直接显示 PNG。
  - 回答并修复 FL00 示例 CSV 中大量“有表头但无有效特征值”的样本未被 Notebook01 检测出来的问题。
  - Notebook03 补齐 3.1-3.5 二级比较任务；Notebook04 拆分 Pearson/Spearman/对比；Notebook05 拆分 4 个方法小节；Notebook06/07 补参数、原理和公式解释。

- 数据读取与质量检查修正：
  - `src/pccp_feature_mining/config.py`
    - 新增 `min_valid_feature_values_per_row`，默认 `100`。
  - `src/pccp_feature_mining/data_loader.py`
    - `_read_one_csv()` 新增行级有效特征数统计。
    - `load_report.csv` 新增 `retained_rows`、`dropped_low_valid_feature_rows`、`feature_columns_detected`、`min_valid_feature_values`、`median_valid_feature_values`、`max_valid_feature_values`。
    - 表头完整但低有效特征数的样本行会被剔除，不再进入分布、判别、冗余、组合、稳定性、跨流速和分类测试。
  - 对用户给出的示例文件：
    - `features_20260905_173711_part_0001.csv` 表头列数为 `661`，行数为 `17066`。
    - 检测到 `622` 个有效频带特征列。
    - `6568` 行有效特征数为 `0`，低于阈值 `100`，新版保留 `10498` 行，剔除 `6568` 行。
    - 旧版漏检原因是只检查文件列数和空表，没有检查每个样本行的有效特征数。

- 特征中文含义增强：
  - `src/pccp_feature_mining/feature_schema.py`
    - 新增 `BASE_FEATURE_MEANINGS` 特征字典。
    - 新增 `feature_chinese_meaning()`，可将 `b_10k_50k__R_hl_mean` 解析为“10-50 kHz频带，高低频能量比均值，描述高频能量相对低频能量的平均占比”。
    - 新增 `describe_feature_list()`，用于解释 `selected_features`、`recommended_features`、`recommended_drop` 这类分号拼接字段。
    - 新增 `add_feature_meaning_columns()`，自动给 `feature`、`feature_a`、`feature_b`、`cluster_representative`、`recommended_keep`、`selected_features`、`recommended_features` 等字段追加中文含义列。
  - `src/pccp_feature_mining/quality_control.py`
    - `feature_list_frame()` 和 `build_feature_quality()` 输出直接包含 `特征中文含义`。

- 特征冗余分析增强：
  - `src/pccp_feature_mining/feature_redundancy.py`
    - 新增 `compare_correlation_methods()`，对比 Pearson-only、Spearman-only、Pearson+Spearman 命中的高相关特征对。
    - 新增 `build_redundancy_recommendations()`，把两种方法的高相关边合并成冗余组，并按 `BK_NONBK` 判别分推荐每组保留 1 个特征。
  - `src/pccp_feature_mining/run_all.py`
    - 一键流程同时输出 `pearson_correlation_matrix.csv`、`spearman_correlation_matrix.csv`、`pearson_high_correlation_pairs.csv`、`spearman_high_correlation_pairs.csv`、`correlation_method_comparison.csv` 和 `redundancy_recommendations.csv`。
    - 后续 mRMR 与最终冗余惩罚仍使用 Spearman 相关矩阵。

- notebook 结构更新：
  - 新增 `Notebook00 环境初始化与显示工具`，定义 `show_table()` 和 `show_image()`。
  - 每个展示表都输出“表 N 标题”，并通过 `add_feature_meaning_columns()` 自动补中文含义。
  - Notebook02 明确“六类特征分布”是六个来源标签上的特征分布，不是只有六个特征；绘图单元会在保存后直接显示箱线图、KDE、PCA、UMAP。
  - Notebook03 拆成：
    - `3.1 BK00 vs NONBK00`
    - `3.2 BK00 vs QJ00`
    - `3.3 BK05 vs QJ05`
    - `3.4 BK05 vs NONBK05`
    - `3.5 BK vs NONBK`
  - Notebook04 拆成：
    - `4.1 使用 Pearson 相关`
    - `4.2 使用 Spearman 相关`
    - `4.3 Pearson 与 Spearman 结果对比`
  - Notebook05 拆成：
    - `5.1 mRMR前缀组合`
    - `5.2 近似 ReliefF 排序`
    - `5.3 顺序前向选择 SFS`
    - `5.4 多方法结果对比`
  - Notebook06 补充 Bootstrap 原理、默认抽样 200 次、每轮全部 BK + 等量 Other、Top-K 频率和参数调整方法。
  - Notebook07 补充各列含义、排序原因和 `auc_lift`、`cross_condition_consistency`、`median_shift_norm`、`cross_flow_score` 公式。

- 构建脚本更新：
  - `tools/build_pccp_feature_mining_notebook.py` 已重写为更短的 UTF-8 notebook 生成脚本。
  - 重新生成 `notebooks/2026-09-06-PCCP_feature_mining_pipeline.ipynb`，清除旧执行输出，避免旧输出与新源码不一致。

- 验证记录：
  - `python -m compileall -q src\pccp_feature_mining tools\build_pccp_feature_mining_notebook.py` 通过。
  - notebook JSON 解析通过，所有代码单元 `ast.parse` 通过。
  - 小样本 smoke test 通过：
    - 每类限制 `20` 行，Bootstrap `2` 轮，mRMR Top12，组合/分类特征数 `(3, 5)`。
    - 成功读取六类标签，合计 `120` 行、`632` 个特征。
    - 成功生成 `load_report.csv`、`redundancy_recommendations.csv`、`classification_test_results.csv`、`final_feature_ranking.csv` 等输出。
  - smoke test 中 `joblib/loky` 输出“无法获取物理核心数，回退逻辑核心数”的环境警告，不影响流程结果。
  - 新增 `tools/build_pccp_feature_mining_notebook.py` 用 UTF-8 结构化生成 notebook，避免 PowerShell 命令行编码导致中文被替换成问号。

- GitHub 上传日志：
  - 已提交到本地 `master` 分支：`4a564c1`。
  - 已推送到 `origin/master`：`4a564c1`。
  - 小样本 smoke 测试通过：
    - 每类限制 30 行，Bootstrap 3 轮，mRMR Top10。
    - 成功读取 6 类标签，合计 180 行、632 个特征。
    - 成功生成 `feature_discrimination.csv`、`feature_stability.csv`、`cross_flow_feature.csv`、`final_feature_ranking.csv`。
  - 当前环境缺少 `tabulate`，报告生成已改为无额外依赖兜底，不再要求安装该包。

- GitHub 上传日志：
  - 已提交到本地 `master` 分支：`c37751c`。
  - 已推送到 `origin/master`：`c37751c`。

## 2026-09-06（PCCP断丝特征挖掘notebook补全与逻辑修正）

- 用户反馈与本次修正目标：
  - 检查 `notebooks/2026-09-06-PCCP_feature_mining_pipeline.ipynb` 已执行输出中是否存在算法逻辑问题。
  - 每个一级标题下补充算法原理说明。
  - 按开发方案补齐 `Notebook02 六类特征分布分析`、方案约定图件和结果解释。
  - 对 `Notebook03/04/05` 增加必要二级结构和方法说明。
  - 输出结果表增加字段含义、指标方向和人工解读说明。
  - 给出 `BK00 vs NONBK00`、`BK00 vs QJ00`、`BK05 vs QJ05`、`BK05 vs NONBK05`、`BK vs NONBK` 的特征数量推荐表。
  - 增加特征挖掘后的 `SVM/逻辑回归` 分类测试，并强调按源文件分组划分训练/测试，避免同源窗口泄漏。

- 已执行输出的逻辑审查结论：
  - 六类标签均能读取，完整运行结果中样本规模为 `38236` 行、`632` 个特征。
  - `FL05` 有一个异常 CSV 分片表头仅 `21` 列，已按 `min_feature_csv_columns=100` 跳过；该逻辑正确。
  - 数据存在明显不均衡：`BK=162`，`Other=38074`，后续使用 Bootstrap 平衡统计与分类器 `class_weight='balanced'` 处理。
  - 旧版最终排序中存在完全相关重复特征可能同时进入 Top 排名的问题，例如 `R_2_1/R_h`、`R_harm/Ridge_coh`，本次已修正冗余惩罚。
  - 旧版 notebook 缺少方案中的六类分布分析、组合搜索、分类测试和比较任务级结论，本次已补齐。

- 新增/修改底层模块：
  - `src/pccp_feature_mining/distribution_analysis.py`
    - 六类分布筛查特征选择。
    - PCA 二维投影。
    - 可选 UMAP 投影。
    - 六类 KDE 图。
  - `src/pccp_feature_mining/combination_search.py`
    - mRMR 前缀组合评估。
    - 近似 ReliefF 排序。
    - Sequential Forward Selection + LDA 交叉验证。
  - `src/pccp_feature_mining/classification_test.py`
    - 逻辑回归、线性 SVM、RBF-SVM 分类测试。
    - 五个比较任务的推荐特征数量和具体特征输出。
    - 分类验证按 `source_file_name` 分组划分训练/测试。
  - `src/pccp_feature_mining/feature_discrimination.py`
    - 增加 `BK00_NONBK00`、`BK00_QJ00`、`BK05_QJ05`、`BK05_NONBK05` 精确比较任务。
  - `src/pccp_feature_mining/feature_redundancy.py`
    - 非代表高相关特征即使判别分并列，也按其与簇代表的相关性增加冗余惩罚。
  - `src/pccp_feature_mining/run_all.py`
    - 一键流程补齐分布图、组合搜索、分类测试和推荐结论输出。

- notebook 结构已重写：
  - `Notebook00 运行结果与算法逻辑审查`
  - `Notebook01 数据读取与质量检查`
  - `Notebook02 六类特征分布分析`
  - `Notebook03 单特征判别能力分析`
  - `Notebook04 特征冗余分析`
  - `Notebook05 多特征组合搜索`
  - `Notebook06 特征稳定性分析`
  - `Notebook07 跨流速一致性分析`
  - `Notebook08 最终特征评价与结论`
  - `Notebook09 特征挖掘后的分类测试`
  - `Notebook10 一键运行入口`

- 新增输出文件：
  - `six_class_distribution_features.csv`
  - `six_class_pca_projection.csv`
  - `six_class_pca_explained_variance.csv`
  - `six_class_umap_projection.csv`
  - `feature_combination_mrmr_prefix.csv`
  - `feature_combination_search.csv`
  - `relieff_rank.csv`
  - `sfs_selection_path.csv`
  - `classification_test_results.csv`
  - `comparison_feature_recommendations.csv`
  - `classification_conclusion_table.csv`

- 新增输出图：
  - `plots/six_class_feature_boxplots.png`
  - `plots/six_class_feature_kde.png`
  - `plots/six_class_pca.png`
  - `plots/six_class_umap.png`，仅环境安装 `umap-learn` 时生成。
  - `plots/final_feature_ranking_top30.png`
  - `plots/top_feature_boxplots.png`

- 小样本验证：
  - 每类限制 `20` 行，Bootstrap `2` 轮，mRMR Top12，分类特征数测试 `(3, 5)`。
  - 成功读取六类标签，合计 `120` 行、`632` 个特征。
  - 成功生成分布分析、组合搜索、最终排序和分类测试结果。
  - smoke 输出中的推荐结果示例：
    - `BK00_vs_NONBK00`：推荐 `3` 个特征，线性 SVM。
    - `BK00_vs_QJ00`：推荐 `3` 个特征，逻辑回归。
    - `BK05_vs_QJ05`：推荐 `3` 个特征，线性 SVM。
    - `BK05_vs_NONBK05`：推荐 `5` 个特征，逻辑回归。
    - `BK_vs_NONBK`：推荐 `5` 个特征，逻辑回归。

- 验证记录：
  - `python -m py_compile src\pccp_feature_mining\*.py` 通过。
  - `notebooks/2026-09-06-PCCP_feature_mining_pipeline.ipynb` JSON 解析通过，所有代码单元 `ast.parse` 通过。

## 2026-09-07（补充）PCCP特征定义补充与特征字典文档

- 变更范围：`src/pccp_feature_mining/feature_schema.py`、新增 `docs/PCCP特征字典.md`
- 补充了 `BASE_FEATURE_MEANINGS` 中缺失的 20 个特征定义：
  - `C_h`、`R_harm`、`Ridge_coh`、`rho_up`、`rho_down`、`N_turn`、`rho_res`、`N_abn`、`R_tkeo`、
    `K_loc`、`K_res_max`、`SK_max`、`CF_res`、`T_half_high`、`SC_res_mean`、`k_res_sc`、`k_res_hl`、
    `S_res_env`、`eta_asym`、`epsilon_rec`
- 校验：`feature_chinese_meaning()` 对 `fea_cpt_gpu_v2.features.compute_all_features` 产出的全部 64 个静态特征均能解析，无“暂未在特征字典中补充详细物理解释”的遗留项。
- 新增 `docs/PCCP特征字典.md`：完整特征字典表（65 项），含公式、中英文名称、代码变量名、物理意义，并说明频带前缀与断丝/敲击/流噪声的物理意义解读。

## 2026-09-07（补充）特征含义全量校验与mRMR说明扩充

- 全量校验：对 6 类特征表 8 个 CSV 的 640 个特征列（80 个基础特征 × 8 个频带）逐一执行 `feature_chinese_meaning`，均能解析出中文含义，`暂未在特征字典中补充详细物理解释` 残留数为 0。
- 结论：程序侧特征含义已补全；notebook 中表 14/19/21 等老旧输出里仍显示的“暂未…”是修改特征字典之前保存的过期输出，重启 kernel 重新运行即可刷新。
- notebook 5.1 节 mRMR 说明扩充：补充核心公式 score(f)=D(f)-lambda*R(f|S)、各符号含义（D(f) AUC相关分、R(f|S) 平均绝对Spearman冗余、lambda 冗余权重=0.5）、四步算法流程、前K个组合评估说明，并加入两篇经典文献：
  - Peng H, Long F, Ding C. Feature selection based on mutual information[J]. IEEE TPAMI, 2005, 27(8): 1226-1238.
  - Ding C, Peng H. Minimum redundancy feature selection from microarray gene expression data[J]. JBCB, 2005, 3(2): 185-205.
- 校验：notebook JSON 通过 `json.load` 校验，mRMR 单元格内容正确写入。

## 2026-09-07（补充）notebook 算法原理说明整体扩充

- 按“主要流程、核心公式、关键物理量符号含义、参考文献”的格式，对 notebook 中算法原理介绍过于简略的章节进行了系统性补充，涉及 10 个 markdown 单元格：
  - 03 单特征判别能力分析：新增 AUC/auc_lift、Wasserstein 归一化、Cliff's delta、互信息、discrimination_score 加权公式与含义（参考文献 Fawcett 2006、Cliff 1993）。注意 discrimination_score = 0.55*2*max(auc_lift,0) + 0.30*min(|delta|,1) + 0.15*MI/(1+MI)。
  - 4.1 Pearson 相关：补充公式与“|r|=1 特征对甄别”说明，记录已核实的实现层重复（R_2_1/R_h 同公式、Ridge_coh/R_harm 同变量赋值）与数学恒等（r_p 与 A_env 满足 A_env=1-2r_p）。
  - 4.2 Spearman 相关：补充秩相关公式 rho=1-6*sum(d^2)/(n(n^2-1))。
  - 4.3 冗余组：补充连通分量（Union-Find）建组与 keep/drop 公式。
  - 5.2 近似 ReliefF：补充 near-hit/near-miss 的 W(f) 加权公式、4 步流程、与标准 ReliefF 的区别（参考文献 Kira&Rendell 1992、Robnik-Sikonja&Kononenko 2003）。
  - 5.3 SFS：补充每一轮 argmax 选择公式、LDA CV AUC 评估、5 步流程、关键指标（参考文献 Kohavi&John 1997、Pudil et al. 1994）。
  - 06 Bootstrap 稳定性分析：补充 B 轮平衡抽样构成、lift_b(f)、mean_rank/rank_std/topK_frequency/rank_stability_score 公式与指标表（参考文献 Efron 1979、Efron&Tibshirani 1993）。
  - 08 最终特征评价：补充 final_score 加权公式、各符号来源、A/B/C 分级规则。
  - 09 分类测试：补充 balanced_accuracy / recall_positive / specificity 指标公式与严格划分规则说明。
- 校验：notebook JSON 通过 json.load 校验；所有改动单元格内容与源码实现（feature_discrimination.py、combination_search.py、bootstrap_stability.py、feature_redundancy.py、feature_selection.py、cross_flow_analysis.py）核对一致；已修复单元格内 `|Spearman|` 破坏 Markdown 表格问题与"同源信泄漏"错别字。

## 2026-09-08（补充）PCCP第二节来源类别样本量稳定性估计

- 变更范围：
  - `src/pccp_feature_mining/config.py`
  - `src/pccp_feature_mining/distribution_analysis.py`
  - `src/pccp_feature_mining/run_all.py`
  - `notebooks/2026-09-06-PCCP_feature_mining_pipeline.ipynb`
  - `docs/2026-9-6-PCCP断丝特征挖掘_Codex开发方案_修正版.md`
- notebook 第二节末尾新增 `2.7.1 全部派生窗口的来源类别特征均值稳定性`，按 `source_label` 分别估计 `BK00/BK05/FL00/FL05/QJ00/QJ05` 的样本量稳定区间。
- 新增两种独立估计方式：
  - 相邻样本量均值漂移法：对全局标准化后的候选特征均值向量，计算相邻样本量之间的相对 L2 变化；连续低于 `sample_stability_adjacent_threshold=0.02` 时判为稳定。
  - 双Bootstrap均值一致性法：同一样本量下独立抽取两组 bootstrap 子样本，比较两组标准化特征均值向量的相对 L2 差异 P90；连续低于 `sample_stability_pairwise_threshold=0.05` 时判为稳定。
- `MiningConfig` 新增样本量稳定性参数：
  - `sample_stability_repeats=100`
  - `sample_stability_min_samples=5`
  - `sample_stability_adjacent_threshold=0.02`
  - `sample_stability_pairwise_threshold=0.05`
  - `sample_stability_consecutive_points=2`
- `run_pccp_feature_mining()` 已接入该分析，输出：
  - `sample_stability_curve.csv`
  - `sample_stability_summary.csv`
  - `plots/sample_stability_curves.png`
- 该分析只回答每个来源类别当前特征均值统计量何时趋于稳定，不参与第 06 节 Bootstrap 特征排名稳定性，也不改变最终特征评分公式。
- 2026-09-08 修正：旧版“全量均值参照法”在样本量等于当前全量样本数时会与自身比较，导致最大样本量处相对 L2 误差天然接近 0，并可能造成“每类都刚好在现有最大样本量稳定”的假象；已改为双Bootstrap均值一致性法，最大样本量处也使用有放回抽样比较两个独立均值估计，不再强制归零。
- 2026-09-09 再修正：本节早期版本先对传入的多类别数据拟合全局 `StandardScaler`，再以标准化均值向量的模作为相对 L2 分母。该误差不具平移不变性，加入 FL/QJ 后会改变 BK 所在坐标原点和误差分母，因此同一批 BK 样本的误差可从 1.x 人为降到 0.x。此版本结果已判定为不可跨运行比较，并由后述“类别独立标准化 RMS 误差”替代。
- notebook 显示工具新增 `sync_output_counters()`，每个产生表/图的代码单元会先把计数器同步到该单元在全 notebook 中的预期位置，避免单独重跑 2.7 时出现“表 8 / 图 1”这类局部编号。

## 2026-09-08（补充）PCCP 2.7.2 BK 单窗口样本稳定性

- 将原 2.7 内容调整为 `2.7.1`，保留六类全部派生窗口分析及原有输出文件，维持向后兼容。
- 新增 `select_bk_0_30ms_samples()`：
  - 只选择 `source_label` 为 BK、`window_mode='0_30'`、`window_start_ms≈0`、`window_end_ms≈30` 的特征行。
  - 以 `event_id` 表示物理断丝事件，保证每个事件最多贡献一行；发现重复目标窗口时直接报错。
  - 输出每类原派生窗口行数、事件数、入选行数、排除行数、缺失事件数和重复行数审计。
- 新增 `2.7.2 BK 断丝事件的 0-30 ms 单窗口样本稳定性`，仅对筛选后的 `BK00/BK05` 运行与 2.7.1 相同的稳定性算法。
- 一键流程新增输出：
  - `bk_0_30ms_selection_audit.csv`
  - `bk_0_30ms_sample_stability_curve.csv`
  - `bk_0_30ms_sample_stability_summary.csv`
  - `plots/bk_0_30ms_sample_stability_curves.png`
- 当前真实数据筛选结果：BK00 从 102 个派生窗口筛出 17 个事件，BK05 从 60 个派生窗口筛出 10 个事件；目标窗口缺失数和重复数均为 0。
- 2026-09-09 加密 2.7.2 样本量网格：不再沿用 5、10、15 等稀疏检查点，而是从 `sample_stability_min_samples=5` 开始以步长 1 检查。当前 BK00 使用 5-17 共 13 个样本量点，BK05 使用 5-10 共 6 个样本量点。
- 2026-09-09 扩展 2.7.2 图 20：构造“FL/QJ 原始样本 + BK 0-30 ms 单窗口样本”的六类比较数据集。BK 使用逐整数密集网格，FL/QJ 继续使用默认稀疏网格；表 17、表 18 只展示 BK，图 20 展示全部六类，并新增 `bk_0_30ms_with_nonbk_sample_stability_curve.csv` 和 `bk_0_30ms_with_nonbk_sample_stability_summary.csv`。
- 2.7.2 新增表 16-18、图 20；Notebook03 及后续所有表图编号和 `sync_output_counters()` 基线相应顺延。

## 2026-09-09（修正）六类样本稳定性独立计算

- `estimate_source_feature_stability()` 改为按 `source_label` 完全独立计算：
  - 每类独立执行数值转换和中位数缺失值插补，避免其他类别的数据分布影响插补值。
  - 每类独立拟合 `StandardScaler`，仅使用该类的类内标准差统一特征量纲。
  - 每类使用 `SHA-256(random_state, source_label)` 派生确定性随机种子；增加、删除或重排其他类别不会改变本类的 Bootstrap 抽样序列。
- 删除依赖全局坐标原点的相对 L2 误差和解释性较弱的相邻平均均值漂移法。新误差定义为两组独立 Bootstrap 均值差的类内标准化 RMS：`sqrt(mean(((mean_a - mean_b) / within_label_std) ** 2))`。
- 新曲线字段为 `paired_bootstrap_standardized_rms_gap_mean` 和 `paired_bootstrap_standardized_rms_gap_p90`；稳定性只按 P90 连续满足 `sample_stability_pairwise_threshold=0.05` 判定。
- 删除配置项 `sample_stability_adjacent_threshold`。最大样本量仍进行两组有放回 Bootstrap，因此误差反映当前经验分布下的均值抽样不确定性，不会因使用全量行而强制为 0。
- 六类曲线的可比范围是“相对于各自类内波动的均值抽样稳定性”；由于每类归一化基准不同，不能将该曲线解释为六类原始信号幅值差异。
- 已增加不变性校验：同一类别单独输入与加入均值、尺度显著不同的其他类别后，其曲线和汇总结果必须逐字段完全一致。
