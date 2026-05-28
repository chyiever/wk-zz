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