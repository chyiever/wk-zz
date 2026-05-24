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

