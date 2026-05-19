# DAS近场断丝声源定位算法 Python 开发方案（两Notebook一次性开发版）

---

# 1. 目标与硬约束

本方案用于 DAS 断丝单事件近场定位，采用 TDOA 思路，一次性完成开发并可直接在给定样例数据上测试。

硬约束：

1. notebook 数量固定为 2 个
2. Python 模块存放到：`src/das_localization`
3. 输出路径固定为：`outputs/das_localization`
4. 示例数据：`E:\codes\ZZ-BK\data\eDAS_loc\eDAS-2000Hz-0032pt-20260323T110032.578.bin`
5. 读取逻辑参照：`E:\codes\eDASread\singledataread\phase_bin_tools.py`
6. 采样与几何默认值（可调）：
   - 通道数约 32
   - 通道间距 `3.2 m`
   - 采样率 `2000 Hz`
   - 预处理默认先做 `100 Hz` 高通滤波
7. 默认单事件分析窗口
8. 全部中文注释必须为 UTF-8 编码，保证 VSCode 打开不乱码

---

# 2. 模型与主流程

## 2.1 TDOA 建模

阵列位置：`x_i = i * dx`。

TOA 推导模型：

` t_i = t0 + (1/v) * sqrt((x_i - x_s)^2 + z_s^2) `

实际使用 TDOA：选参考通道 `r`，

` Δt_i = t_i - t_r = (1/v) * [sqrt((x_i-x_s)^2+z_s^2) - sqrt((x_r-x_s)^2+z_s^2)] `

说明：

- 由于绝对事件时刻未知，仅使用 TDOA
- `x_s` 为主目标，`z_s` 为次目标

## 2.2 工程主链路

```text
读取bin并转相位(rad)
-> 预处理(去均值/高通/可选带通/归一化)
-> 参考通道选择(best_snr vs fixed)
-> 时延估计(GCC-PHAT/NCC/FFT相关)
-> 峰值亚采样插值
-> PSR+物理时延窗筛选
-> TDOA曲线拟合(抛物线/双曲线)
-> 最小二乘/鲁棒最小二乘/RANSAC
-> 置信度与状态判决
-> 输出结果与日志
```

---

# 3. 目录与文件约定

```text
ZZ-BK/
├── docs/
├── notebooks/
│   ├── 2026-05-18-das-loc-notebook1-dev.ipynb
│   └── 2026-05-18-das-loc-notebook2-ablation.ipynb
├── outputs/
│   └── das_localization/
│       ├── figures/
│       ├── metrics/
│       ├── logs/
│       └── intermediate/
└── src/
    └── das_localization/
        ├── __init__.py
        ├── io_utils.py
        ├── preprocess.py
        ├── tdoa.py
        ├── fitting.py
        ├── localization.py
        ├── evaluation.py
        └── batch.py
```

约束：

- 绘图主逻辑写在 notebook，不写在 `src`
- `src` 只做算法与数据处理
- 参数在 notebook 顶部统一管理

---

# 4. Notebook 设计

## 4.1 notebook1：`2026-05-18-das-loc-notebook1-dev.ipynb`

用途：单样本开发 + 批处理串联开发（主工程 notebook）。

### 结构（按你指定落地）

1. 初始化
   - 1.1 导入库与函数
   - 1.2 参数区（路径、采样率、通道间距、滤波参数、拟合参数、阈值）
2. 数据加载与查看
   - 2.1 读取数据
   - 2.2 数据预处理
   - 2.3 绘图查看（时域、频域、空间-时间图）
3. 参考通道选择
   - `best_snr` vs `fixed` 对比
4. GCC-PHAT 计算与峰值提取
   - 亚采样插值验证
5. PSR 与物理时延窗筛选
   - 产出 `tdoa + quality metrics`
6. 读取 `tdoa + quality metrics` 并可视化与可辨识性检查
7. 定位求解
   - 网格粗搜索初始化
   - `least_squares` 受约束精修
   - 可选 `RANSAC / robust loss` 对比
   - 置信度计算与状态判决（`ok/low_confidence/rejected`）
8. 误差分析（单样本）
9. 编写批处理函数
   - 串联调用
   - 批处理运行与日志保存
   - 批样本误差统计

### notebook1 必须输出

- 原始/预处理后波形图
- 多通道频谱对比图
- 参考通道对比结果
- 相关曲线与峰值标注
- TDOA 与质量指标表
- 拟合曲线与残差图
- 单样本定位结果字典
- 批处理日志与统计表

## 4.2 notebook2：`2026-05-18-das-loc-notebook2-ablation.ipynb`

用途：消融、鲁棒性、参数敏感性评估（论文/报告 notebook）。

### 结构

1. 参数区
2. 消融试验1：时延估计方法对比
   - NCC
   - GCC-PHAT
   - FFT互相关
3. 消融试验2：拟合模型对比
   - 双曲线
   - 抛物线
4. 消融试验3：拟合求解器对比
   - 普通最小二乘（`linear` / `lm`）
   - 鲁棒最小二乘（`huber` / `soft_l1` / `cauchy`）
5. 消融试验4：是否启用 RANSAC
6. 噪声鲁棒性
   - 添加高斯噪声（多档 SNR）
7. 参数敏感性分析
   - `PSR_TH`, `MAX_TAU`, `VELOCITY`, `Z_BOUNDS`, `REF_MODE`
8. 汇总图表与结论

### notebook2 必须输出

- 各消融维度指标对比图（误差、成功率、拒识率、RMSE）
- 噪声等级性能曲线
- 参数敏感性热图/折线图
- 推荐参数组合结论表

---

# 5. src/das_localization 模块设计

## 5.1 `io_utils.py`

- 解析文件名中的 `Hz` 和 `pt`
- 读取 `.bin` 为 `int32`
- 转换为相位弧度：`phase = int32 / 32767 * pi`
- 输出形状：`(frames, points)`

## 5.2 `preprocess.py`

- 去均值
- 高通滤波（默认 100Hz，可调）
- 可选带通
- 通道归一化

## 5.3 `tdoa.py`

- 参考通道选择（best_snr/fixed）
- NCC / GCC-PHAT / FFT相关三种时延估计
- 峰值提取与抛物线亚采样插值
- PSR 计算
- 物理时延窗筛选
- 输出 `tdoa` 与质量指标

## 5.4 `fitting.py`

- 抛物线模型拟合
- 双曲线 TDOA 模型拟合
- 网格粗搜索初始化
- 最小二乘/鲁棒最小二乘
- 可选 RANSAC

## 5.5 `localization.py`

- 单样本定位主函数
- 置信度计算
- 状态判决（ok/low_confidence/rejected）

## 5.6 `evaluation.py`

- 指标计算：位置误差、TDOA RMSE、成功率、拒识率
- 消融结果整理成 DataFrame

## 5.7 `batch.py`

- 批处理串联调用
- 结果落盘到 `outputs/das_localization`

---

# 6. 参数区统一模板（两个 notebook 顶部）

```python
# 路径
DATA_FILE = r"E:\codes\ZZ-BK\data\eDAS_loc\eDAS-2000Hz-0032pt-20260323T110032.578.bin"
OUTPUT_ROOT = r"E:\codes\ZZ-BK\outputs\das_localization"

# 采样与几何
FS = 2000.0
DX = 3.2
N_CHANNELS = 32

# 预处理
DO_DEMEAN = True
DO_HIGHPASS = True
HP_CUTOFF = 100.0
HP_ORDER = 2
DO_BANDPASS = False
BP_LOWCUT = 120.0
BP_HIGHCUT = 800.0
BP_ORDER = 4
NORMALIZE = True

# TDOA
REF_MODE = "best_snr"  # best_snr | fixed
REF_CH = 15
DELAY_METHOD = "gcc_phat"  # gcc_phat | ncc | fft_xcorr
MAX_TAU_SEC = 0.02
EPS = 1e-12

# 质量筛选
PSR_TH = 6.0
N_VALID_MIN = 8

# 拟合
FIT_MODEL = "hyperbola"  # hyperbola | parabola
FIT_SOLVER = "least_squares"  # least_squares | robust
ROBUST_LOSS = "huber"  # huber | soft_l1 | cauchy
USE_RANSAC = True
RANSAC_ITERS = 200
RANSAC_RESID_TH = 0.0008

# 物理参数与边界
VELOCITY = 343.0
X_BOUNDS = (-20.0, 120.0)
Z_BOUNDS = (0.1, 5.0)

# 判决
RMSE_TH = 0.0010
CONF_TH = 0.60
```

---

# 7. 输出约定

所有输出统一写入 `outputs/das_localization`：

- `intermediate/`：tdoa、质量指标、缓存npy/csv
- `figures/`：单样本图、消融图、噪声鲁棒性图
- `metrics/`：批处理统计与消融结果csv
- `logs/`：运行日志与参数快照

---

# 8. 编码与注释规范（强制）

1. 所有 `.py/.ipynb/.md/.csv` 文本统一 UTF-8 编码
2. 关键函数、关键步骤、关键公式都写中文注释
3. 注释要求“说明为什么这样做”，不是仅重复代码
4. notebook 的每个主要单元格都包含中文说明（Markdown 或行内注释）

---

# 9. 开发与测试执行顺序

1. 完善并冻结本 md 方案
2. 在 `src/das_localization` 完成模块化开发
3. 生成两个 notebook 并打通调用
4. 使用给定 `.bin` 样例完成 notebook1 单样本与批处理测试
5. 在 notebook2 完成消融、噪声鲁棒性、参数敏感性测试
6. 输出图表与指标到 `outputs/das_localization`

---

# 10. 验收标准

1. 两个 notebook 都可独立运行
2. notebook1 完成单样本全流程与批处理串联
3. notebook2 覆盖指定消融项：
   - 时延估计方式
   - 双曲线/抛物线
   - 是否RANSAC
   - 最小二乘与鲁棒拟合方式
4. 所有结果落盘到 `outputs/das_localization`
5. 注释完整，UTF-8 无乱码


- ?????????

### notebook1 ?4????2026-05-19?

4.1 Cross-correlation and peak extraction?
- ??????????????????cross-correlation??????? `[-MAX_TAU_SEC, +MAX_TAU_SEC]`?
- ??????????????? `peak_tau_all` ? `peak_val_all`?
- ????????? `tau=0`?
- ???? `estimate_delays(...)` ???????????????????

?????QC??
- ???????? `proc` ????? `raw_snr_proxy = q90(power) / q10(power)`?
- ??????`snr_th = quantile(raw_snr_proxy, 0.60)`?`peak_th = quantile(peak_val_all, 0.60)`?
- ??????`abs(peak_tau_all) <= MAX_TAU_SEC`?
- ????????`psr >= PSR_TH`?
- ?????`quality_mask = snr_pass & peak_pass & psr_pass & physical_pass`???? `quality_mask[ref_idx] = True`?
- ? `quality_mask=True` ????????????????????????? `NaN`?

4.2 Full-channel cross-correlation visualization?
- ????????????????
- ?????????global scale???????????????
- ??????????????????

4.3 Cross-correlation heatmap?
- ????????????????????
- ?????????????????????????

4.4 Full-channel auto-correlation visualization?
- ???????????????????????????/?????
- ? 4.2 ????????????????????????

?????
- ??`nb1_42_xcorr_vertical_curves.png`?`nb1_43_xcorr_heatmap.png`?`nb1_44_acorr_vertical_curves.png`
- ??`intermediate/nb1_xcorr_peaks.csv`?? `raw_snr_proxy`?`snr_pass`?`peak_pass`?`psr_pass`?`physical_pass`?`quality_pass` ????
