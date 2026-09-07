# PCCP 断丝识别特征字典

本文档是 PCCP 断丝信号特征挖掘所用候选特征的完整字典表，包含公式、中英文名称、代码变量名与物理意义。
代码实现见 `src/fea_cpt_gpu_v2/features.py`，中文含义映射见 `src/pccp_feature_mining/feature_schema.py` 的 `BASE_FEATURE_MEANINGS`。

## 符号约定

- $x[n]$：预处理后的带通信号（默认 5–60 kHz），$x_{\text{res}}[n]$：谐波模型剥离后的残差信号
- $e[n]$：包络 $e[n]=|\mathrm{hilbert}(s[n])|$，$s[n]$ 取 $x_B,x_{H1},x_{H2}$ 或 $x_{\text{res}}$
- $f_1(t)$：主脊线频率轨迹；$f_2(t)$：二倍频脊线频率轨迹
- $P(t,f)$：STFT 功率谱；$E_B=\sum_{f\in B}P(t,f)$ 为频带能量
- $N_{\text{active}}$：脊线有效帧数；$t_{on},t_{off},t_p$：事件起点、终点、包络峰值时刻
- 所有比值类特征分母均加 $\varepsilon=10^{-12}$ 防除零

## 特征总表

| # | 变量名 | 中文名称 | 英文名称 | 计算公式 | 物理意义 |
|---|--------|----------|----------|----------|----------|
| 1 | `r_p` | 峰值位置比 | Peak Position Ratio | $r_p=\dfrac{t_p-t_{on}}{t_{off}-t_{on}}$ | 峰值在事件窗内的相对出现位置，判断能量前偏或后偏 |
| 2 | `C_E` | 能量质心位置 | Energy Centroid | $C_E=\dfrac{\sum_t t e^2(t)}{\sum_t e^2(t)}$ | 能量在时间轴上的加权中心，表征能量前/后偏置 |
| 3 | `A_env` | 包络不对称度 | Envelope Asymmetry | $A_{\text{env}}=\dfrac{(t_{off}-t_p)-(t_p-t_{on})}{t_{off}-t_{on}}$ | 上升段与衰减段的长度差异，正值为拖尾型 |
| 4 | `S_env` | 包络对称度 | Envelope Symmetry | $S_{\text{env}}=\mathrm{corr}(e,e^{rev})$ | 包络与其反折的相关系数，越接近1越对称 |
| 5 | `Sk_env` | 包络偏度 | Envelope Skewness | $Sk_{\text{env}}=\dfrac{1}{N}\sum\dfrac{(e_i-\mu_e)^3}{\sigma_e^3}$ | 包络幅值分布的重尾偏斜方向 |
| 6 | `R_td` | 上升/衰减时间比 | Rise/Decay Time Ratio | $R_{td}=\dfrac{T_{\text{decay}}}{T_{\text{rise}}}$ | 衰减时间相对上升时间长度，断丝多为快速衰减 |
| 7 | `R_fb` | 前后能量比 | Front/Back Energy Ratio | $R_{fb}=\dfrac{E_{\text{before}}}{E_{\text{after}}}$ | 峰值前/后能量之比，表征事件能量集中在前段或后段 |
| 8 | `C_bulge` | 鼓包集中度 | Bulge Concentration | $C_{\text{bulge}}=\dfrac{T_\eta}{T}$ | 包络超过阈值 $\eta$ 的时长占比，衡量主体鼓包集中度 |
| 9 | `N_bulge` | 鼓包数量 | Bulge Count | $N_{\text{bulge}}=\#\{\text{最小间距约束下的局部鼓包}\}$ | 显著局部鼓包个数，多鼓包反映多次反射叠加 |
| 10 | `epsilon_env` | 包络拟合残差 | Envelope Fit Residual | $\varepsilon_{\text{env}}=\dfrac{\lVert e-\hat e\rVert_2}{\lVert e\rVert_2}$ | 包络偏离基准模型的程度 |
| 11 | `eta_bw` | 有效带宽比 | Effective Bandwidth Ratio | $\eta_{\text{bw}}=\dfrac{t_{95}-t_p}{t_p-t_{05}}$ | 5%–95% 累积能量区间围绕峰值的前后宽度比 |
| 12 | `SC_mean` | 谱质心均值 | Spectral Centroid Mean | $SC(t)=\dfrac{\sum_f f P(t,f)}{\sum_f P(t,f)}$ | 频谱能量中心的时均值，表征主频段位置 |
| 13 | `k_sc` | 谱质心变化斜率 | Spectral Centroid Slope | $SC(t)\approx k_{sc}t+b_{sc}$ | 谱质心随时间迁移的速度 |
| 14 | `R_hl_mean` | 高低频能量比均值 | High/Low Energy Ratio | $R_{hl}(t)=\dfrac{E_{\text{high}}(t)}{E_{\text{low}}(t)}$ | 高频相对低频能量占比的时均值 |
| 15 | `k_hl` | 高低频比变化斜率 | High/Low Ratio Slope | $R_{hl}(t)\approx k_{hl}t+b_{hl}$ | 高频占比随时间的演化速率 |
| 16 | `beta_H` | 高频衰减斜率 | High-band Decay Slope | $\log(E_H(t)+\varepsilon)\approx\beta_H t+b_H$ | 高频带对数能量的衰减速率，测衰减快慢 |
| 17 | `H_tf` | 时频熵 | Time-Frequency Entropy | $H_{tf}=-\sum_{i,j}p_{ij}\ln p_{ij}$ | 时频能量分布的扩散度，浓度高则熵小 |
| 18 | `SF` | 谱平坦度 | Spectral Flatness | $SF=\dfrac{\exp(\frac1K\sum\ln P_k)}{\frac1K\sum P_k}$ | 频谱接近噪声（平坦/1）或窄带（接近0）的程度 |
| 19 | `H_alpha` | Renyi 熵 | Renyi Entropy | $H_\alpha=\dfrac{1}{1-\alpha}\log\sum p_{tf}^{\alpha}$ | 衡量时频能量集中度，$\alpha$ 越大越强调主集中分量 |
| 20 | `rho_r` | 脊线有效占比 | Ridge Activity Ratio | $\rho_r=\dfrac{L_{\text{valid}}}{L_{\text{all}}}$ | 主脊线有效帧占比，衡量脊线结构存在程度 |
| 21 | `G_gap` | 脊线间隙比 | Ridge Gap Ratio | $G_{\text{gap}}=\dfrac{N_{\text{gap}}}{L_{\text{all}}}$ | 脊线平直停顿帧占比，衡量间断/锁定程度 |
| 22 | `R2_ridge` | 脊线拟合优度 | Ridge R² Fit | $R^2(f_1,\hat f_1)$ | 主脊线二阶多项式拟合的确定系数 |
| 23 | `S_arch` | 拱形轨迹分数 | Arch Trajectory Score | $S_{\text{arch}}=R^2\mathbf1(a<0)\mathbf1(t_v\in[0.2T,0.8T])$ | 频率轨迹是否呈先升后降的拱形 |
| 24 | `H2_ratio` | 二倍频一致性比 | H2 Consistency Ratio | $H2_{\text{ratio}}=\dfrac{N\{\lvert f_2-2f_1\rvert<\Delta f\}}{N_{\text{active}}}$ | 二倍频与基频满足严格 2:1 关系的帧占比 |
| 25 | `R_2_1` | 二倍频/基频能量比 | 2nd/1st Harmonic Ratio | $R_{2:1}=\dfrac{\sum E(t,2f_1\pm\Delta f)}{\sum E(t,f_1\pm\Delta f)}$ | 二倍频能量相对基频能量的强度 |
| 26 | `H_stack` | 谐波栈能量比 | Harmonic Stack Ratio | $H_{\text{stack}}=\dfrac{\sum_t\sum_{m=1}^M E(t,mf_1\pm\Delta f)}{\sum_{t,f}P(t,f)}$ | 各阶谐波能量占全谱能量比例，测谐波组织度 |
| 27 | `R_h` | 谐波能量占比 | Harmonic Energy Ratio | $R_h=\dfrac{E_{2nd}}{E_{1st}}$ | 二倍频相对基频的强度（与 `R_2_1` 口径相近） |
| 28 | `epsilon_2x` | 二倍频一致性误差 | H2 Consistency Error | $\varepsilon_{2\times}=\mathrm{median}\!\left(\dfrac{\lvert f_2-2f_1\rvert}{f_1}\right)$ | 二倍频轨迹偏离 2:1 关系的相对误差中位数 |
| 29 | `C_h` | 谐波一致性水平 | Harmonic Consistency Level | $C_h=\mathrm{mean}(\lvert f_2-2f_1\rvert)$ | 二倍频与基频间距的绝对均值，越小谐波关系越严格 |
| 30 | `R_harm` | 谐波能量占比 | Ridge Harmonic Energy | $R_{\text{harm}}=\dfrac{E_{\text{ridge}}}{E_{\text{total}}}$ | 谐波脊线能量占总能量的比例 |
| 31 | `Ridge_coh` | 脊线相干度 | Ridge Coherence | $Ridge_{\text{coh}}=\dfrac{E_{\text{ridge}}}{E_{\text{total}}}$ | 与 `R_harm` 同口径，脊线能量集中程度 |
| 32 | `rho_up` | 脊线上升率 | Ridge Up-Rate | $\rho_{up}=\dfrac{N\{df_1/dt>\tau_s\}}{N_{\text{active}}}$ | 主脊线频率上扫活跃程度 |
| 33 | `rho_down` | 脊线下降率 | Ridge Down-Rate | $\rho_{down}=\dfrac{N\{df_1/dt<-\tau_s\}}{N_{\text{active}}}$ | 主脊线频率下扫活跃程度 |
| 34 | `N_turn` | 脊线转向次数 | Ridge Turn Count | $N_{\text{turn}}=N\{\mathrm{sign}(df_1/dt)\ \text{变化}\}$ | 频率轨迹斜率变号次数，测曲折程度 |
| 35 | `Delta_f_span` | 频率跨度 | Frequency Span | $\Delta f_{\text{span}}=\max(f_1)-\min(f_1)$ | 主脊线频率摆动覆盖范围 |
| 36 | `C_f` | 频率曲率 | Frequency Curvature | $C_f=\mathrm{mean}\!\left(\left\lvert\dfrac{d^2f_1}{dt^2}\right\rvert\right)$ | 主频轨迹的弯折程度 |
| 37 | `E_harm` | 谐波能量 | Harmonic Energy | $E_{\text{harm}}=\sum M_hP$ | 谐波脊线掩模下的绝对能量 |
| 38 | `E_res` | 残差能量 | Residual Energy | $E_{\text{res}}=E_{\text{total}}-E_{\text{harm}}$ | 谐波模型未解释的能量 |
| 39 | `rho_res` | 残差能量占比 | Residual Energy Ratio | $\rho_{\text{res}}=\dfrac{E_{\text{res}}}{E_{\text{total}}}$ | 未能被模型解释的能量占比，异常程度指标 |
| 40 | `R_high_res` | 高频残差占比 | High-freq Residual Ratio | $R_{\text{high,res}}=\dfrac{E_{\text{res,high}}}{E_{\text{total}}}$ | 高频带内残差能量占总能量比例 |
| 41 | `epsilon_rec` | 重构归一化误差 | Reconstruction Error | $\varepsilon_{\text{rec}}=\dfrac{\lVert x-\hat x_f\rVert_2^2}{\lVert x\rVert_2^2}$ | 原始信号与谐波重构信号的整体失配程度 |
| 42 | `N_abn` | 异常帧数 | Abnormal Frame Count | $N_{\text{abn}}=N\{\rho_{\text{res}}^{(m)}>\tau_{\text{res}}\}$ | 残差能量占比超阈值的帧数，测残差爆发帧规模 |
| 43 | `F_peak` | 谱通量峰值 | Spectral Flux Peak | $F_{\text{peak}}=\max_i\sum_j[P(i,j)-P(i-1,j)]_+$ | 频谱突变的峰值强度，测瞬态冲击 |
| 44 | `R_tkeo` | TKEO 峰值比 | TKEO Peak Ratio | $R_{\text{tkeo}}=\dfrac{\max(\Psi[n])}{\mathrm{mean}(\Psi[n])}$ | 原信号 Teager 能量的峰值/均值比，测冲击尖锐度 |
| 45 | `R_res_tkeo` | 残差 TKEO 峰值比 | Residual TKEO Peak Ratio | $R_{\text{res,tkeo}}=\dfrac{\max(\Psi_{\text{res}}[n])}{\mathrm{mean}(\Psi_{\text{res}}[n])}$ | 残差瞬态冲击的尖锐程度 |
| 46 | `K_loc` | 全局峭度 | Kurtosis | $K_{\text{loc}}=\dfrac{\frac1N\sum(x-\mu)^4}{\sigma^4}$ | 原信号幅值分布的重尾程度 |
| 47 | `K_res_max` | 残差局部峭度最大 | Residual Local Kurtosis Max | $K_{\text{res,max}}=\max_m K_{\text{res}}^{(m)}$ | 滑动窗内残差峭度最大值，测最强局部冲击 |
| 48 | `SK_max` | 谱峭度峰值 | Spectral Kurtosis Max | $SK_{\max}=\max_k \widehat{SK}(k)$ | 各频带谱峭度最大值，定位瞬态主导频带 |
| 49 | `CF_res` | 残差峰值因子 | Residual Crest Factor | $CF_{\text{res}}=\dfrac{\max\lvert x_{\text{res}}\rvert}{\mathrm{RMS}(x_{\text{res}})}$ | 残差峰值相对 RMS 的突出程度 |
| 50 | `T_half_high` | 高频半衰期 | High-band Half-life | $T_{\text{half,high}}=t(e_H=0.5e_{H,\max})-t_{\text{peak}}$ | 高频包络从峰值衰减到一半的时间，测衰减快慢 |
| 51 | `SC_res_mean` | 残差谱质心均值 | Residual Spectral Centroid | $SC_{\text{res}}(t)=\dfrac{\sum f P_{\text{res}}}{\sum P_{\text{res}}}$ | 残差能量集中的平均频段 |
| 52 | `k_res_sc` | 残差谱质心斜率 | Residual Centroid Slope | $SC_{\text{res}}(t)\approx k_{\text{res,sc}}t+b$ | 残差主频质心随时间的迁移速度 |
| 53 | `R_res_hl_mean` | 残差高低频能量比 | Residual High/Low Ratio | $R_{\text{res,hl}}(t)=\dfrac{E_{\text{res,high}}(t)}{E_{\text{res,low}}(t)}$ | 残差中高频相对低频能量占比 |
| 54 | `k_res_hl` | 残差高低频比斜率 | Residual H/L Slope | $R_{\text{res,hl}}(t)\approx k_{\text{res,hl}}t+b$ | 残差高频占比的时变速率 |
| 55 | `S_res_env` | 残差包络对称度 | Residual Env Symmetry | $S_{\text{res,env}}=\mathrm{corr}(e_{\text{res}},e_{\text{res}}^{rev})$ | 残差包络形状的对称性 |
| 56 | `eta_asym` | 残差不对称比 | Residual Asymmetry | $\eta_{\text{asym}}=\dfrac{E[t_0,t_0+\tau]}{E[t_0-\tau,t_0]}$ | 峰值后/前残差能量比，测上升与衰减不对称性 |
| 57 | `R_wp_*` | 小波包节点能量占比 | Wavelet Packet Node Ratio | $R_{\text{wp},i}=\dfrac{E_{\text{wp},i}}{\sum_j E_{\text{wp},j}}$ | 各小波包子带能量占全谱（小波域）比例 |
| 58 | `H_wp` | 小波包熵 | Wavelet Packet Entropy | $H_{\text{wp}}=-\sum_i p_i\ln p_i$ | 小波包能量分布扩散度 |
| 59 | `I_burst` | 子带突发指数 | Burst Index | $I_{\text{burst}}=\dfrac{\max(E_{\text{wp}})}{\mathrm{median}(E_{\text{wp}})}$ | 峰值子带能量相对中位水平的突出倍数 |
| 60 | `D_WPT` | 高频-低频能差 | WPT High-Low Difference | $D_{\text{WPT}}=\sum_{i\in HF}p_i-\sum_{i\in LF}p_i$ | 高低频小波包能量差，测高频异常偏置 |
| 61 | `C_damp` | 阻尼原子匹配度 | Damped Atom Match | $C_{\text{damp}}=\dfrac{\max_\theta\lvert\langle r,g_\theta\rangle\rvert}{\lVert r\rVert\lVert g_\theta\rVert}$ | 残差与阻尼正弦原子库的最佳匹配相似度 |
| 62 | `alpha_hat` | 估计阻尼系数 | Damping Coefficient | $\log e_{\text{res}}(t)\approx-\hat\alpha t+b$ | 残差包络对数线性拟合的斜率，测衰减速度 |
| 63 | `Q_MP` | 矩阵铅笔拟合优度 | Matrix Pencil Quality | 基于残差的 Hankel 矩阵秩 2 近似质量 | 阻尼正弦模型对残差的解释能力 |
| 64 | `Delta_J` | 损伤原子增益 | Damage Atom Gain | $\Delta J=\dfrac{\lVert x-\hat x_f\rVert^2-\lVert x-\hat x_f-\hat x_d\rVert^2}{\lVert x\rVert^2}$ | 加入损伤原子后重构误差下降率 |
| 65 | `eta_dict` | 字典损伤系数比 | Dictionary Damage Ratio | $\eta_{\text{dict}}=\dfrac{\lVert\alpha_d\rVert_1}{\lVert\alpha_f\rVert_1+\lVert\alpha_d\rVert_1}$ | 损伤字典系数在总稀疏系数中的比重 |

## 补充说明

### 频带前缀
同一条基础特征在多个频带上重复计算，CSV 列名为 `b_<band>__<base >`，例如：
- `b_1k_10k__rho_up` → 1–10 kHz 频带上的 `rho_up`
- `b_20k_30k__H2_ratio` → 20–30 kHz 频带上的 `H2_ratio`
- `b_1k_50k__epsilon_rec`、`b_1k_100k__epsilon_rec` → 更宽频带上的重构误差

频带划分见 `feature_schema.py:band_chinese_label()` 与 `src/fea_cpt_gpu_v2/params.py` 的 `LOW/MID/HIGH1/HIGH2/HARMONIC_BAND_HZ`。

### 物理意义解读
- **断丝（BK）信号**：典型表现为在管道共振频带产生狭窄的高次谐波列（钢丝弹性波 + 结构共振），故 `H_stack`、`H2_ratio`、`epsilon_2x` 等谐波族特征应对断丝敏感。
- **敲击（QJ）信号**：能量集中在冲击瞬间，包络尖锐、频谱偏白、谐波组织度低，故 `S_env`、`CF_res`、`SK_max`、`SF` 等瞬态/平坦度特征有区分价值。
- **正常流噪声（FL）信号**：宽带、平稳、无确定性谐波结构，`rho_res`、`N_abn`、`H_tf` 等反映其随机性。

### 与代码的对应关系
- `feature_schema.py:BASE_FEATURE_MEANINGS`：Notebook 展示用的中文含义映射（当前已覆盖全部 65 项静态特征）。
- `features.py:compute_all_features()`：上述所有特征的实际计算实现。