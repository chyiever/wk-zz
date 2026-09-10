# PCCP 断丝识别特征字典

本文档是 PCCP 断丝信号特征挖掘所用候选特征的完整字典表，包含公式、中英文名称、代码变量名与物理意义。
代码实现见 `src/fea_cpt_gpu_v2_2/features.py`（`DATA09_v0-flow_feature_extraction.ipynb`、`DATA09_v0-qj` 样本特征提取等实际调用的版本），中文含义映射见 `src/pccp_feature_mining/feature_schema.py` 的 `BASE_FEATURE_MEANINGS`。

> 条目说明：原表按 65 行计（`R_wp_*` 记为 1 行）。当前 schema `pccp-v5-band100k-stat-entropy-20260910` 已增加经典统计、熵、MFCC 和可观测性特征，并按**频带适用的特征族输出**；不再把全部特征机械复制到每个频带。当前 8 带动态特征列合计 563 列，实际列数由 `feature_families_by_band` 决定。

## 符号约定

- $x_B[n]$：当前规范带通信号（默认关注 100 Hz–100 kHz）；$X_B$：由同一 `STFT` 后端计算的复谱；$M_h\in\{0,1\}$：谐波脊线掩模；唯一残差定义为 $X_{\text{res}}=(1-M_h)X_B$、$x_{\text{res}}=\mathrm{ISTFT}(X_{\text{res}})$
- $e[n]$：包络 $e[n]=|\mathrm{hilbert}(s[n])|$，$s[n]$ 取 $x_B,x_{H1},x_{H2}$ 或 $x_{\text{res}}$
- $f_1(t)$：主脊线频率轨迹；$f_2(t)$：二倍频脊线频率轨迹
- $P(t,f)$：STFT 功率谱；$E_B=\sum_{f\in B}P(t,f)$ 为频带能量
- $N_{\text{active}}$：脊线有效帧数（有效帧判据：主带帧能量 > $\lambda\cdot$ 全窗帧能量中位数，$\lambda=0.2$，且脊线频率 $>0$）；$t_{on},t_{off},t_p$：事件起点、终点、包络峰值时刻
- 所有比值类特征分母均加 $\varepsilon=10^{-12}$ 防除零
- 偏度/峭度等矩量实现采用**无偏估计**（`scipy.stats` `bias=False`）
- `R_td` 的"上升/衰减时间"在实现中由事件边界能量分位点（5%/95% 累积能量）代替阈值穿越时刻

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
| 11 | `eta_bw` | 包络左右宽度比 | Envelope Left/Right Width Ratio | $\eta_{\text{bw}}=\dfrac{t_{95}-t_p}{t_p-t_{05}}$ | 5%–95% 累积能量区间围绕峰值的前后宽度比（时域量，与频率带宽无关） |
| 12 | `SC_mean` | 谱质心均值 | Spectral Centroid Mean | $SC(t)=\dfrac{\sum_f f P(t,f)}{\sum_f P(t,f)}$ | 频谱能量中心的时均值，表征主频段位置 |
| 13 | `k_sc` | 谱质心变化斜率 | Spectral Centroid Slope | $SC(t)\approx k_{sc}t+b_{sc}$ | 谱质心随时间迁移的速度 |
| 14 | `R_hl_mean` | 高低频能量比均值 | High/Low Energy Ratio | $R_{hl}(t)=\dfrac{E_{\text{high}}(t)}{E_{\text{low}}(t)}$ | 高频相对低频能量占比的时均值 |
| 15 | `k_hl` | 高低频比变化斜率 | High/Low Ratio Slope | $R_{hl}(t)\approx k_{hl}t+b_{hl}$ | 高频占比随时间的演化速率 |
| 16 | `beta_H` | 高频衰减斜率 | High-band Decay Slope | $\log(E_H(t)+\varepsilon)\approx\beta_H t+b_H,\ t\in[t_p,t_{off}]$ | 高频带对数能量在**峰后衰减段**的拟合斜率，测衰减快慢（不混入上升段） |
| 17 | `H_tf` | 时频熵 | Time-Frequency Entropy | $H_{tf}=-\sum_{i,j}p_{ij}\ln p_{ij}$ | 时频能量分布的扩散度，浓度高则熵小 |
| 18 | `SF` | 谱平坦度 | Spectral Flatness | $SF=\dfrac{\exp(\frac1K\sum\ln P_k)}{\frac1K\sum P_k}$ | 频谱接近噪声（平坦/1）或窄带（接近0）的程度 |
| 19 | `H_alpha` | Renyi 熵 | Renyi Entropy | $H_\alpha=\dfrac{1}{1-\alpha}\log\sum p_{tf}^{\alpha}$ | 衡量时频能量集中度，$\alpha$ 越大越强调主集中分量 |
| 20 | `rho_r` | 脊线有效占比 | Ridge Activity Ratio | $\rho_r=\dfrac{L_{\text{valid}}}{L_{\text{all}}}$ | 主脊线有效帧占比。$L_{\text{valid}}$ 按"主带帧能量 $>0.2\times$ 帧能量中位数"判据统计，衡量事件占据的帧比例 |
| 21 | `G_gap` | 脊线间隙比 | Ridge Gap Ratio | $G_{\text{gap}}=\dfrac{N_{\text{gap}}}{L_{\text{all}}}$ | 间隙 = 脊线丢失帧 或 有效脊线相邻帧跳变 $>2$ 倍频率分辨率；衡量间断/跳变程度 |
| 22 | `R2_ridge` | 脊线拟合优度 | Ridge R² Fit | $R^2(f_1,\hat f_1)$ | 主脊线二阶多项式拟合的确定系数 |
| 23 | `S_arch` | 拱形轨迹分数 | Arch Trajectory Score | $S_{\text{arch}}=R^2\mathbf1(a<0)\mathbf1(t_v\in[0.2T,0.8T])$ | 频率轨迹是否呈先升后降的拱形 |
| 24 | `H2_ratio` | 二倍频一致性比 | H2 Consistency Ratio | $H2_{\text{ratio}}=\dfrac{N\{\lvert f_2-2f_1\rvert<\Delta f\}}{N_{\text{active}}}$ | 二倍频与基频满足严格 2:1 关系的帧占比 |
| 25 | `R_2_1` | 二倍频/基频能量比 | 2nd/1st Harmonic Ratio | $R_{2:1}=\dfrac{\sum_t E(t,2f_1\pm\Delta f)}{\sum_t E(t,f_1\pm\Delta f)}$ | 二倍频能量相对基频能量的强度，$\Delta f=\max(2\text{bin},0.08f_1)$ 相对带宽邻域积分 |
| 26 | `H_stack` | 谐波栈能量比 | Harmonic Stack Ratio | $H_{\text{stack}}=\dfrac{\sum_t\sum_{m=1}^M E(t,mf_1\pm\Delta f)}{\sum_{t,f}P(t,f)}$ | 各阶谐波能量占全谱能量比例，测谐波组织度 |
| 27 | `R_h` | 谐波能量占比 | Harmonic Energy Ratio | $R_h=\dfrac{E_{2nd,\pm1\text{kHz}}}{E_{1st,\pm1\text{kHz}}}$ | 二倍频相对基频的强度。实现取 $f_1/f_2$ 脊线固定 $\pm1$ kHz 邻域带积分，与 `R_2_1`（相对带宽 $\pm0.08f_1$）互补，非重复列 |
| 28 | `epsilon_2x` | 二倍频一致性误差 | H2 Consistency Error | $\varepsilon_{2\times}=\mathrm{median}\!\left(\dfrac{\lvert f_2-2f_1\rvert}{f_1}\right)$ | 二倍频轨迹偏离 2:1 关系的相对误差中位数 |
| 29 | `C_h` | 谐波一致性水平 | Harmonic Consistency Level | $C_h=\mathrm{mean}(\lvert f_2-2f_1\rvert)$ | 二倍频与基频间距的绝对均值，越小谐波关系越严格 |
| 30 | `R_harm` | 谐波能量占比 | Ridge Harmonic Energy | $R_{\text{harm}}=\dfrac{E_{\text{ridge}}}{E_{\text{total}}}$ | 谐波脊线能量占总能量的比例 |
| 31 | `Ridge_coh` | 脊线相干度（已停用） | Ridge Coherence (deprecated alias) | $Ridge_{\text{coh}}=\dfrac{E_{\text{ridge}}}{E_{\text{total}}}$ | **代码层面与 `R_harm` 完全同值的别名列，已停用**（不参与模型选择）；如不需要兼容请直接忽略该列 |
| 32 | `rho_up` | 脊线上升率 | Ridge Up-Rate | $\rho_{up}=\dfrac{N\{df_1/dt>\tau_s\}}{N_{\text{active}}}$ | 主脊线频率上扫活跃程度 |
| 33 | `rho_down` | 脊线下降率 | Ridge Down-Rate | $\rho_{down}=\dfrac{N\{df_1/dt<-\tau_s\}}{N_{\text{active}}}$ | 主脊线频率下扫活跃程度 |
| 34 | `N_turn` | 脊线转向次数 | Ridge Turn Count | $N_{\text{turn}}=N\{\mathrm{sign}(df_1/dt)\ \text{变化}\}$ | 频率轨迹斜率变号次数；实现先剔除零斜率样本再统计，避免 `+→0→−` 重复计数 |
| 35 | `Delta_f_span` | 频率跨度 | Frequency Span | $\Delta f_{\text{span}}=\max(f_1)-\min(f_1)$ | 主脊线频率摆动覆盖范围 |
| 36 | `C_f` | 归一化频率曲率 | Normalized Frequency Curvature | $C_f=\dfrac{\mathrm{mean}\left(\left\lvert\dfrac{d^2f_1}{dt^2}\right\rvert\right)}{\mathrm{mean}\lvert f_1\rvert}$ | 主频轨迹的弯折程度，除以平均主频做**相对曲率**归一化，跨频带可比 |
| 37 | `E_harm` | 谐波能量 | Harmonic Energy | $E_{\text{harm}}=\sum M_hP$ | 谐波脊线掩模下的绝对能量 |
| 38 | `E_res` | 残差能量 | Residual Energy | $E_{\text{res}}=\sum\lvert(1-M_h)X_B\rvert^2$ | 当前主频带范围内、谐波模型未解释的能量；与 `E_total`、`E_harm` 使用同一复谱和频率范围 |
| 39 | `rho_res` | 残差能量占比 | Residual Energy Ratio | $\rho_{\text{res}}=\dfrac{E_{\text{res}}}{E_{\text{total}}}$ | 未能被模型解释的能量占比，异常程度指标 |
| 40 | `R_high_res` | 高频残差占比 | High-freq Residual Ratio | $R_{\text{high,res}}=\dfrac{E_{\text{res,high}}}{E_{\text{total}}}$ | 高频带内残差能量占总能量比例 |
| 41 | `epsilon_rec` | 重构归一化误差 | Reconstruction Error | $\varepsilon_{\text{rec}}=\dfrac{\lVert x_{\text{res}}\rVert_2^2}{\lVert x_B\rVert_2^2}$ | 使用唯一残差 `ISTFT((1-M_h)X_B)` 的时域归一化能量，避免另行用减法定义第二种残差 |
| 42 | `N_abn` | 异常帧数 | Abnormal Frame Count | $N_{\text{abn}}=N\{\rho_{\text{res}}^{(m)}>\tau_{\text{res}}\}$ | 残差能量占比超阈值的帧数，测残差爆发帧规模 |
| 43 | `F_peak` | 谱通量峰值 | Spectral Flux Peak | $F_{\text{peak}}=\max_i\sum_j[P(i,j)-P(i-1,j)]_+$ | 频谱突变的峰值强度，测瞬态冲击 |
| 44 | `R_tkeo` | TKEO 峰值比 | TKEO Peak Ratio | $R_{\text{tkeo}}=\dfrac{\max(\Psi[n])}{\mathrm{mean}(\Psi[n])}$ | 原信号 Teager 能量的峰值/均值比，测冲击尖锐度 |
| 45 | `R_res_tkeo` | 残差 TKEO 峰值比 | Residual TKEO Peak Ratio | $R_{\text{res,tkeo}}=\dfrac{\max(\Psi_{\text{res}}[n])}{\mathrm{mean}(\Psi_{\text{res}}[n])}$ | 残差瞬态冲击的尖锐程度 |
| 46 | `K_loc` | 全局峭度 | Kurtosis | $K_{\text{loc}}=\dfrac{\frac1N\sum(x-\mu)^4}{\sigma^4}$ | 原信号幅值分布的重尾程度 |
| 47 | `K_res_max` | 残差局部峭度最大 | Residual Local Kurtosis Max | $K_{\text{res,max}}=\max_m K_{\text{res}}^{(m)}$ | 滑动窗内残差峭度最大值，测最强局部冲击 |
| 48 | `SK_max` | 谱峭度峰值 | Spectral Kurtosis Max | $SK_{\max}=\max_k\left[\dfrac{\langle\lvert X_m(k)\rvert^4\rangle}{\langle\lvert X_m(k)\rvert^2\rangle^2}-2\right]$ | 短窗 STFT 各频点经典谱峭度（Antoni 四阶累积量，高斯过程→0）的最大值，定位瞬态主导频带 |
| 49 | `CF_res` | 残差峰值因子 | Residual Crest Factor | $CF_{\text{res}}=\dfrac{\max\lvert x_{\text{res}}\rvert}{\mathrm{RMS}(x_{\text{res}})}$ | 残差峰值相对 RMS 的突出程度 |
| 50 | `T_half_high` | 高频半衰期 | High-band Half-life | $T_{\text{half,high}}=t(e_H=0.5e_{H,\max})-t_{\text{peak}}$ | 高频包络从峰值衰减到一半的时长（峰值起算）；峰后从未跌破 50% 时返回 `NaN` |
| 51 | `SC_res_mean` | 残差谱质心均值 | Residual Spectral Centroid | $SC_{\text{res}}(t)=\dfrac{\sum f P_{\text{res}}}{\sum P_{\text{res}}}$ | 残差能量集中的平均频段 |
| 52 | `k_res_sc` | 残差谱质心斜率 | Residual Centroid Slope | $SC_{\text{res}}(t)\approx k_{\text{res,sc}}t+b$ | 残差主频质心随时间的迁移速度 |
| 53 | `R_res_hl_mean` | 残差高低频能量比 | Residual High/Low Ratio | $R_{\text{res,hl}}(t)=\dfrac{E_{\text{res,high}}(t)}{E_{\text{res,low}}(t)}$ | 残差中高频相对低频能量占比 |
| 54 | `k_res_hl` | 残差高低频比斜率 | Residual H/L Slope | $R_{\text{res,hl}}(t)\approx k_{\text{res,hl}}t+b$ | 残差高频占比的时变速率 |
| 55 | `S_res_env` | 残差包络对称度 | Residual Env Symmetry | $S_{\text{res,env}}=\mathrm{corr}(e_{\text{res}},e_{\text{res}}^{rev})$ | 残差包络形状的对称性 |
| 56 | `eta_asym` | 残差不对称比 | Residual Asymmetry | $\eta_{\text{asym}}=\dfrac{E[t_0,t_0+\tau]}{E[t_0-\tau,t_0]}$ | 峰值后/前残差能量比，测上升与衰减不对称性 |
| 57 | `R_wp_*` | 小波包节点能量占比 | Wavelet Packet Node Ratio | $R_{\text{wp},i}=\dfrac{E_{\text{wp},i}}{\sum_j E_{\text{wp},j}}$ | 各小波包子带能量占全谱（小波域）比例 |
| 58 | `H_wp` | 小波包熵 | Wavelet Packet Entropy | $H_{\text{wp}}=-\sum_i p_i\ln p_i$ | 小波包能量分布扩散度 |
| 59 | `I_burst` | 小波包子带能量集中度 | Subband Energy Concentration | $I_{\text{burst}}=\dfrac{\max(E_{\text{wp}})}{\mathrm{median}(E_{\text{wp}})}$ | 能量在子带间的集中度（跨子带 max/median），与"时间域突发性"无关 |
| 60 | `D_WPT` | 高频-低频能差 | WPT High-Low Difference | $D_{\text{WPT}}=\sum_{i\in HF}p_i-\sum_{i\in LF}p_i$ | WPT 前按主带上限自适应降采样（目标约为 $\max(4\text{kHz},3f_H)$，且不超过原采样率）；节点按 PyWavelets `order="freq"` 的显式频率次序映射，HF/LF 以主带几何中点划分 |
| 61 | `C_damp` | 阻尼原子匹配度 | Damped Atom Match | $g(t)=\mathbf1_{t\ge t_0}e^{-(t-t_0)/\tau}\cos(2\pi f(t-t_0)+\phi)$，$C_{\text{damp}}=\max_{t_0,f,\tau,\phi}\dfrac{\lvert\langle r,g\rangle\rvert}{\lVert r\rVert\lVert g\rVert}$ | 原子从候选事件起点/残差峰值起振，并以正交正弦—余弦基消除固定零相位偏置 |
| 62 | `alpha_hat` | 估计阻尼系数 | Damping Coefficient | $\log e_{\text{res}}(t)\approx-\hat\alpha t+b,\ t\in[t_p,t_{off}]$ | 残差包络对数在**峰后衰减段**线性拟合的斜率，测衰减速度（不混入上升段） |
| 63 | `Q_MP` | Hankel 低秩重建质量 | Hankel Low-rank Quality | 基于残差的 Hankel 矩阵秩 2 SVD 最优近似的重建质量 | 残差能否用单个阻尼振荡解释的度量（**非矩阵铅笔法**，无需估计极点） |
| 64 | `Delta_J` | 损伤原子增益 | Damage Atom Gain | $\Delta J=\dfrac{\lVert x-\hat x_f\rVert^2-\lVert x-\hat x_f-\hat x_d\rVert^2}{\lVert x\rVert^2}$ | 加入损伤原子后重构误差下降率 |
| 65 | `eta_dict` | 损伤分量波形占比 | Damage Component Waveform Ratio | $\eta_{\text{dict}}=\dfrac{\lVert x_d\rVert_1}{\lVert x_f\rVert_1+\lVert x_d\rVert_1}$ | 损伤分量波形 L1 范数占比。实现口径：$x_f$=谐波重构信号、$x_d$=最佳阻尼原子的损伤投影分量；**非稀疏字典系数占比**（未做字典学习/OMP） |

### 可观测性与背景增补特征（2026-09-10）

| 变量名 | 中文名称 | 计算公式 | 门限/用途 |
|---|---|---|---|
| `SNR_band_db` | 当前频带局部背景信噪比 | $10\log_{10}\dfrac{\bar E_{\mathrm{event}}+\varepsilon}{\bar E_{\mathrm{bg}}+\varepsilon}$ | 事件段由包络边界映射到 STFT 帧；背景不足时取窗口前 20% 作为局部背景 |
| `E_excess` | 超额能量 | $\max(\sum_{t\in event}E_B(t)-N_{event}\bar E_{bg},0)$ | 扣除局部背景基线后保留的非负能量，适合高频能量自然衰减场景 |
| `SNR_high_db` | 高频子带局部背景信噪比 | $10\log_{10}\dfrac{\bar E_{H,event}+\varepsilon}{\bar E_{H,bg}+\varepsilon}$ | 判断当前频带内部高频子带是否达到可解释水平 |
| `high_observable` | 高频可观测标志 | $\mathbf1(SNR_{high}\ge3\,\mathrm{dB}\land \bar E_{H,event}>\varepsilon)$ | 为 0 时 `beta_H`、`T_half_high` 输出 `NaN`，不把不可测误写成物理零值 |
| `H2_observable` | 二倍频可观测标志 | $\mathbf1(SNR_{2nd}\ge3\,\mathrm{dB}\land \bar E_{2,event}>\varepsilon)$ | `SNR_2nd` 使用二倍频脊线邻域的事件/背景帧能量；只在明确的宽带谐波上下文计算，为 0 时二倍频比值/偏差类量输出 `NaN` |

### 经典统计、熵与倒谱增补特征（2026-09-10）

| 特征族 | 变量名 | 计算口径 |
|---|---|---|
| 经典幅值统计 | `mean`、`variance`、`rms`、`skewness`、`kurtosis` | 当前规范带通信号的均值、方差、均方根、偏度和 Pearson 峭度（`fisher=False`） |
| 经典因子 | `waveform_factor`、`crest_factor`、`impulse_factor`、`clearance_factor` | 分别为 RMS/平均绝对值、峰值/RMS、峰值/平均绝对值、峰值/(平均平方根幅值)$^2$ |
| 排列熵 | `permutation_entropy`、`MPE_scale2`、`MPE_scale3` | Bandt–Pompe 三阶排列熵及时间粗粒化尺度 2、3；归一化到 0–1 |
| 奇异谱熵 | `singular_spectrum_entropy` | 对限长 Hankel 轨迹矩阵的奇异值平方归一化后计算 Shannon 熵，并归一化到 0–1 |
| 频谱延展度 | `spectral_spread` | $sqrt{\sum_{f,t}(f-SC)^2P(f,t)/\sum_{f,t}P(f,t)}$ |
| 功率谱熵 | `power_spectral_entropy` | 对时间平均功率谱归一化后计算 Shannon 熵，并除以 $\log K$ |
| 分解能量熵 | `energy_entropy` | 将时域窗口等分为 8 段，对各段能量归一化后计算 Shannon 熵，并除以 $\log 8$；它与 WPT 熵是不同分解域 |
| 梅尔倒谱 | `MFCC_01`–`MFCC_13` | 当前带 STFT 平均功率经 26 个 Mel 三角滤波器、对数压缩和正交 DCT 得到 13 个系数 |

这些增补特征通过 `classic`、`entropy`、`cepstral` 特征族选择输出。100 Hz–1 kHz 当前仍只输出 `classic/time/background`；该带的谱熵、MFCC 和频谱延展度应在长窗/低采样率多分辨率路径上线后再启用。

## 补充说明

### 频带前缀与子带划分
CSV 列名仍采用 `b_<band>__<base>`，但特征按物理适用的族选择，不再在每个频带上全部重复。例如：

- `b_100_100k` 是明确的宽带谐波上下文，可输出 harmonic/wavelet/damped 等完整族；
- `b_100_1k` 主要输出 time/spectral/background；
- `b_5k_15k` 主要输出 time/spectral/ridge/background；
- `b_30k_60k` 输出 time/spectral/residual/background，并通过高频可观测门限解释衰减量。

当前 DATA09 正式频带为 `b_100_100k`、`b_1k_100k`、`b_100_1k`、`b_1k_5k`、`b_5k_15k`、`b_15k_30k`、`b_30k_60k`、`b_60k_100k`；预处理先对整段信号去均值，再执行 100 Hz 高通，并以 105 kHz 低通提供 100 kHz 通带的过渡余量。

`b_<band>` 前缀决定该频带内的子带参数：`sliding_window.build_params_for_band()` 按主带跨度分数生成 `low/mid/high1/high2/harmonic` 子带（低频 $0.30\times$、中频 $0.30$–$0.60\times$、高频 $0.50$–$0.80\times$、高频带 $0.60\times$–上限、谐波带 $0.50\times$–上限），主脊线搜索 $0$–$0.65\times$ 跨度。因此同名基础特征在不同频带前缀下的高/低频含义不同，比较时须带上前缀。
频带划分的常量定义见 `src/fea_cpt_gpu_v2_2/params.py` 的 `LOW/MID/HIGH1/HIGH2/HARMONIC_BAND_HZ`，中文展示见 `feature_schema.py:band_chinese_label()`。

### 实现口径补充说明（与代码对齐）
- `rho_r` 及 `H2_ratio`、`epsilon_2x`、`C_h`、`rho_up/down` 的 $N_{\text{active}}$ 分母共用**帧有效性判据**（主带帧能量 $>0.2\times$ 帧能量中位数）。
- `beta_H`、`alpha_hat` 为**峰后窗拟合**（$t\in[t_p,t_{off}]$），避免全窗拟合混入能量上升段；高频不可观测时衰减量返回 `NaN`。
- `STFT/ISTFT` 必须成对使用同一库与完全一致的窗、步长、中心化及长度约定；Torch 批量前向在主进程集中执行，worker 使用 Torch CPU 逆变换，避免多进程争用 CUDA 上下文。
- `E_total`、`E_harm`、`E_res` 统一取当前规范带通信号的同一 STFT 频率范围，并用 `float64` 累加；唯一残差为 `ISTFT((1-M_h)X_B)`。
- WPT 四层分解前按当前主带上限自适应降采样，节点中心频率依据 `order="freq"` 返回位置而非 `a/d` 路径二进制直译。
- 阻尼原子以事件起点/残差峰值为候选 $t_0$，并同时投影正弦、余弦基以消除起点和相位偏置。
- `F_peak` 按**逐频点正增量（半波整流）再求和**计算，首帧通量置 0。
- `T_half_high` 为**峰后衰减时长**，峰后从未跌破 50% 时返回 `NaN`。
- `R_2_1`、`H_stack` 采用脊线 $\pm\Delta f$（$\Delta f=\max(2\text{bin},0.08f_1)$）带内积分；`R_h` 采用 $f_1/f_2$ 固定 $\pm1$ kHz 邻域积分（超出频率轴范围的帧跳过），二者互补而非重复。
- `Ridge_coh` 为 `R_harm` 的**同值别名列（已停用）**。
- `Sk_env`、`K_loc`、`K_res_max` 等矩量采用无偏估计；`R_td` 的边界用 5%/95% 累积能量分位点。
- `eta_dict` 为波形 L1 占比（谐波重构 vs 损伤分量），非稀疏字典系数比。

### 物理意义解读
- **断丝（BK）信号**：典型表现为在管道共振频带产生狭窄的高次谐波列（钢丝弹性波 + 结构共振），故 `H_stack`、`H2_ratio`、`epsilon_2x` 等谐波族特征应对断丝敏感。
- **敲击（QJ）信号**：能量集中在冲击瞬间，包络尖锐、频谱偏白、谐波组织度低，故 `S_env`、`CF_res`、`SK_max`、`SF` 等瞬态/平坦度特征有区分价值。
- **正常流噪声（FL）信号**：宽带、平稳、无确定性谐波结构，`rho_res`、`N_abn`、`H_tf` 等反映其随机性。

### 与代码的对应关系
- `feature_schema.py:BASE_FEATURE_MEANINGS`：Notebook 展示用的中文含义映射（包括新增背景/可观测性特征；`R_wp_*` 支持动态节点名回退）。
- `fea_cpt_gpu_v2_2/features.py:compute_all_features()`：上述所有特征的实际计算实现。
- `fea_cpt_gpu_v2_2/sliding_window.py`：滑窗流水线，`DATA09_v0-flow_feature_extraction.ipynb` 即调用该流水线。

---

## 修订记录

| 日期 | 修改内容 | 原（修改前） | 现（修改后） |
|---|---|---|---|
| 2026-09-10 | 模块路径更正 | `src/fea_cpt_gpu_v2/` | `src/fea_cpt_gpu_v2_2/`（`DATA09_v0-flow_*` 等实际调用版本） |
| 2026-09-10 | #11 `eta_bw` 名称 | 有效带宽比 | 包络左右宽度比（时域量，与频率带宽无关） |
| 2026-09-10 | #16/#62 `beta_H`、`alpha_hat` 拟合区间 | 全窗拟合（未限定区间） | 峰后窗拟合 $t\in[t_p,t_{off}]$，避免上升段污染衰减斜率 |
| 2026-09-10 | #20/#21/#32/#33/#34 `rho_r`、`G_gap`、`rho_up/down`、`N_turn` | 无有效性判据，DP 脊线必返回正频率 → `rho_r` 恒为 1；`G_gap` 判据为"卡同一频点(<1e-12)"；`N_turn` 零斜率段重复计数 | 帧有效性判据（主带帧能量>0.2×中位数）；`G_gap`=脊线丢失/跳变>2bin 帧占比；`N_turn` 剔除零斜率后再计方向变化 |
| 2026-09-10 | #25 `R_2_1`、#26 `H_stack` 积分口径 | 脊线单 bin 采样 | 脊线 ±Δf（Δf=max(2bin,0.08f₁)）带内积分 |
| 2026-09-10 | #27 `R_h` 口径 | 与 `R_2_1` 同值重复（单 bin），文档称"口径相近" | 改为 $f_1/f_2$ 固定 ±1 kHz 邻域带积分，与 `R_2_1`（相对带宽）互补 |
| 2026-09-10 | #31 `Ridge_coh` | 名称"脊线相干度"，与 `R_harm` 同值且未标注 | 明确为 `R_harm` 的同值别名列，标注**已停用** |
| 2026-09-10 | #36 `C_f` | 名称"频率曲率"，公式未归一化 | 名称"归一化频率曲率"，公式补除以 $\mathrm{mean}\lvert f_1\rvert$（相对曲率，跨频带可比） |
| 2026-09-10 | #43 `F_peak` | 先求和后取正（代码） | 逐频点正增量（半波整流）后求和，首帧通量置 0 |
| 2026-09-10 | #48 `SK_max` | 各频点功率四阶矩，无 −2 项 | 经典谱峭度估计器 $\langle\lvert X\rvert^4\rangle/\langle\lvert X\rvert^2\rangle^2-2$（高斯→0） |
| 2026-09-10 | #50 `T_half_high` | 峰后从未跌破 50% 返回 0.0 | 返回 `NaN`（时长口径保持为峰值起算的衰减时长） |
| 2026-09-10 | #59 `I_burst` 名称 | 子带突发指数 / 突发性 | 小波包子带能量集中度（跨子带集中度，非时间域突发） |
| 2026-09-10 | #60 `D_WPT` 节点划分 | 按节点序对半分 | 按子带中心频率 > 主带几何中点划分 HF/LF |
| 2026-09-10 | #63 `Q_MP` 名称/含义 | 矩阵铅笔拟合优度（暗示矩阵铅笔法） | Hankel 低秩重建质量（秩 2 SVD 重建质量，无需估计极点） |
| 2026-09-10 | #65 `eta_dict` 名称/公式 | 字典损伤系数比（稀疏系数 L1 比） | 损伤分量波形占比（波形 L1 比，谐波重构 vs 损伤分量；无字典学习/OMP） |
| 2026-09-10 | 频带/子带说明 | 仅固定 5–60 kHz 主带描述，路径指向 v2 | 补充流水线按频带前缀动态生成子带规则（`build_params_for_band`），条目数（64 静态+16 节点）与模块路径更正 |
| 2026-09-10 | 无偏/口径说明 | — | 补注 `Sk_env`/`K_loc` 等无偏估计、`R_td` 能量分位边界、`T_half_high` NaN、`eta_dict` 波形口径 |
| 2026-09-10 | 经典统计/熵/倒谱增补 | 均值、方差、RMS、脉冲/裕度因子、排列熵、奇异谱熵、频谱延展度、功率谱熵、分解能量熵和 MFCC 缺失或仅有近似量 | 新增 `classic`、`entropy`、`cepstral` 特征族；低频 100 Hz–1 kHz 因当前短窗频率分辨率不足暂不输出谱类增补 |
| 2026-09-10 | schema 与频带更新 | 关注上限为 60 kHz，schema 为 v4 | 预处理先去均值再 100 Hz 高通，目标关注上限扩展至 100 kHz，schema 升级为 `pccp-v5-band100k-stat-entropy-20260910` |
