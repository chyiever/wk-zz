# 特征汇总

## 1. 说明

本文基于以下三个文档汇总并统一命名：

1. `PCCP纯流噪声与混合信号判别技术文档.md`
2. `断丝识别研究方案.md`
3. `超短高频声学信号区分的特征分析与快速实现方案.md`

本文目标：

- 提取三份文档中出现的全部主要特征；
- 对同名异符号、同义异名特征做统一；
- 统一给出分类、符号、物理意义、计算公式、基础量来源公式、建议预处理频带。

统一约定：

- 采样率记为 $f_s$；
- 原始信号记为 $x_{\text{raw}}[n]$，预处理主带通信号记为 $x_B[n]$；
- 默认主分析频带记为 $5$–$60\,\text{kHz}$；
- 若前端有效上限不足 $60\,\text{kHz}$，则高频相关特征降阶到 $20$–$40\,\text{kHz}$ 或 $25$–$45\,\text{kHz}$；
- 对“整窗特征”和“局部滑窗特征”同时存在的项目，优先保留“局部计算 + 全窗聚合”的实现方式；
- 本文中行内物理量、频带、窗口、阈值、比例式、表格公式统一采用 `$...$` 书写。

---

## 2. 建议统一预处理与基础量定义

### 2.1 建议预处理频带

建议统一保留以下几个频带版本，便于全部特征复用：

- 主分析带：$x_B[n] = BP_{5-60k}(x_0[n])$
- 低频带：$x_L[n] = BP_{5-15k}(x_0[n])$
- 中频带：$x_M[n] = BP_{15-30k}(x_0[n])$
- 高频带 1：$x_{H1}[n] = BP_{20-40k}(x_0[n])$
- 高频带 2：$x_{H2}[n] = BP_{20-60k}(x_0[n])$
- 可替代高频带：$25$–$45\,\text{kHz}$、$25$–$60\,\text{kHz}$、$30$–$60\,\text{kHz}$

其中去均值、去趋势后信号记为：

$$
x_0[n] = \text{detrend}\!\left(x_{\text{raw}}[n] - \text{mean}(x_{\text{raw}})\right)
$$

如需稳健归一化，可用：

$$
x_n[n] = \frac{x_B[n]-\mathrm{median}(x_B)}{1.4826\cdot \mathrm{MAD}(x_B)+\varepsilon}
$$

### 2.2 基础量来源公式

#### 2.2.1 解析包络

$$
e[n] = \left| \mathrm{hilbert}(s[n]) \right|
$$

其中 $s[n]$ 可取 $x_B[n]$、$x_{H1}[n]$、$x_{H2}[n]$ 或残差 $x_{\text{res}}[n]$。

#### 2.2.2 STFT 与功率谱

$$
X[i,k] = \sum_{m=0}^{L_w-1} x[m+iH]\,w[m]\,e^{-j2\pi km/L_w}
$$

$$
P[i,k] = |X[i,k]|^2
$$

频率映射：

$$
f_k = \frac{k f_s}{N_{\text{fft}}}
$$

#### 2.2.3 子带能量

对任一频带 $B=[f_a,f_b]$，定义：

$$
E_B(i) = \sum_{f_k\in B} P[i,k]
$$

全窗子带总能量：

$$
E_B = \sum_i E_B(i)
$$

#### 2.2.4 时频概率与熵

$$
p_{ij} = \frac{P[i,j]}{\sum_{i,j}P[i,j]}
$$

$$
H_{tf} = -\sum_{i,j} p_{ij}\ln p_{ij}
$$

#### 2.2.5 主脊线与二倍频脊线

主脊线、二倍频脊线分别记为：

$$
f_1(t_i),\quad f_2(t_i)
$$

通常：

- $f_1$ 搜索带：$5$–$30\,\text{kHz}$
- $f_2$ 搜索带：$10$–$60\,\text{kHz}$ 或 $2f_1(t)$ 邻域

#### 2.2.6 谐波掩膜与残差

谐波掩膜：

$$
M_h(i,k)=
\begin{cases}
1, & f_k \in f_1(t_i)\pm\Delta f \ \text{或}\ f_2(t_i)\pm\Delta f \\
0, & \text{其他}
\end{cases}
$$

建议：

$$
\Delta f = \max(2\text{ bin},\,0.08 f_1(t_i))
$$

时频残差：

$$
P_{\text{res}}[i,k] = (1-M_h(i,k))P[i,k]
$$

时域残差：

$$
\hat X_f[i,k] = M_h(i,k)X[i,k],\qquad
\hat x_f(t)=\mathrm{iSTFT}(\hat X_f)
$$

$$
x_{\text{res}}(t)=x_B(t)-\hat x_f(t)
$$

#### 2.2.7 残差与谐波能量

$$
E_{\text{harm}} = \sum_{i,k} M_h(i,k)P[i,k]
$$

$$
E_{\text{res}} = \sum_{i,k} P_{\text{res}}[i,k]
$$

$$
E_{\text{total}} = \sum_{i,k} P[i,k]
$$

#### 2.2.8 TKEO

$$
\Psi[n] = x[n]^2 - x[n-1]x[n+1]
$$

#### 2.2.9 峭度

$$
K = \frac{\frac{1}{N}\sum_{n=1}^N (x[n]-\mu)^4}{\sigma^4}
$$

#### 2.2.10 阻尼原子

$$
g_\theta(t)=e^{-(t-t_0)/\tau_d}\cos(2\pi f_d(t-t_0)+\phi)\,u(t-t_0)
$$

---

## 3. 特征分类总览

本文将全部特征统一分为以下九类：

1. 时间与包络类
2. 能量与比例类
3. 频域/时频演化类
4. 二倍频/谐波/脊线结构类
5. 残差与背景扣除类
6. 瞬态/冲击/高阶统计类
7. 小波/多分辨率类
8. 阻尼模型类
9. 稀疏表示与模型竞争类

---

## 4. 汇总表

| 序号 | 名称 | 符号 | 分类 | 物理意义（关键词） | 计算公式 |
|---|---|---|---|---|---|
| 1 | 包络峰位比 | $r_p$ | 时间/包络 | 峰值早晚、前移程度 | $r_p=\dfrac{t_p-t_{on}}{t_{off}-t_{on}}$ |
| 2 | 能量质心位置 | $C_E$ | 时间/包络 | 能量前后偏置、重心位置 | $C_E=\dfrac{\sum_t t\,e^2(t)}{\sum_t e^2(t)}$ |
| 3 | 包络不对称度 | $A_{\text{env}}$ | 时间/包络 | 早峰拖尾、快起慢衰 | $A_{\text{env}}=\dfrac{(t_{off}-t_p)-(t_p-t_{on})}{t_{off}-t_{on}}$ |
| 4 | 包络对称性 | $S_{\text{env}}$ | 时间/包络 | 左右镜像相似、鼓包程度 | $S_{\text{env}}=\mathrm{corr}(e,e^{rev})$ |
| 5 | 包络偏度 | $Sk_{\text{env}}$ | 时间/包络 | 拖尾偏斜、偏态 | $Sk_{\text{env}}=\dfrac{1}{N}\sum\dfrac{(e_i-\mu_e)^3}{\sigma_e^3}$ |
| 6 | 上升下降时间比 | $R_{td}$ | 时间/包络 | 快起慢衰 | $R_{td}=\dfrac{T_{\text{decay}}}{T_{\text{rise}}}$ |
| 7 | 前后能量比 | $R_{fb}$ | 时间/包络 | 峰前峰后能量不均衡 | $R_{fb}=\dfrac{E_{\text{before}}}{E_{\text{after}}}$ |
| 8 | 鼓包集中度 | $C_{\text{bulge}}$ | 时间/包络 | 鼓包主体占比 | $C_{\text{bulge}}=\dfrac{T_\eta}{T}$ |
| 9 | 局部鼓包个数 | $N_{\text{bulge}}$ | 时间/包络 | 多鼓包背景复杂度 | $N_{\text{bulge}}=\#\{\text{满足阈值与最小间隔约束的鼓包}\}$ |
| 10 | 包络拟合误差 | $\varepsilon_{\text{env}}$ | 时间/包络 | 偏离标准鼓包 | $\varepsilon_{\text{env}}=\dfrac{\|e-\hat e\|_2}{\|e\|_2}$ |
| 11 | 包络左右宽度比 | $\eta_{\text{bw}}$ | 时间/包络 | 左右展宽对称性 | $\eta_{\text{bw}}=\dfrac{t_{95}-t_p}{t_p-t_{05}}$ |
| 12 | 谱质心 | $SC(t)$ | 时频演化 | 频谱中心位置 | $SC(t)=\dfrac{\sum_f fP(t,f)}{\sum_f P(t,f)}$ |
| 13 | 谱质心衰减斜率 | $k_{sc}$ | 时频演化 | 高频向低频迁移速度 | $SC(t)\approx k_{sc}t+b_{sc}$ |
| 14 | 高频低频能量比 | $R_{hl}(t)$ | 能量/时频 | 高频占比 | $R_{hl}(t)=\dfrac{E_{\text{high}}(t)}{E_{\text{low}}(t)+\varepsilon}$ |
| 15 | 高频低频能量比斜率 | $k_{hl}$ | 能量/时频 | 高频衰减速度 | $R_{hl}(t)\approx k_{hl}t+b_{hl}$ |
| 16 | 高频衰减斜率 | $\beta_H$ | 能量/时频 | 峰后高频衰减 | $\log(E_H(t)+\varepsilon)\approx \beta_H t+b_H$ |
| 17 | 时频熵 | $H_{tf}$ | 时频演化 | 能量分散度 | $H_{tf}=-\sum p_{ij}\ln p_{ij}$ |
| 18 | 频谱平坦度 | $SF$ | 频域 | 谱平坦/尖峰程度 | $SF=\dfrac{\exp(\frac1K\sum \ln P_k)}{\frac1K\sum P_k}$ |
| 19 | Renyi 熵 | $H_\alpha$ | 时频演化 | 集中度/复杂度 | $H_\alpha=\dfrac{1}{1-\alpha}\log\sum p_{tf}^{\alpha}$ |
| 20 | 主脊线连续率 | $\rho_r$ | 谐波/脊线 | 主结构连续性 | $\rho_r=\dfrac{L_{\text{valid}}}{L_{\text{all}}}$ |
| 21 | 脊线缺口率 | $G_{\text{gap}}$ | 谐波/脊线 | 中断破坏程度 | $G_{\text{gap}}=\dfrac{N_{\text{gap}}}{L_{\text{all}}}$ |
| 22 | 脊线拟合优度 | $R^2_{\text{ridge}}$ | 谐波/脊线 | 主频轨迹可建模性 | $R^2(f_1,\hat f_1)$ |
| 23 | 拱形轨迹分数 | $S_{\text{arch}}$ | 谐波/脊线 | 先升后降拱形趋势 | $S_{\text{arch}}=R^2(f_1,q)\mathbf1(a<0)\mathbf1(t_v\in[0.2T,0.8T])$ |
| 24 | 二倍频存在度 | $H2_{\text{ratio}}$ | 二倍频/谐波 | 2:1 关系出现频率 | $H2_{\text{ratio}}=\dfrac{N\{|f_2-2f_1|<\Delta f\}}{N_{\text{active}}}$ |
| 25 | 二倍频比 | $R_{2:1}$ | 二倍频/谐波 | 二倍频能量强度 | $R_{2:1}=\dfrac{\sum_t E(t,2f_1\pm\Delta f)}{\sum_t E(t,f_1\pm\Delta f)+\varepsilon}$ |
| 26 | 谐波栈能量比 | $H_{\text{stack}}$ | 二倍频/谐波 | 多谐波组织程度 | $H_{\text{stack}}=\dfrac{\sum_t\sum_{m=1}^M E(t,mf_1\pm\Delta f)}{\sum_{t,f}S(t,f)}$ |
| 27 | 谐波能量比 | $R_h$ | 二倍频/谐波 | 二倍频/基频强度比 | $R_h=\dfrac{E_{2nd}}{E_{1st}}$ |
| 28 | 二倍频一致性误差 | $\varepsilon_{2\times}$ | 二倍频/谐波 | 偏离 2:1 程度 | $\varepsilon_{2\times}=\mathrm{median}\!\left(\dfrac{|f_2-2f_1|}{f_1}\right)$ |
| 29 | 谐波一致性误差 | $C_h$ | 二倍频/谐波 | 二倍频跟随误差 | $C_h=\mathrm{mean}(|f_2-2f_1|)$ |
| 30 | 谐波能量占比 | $R_{\text{harm}}$ | 二倍频/谐波 | 主谐波可解释占比 | $R_{\text{harm}}=\dfrac{E_{\text{ridge}}}{E_{\text{total}}}$ |
| 31 | 脊线连续性 | $Ridge_{\text{coh}}$ | 二倍频/谐波 | 谐波掩膜内能量占比 | $Ridge_{\text{coh}}=\dfrac{E_{\text{ridge}}}{E_{\text{total}}}$ |
| 32 | 上升帧占比 | $\rho_{up}$ | 脊线演化 | 上升段占比 | $\rho_{up}=\dfrac{N\{df_1/dt>\tau_s\}}{N_{\text{active}}}$ |
| 33 | 下降帧占比 | $\rho_{down}$ | 脊线演化 | 下降段占比 | $\rho_{down}=\dfrac{N\{df_1/dt<-\tau_s\}}{N_{\text{active}}}$ |
| 34 | 转折次数 | $N_{\text{turn}}$ | 脊线演化 | 拐点/多鼓包复杂度 | $N_{\text{turn}}=N\{\mathrm{sign}(df_1/dt)\text{变化}\}$ |
| 35 | 频率跨度 | $\Delta f_{\text{span}}$ | 脊线演化 | 基频摆动范围 | $\Delta f_{\text{span}}=\max(f_1)-\min(f_1)$ |
| 36 | 归一化曲率 | $C_f$ | 脊线演化 | 弯曲程度 | $C_f=\mathrm{mean}(|d^2f_1/dt^2|)$ |
| 37 | 谐波可解释能量 | $E_{\text{harm}}$ | 残差/背景扣除 | 可由流噪声解释的能量 | $E_{\text{harm}}=\sum M_hP$ |
| 38 | 残差能量 | $E_{\text{res}}$ | 残差/背景扣除 | 未解释异常能量 | $E_{\text{res}}=E_{\text{total}}-E_{\text{harm}}$ |
| 39 | 非谐波残差占比 | $\rho_{\text{res}}$ | 残差/背景扣除 | 异常残差占比 | $\rho_{\text{res}}=\dfrac{E_{\text{res}}}{E_{\text{total}}}$ |
| 40 | 高频残差占比 | $R_{\text{high,res}}$ | 残差/背景扣除 | 高频异常强度 | $R_{\text{high,res}}=\dfrac{E_{\text{res,high}}}{E_{\text{total}}}$ |
| 41 | 流噪声重构误差 | $\varepsilon_{\text{rec}}$ | 残差/背景扣除 | 主结构失配 | $\varepsilon_{\text{rec}}=\dfrac{\|x-\hat x_f\|_2^2}{\|x\|_2^2}$ |
| 42 | 异常子窗个数 | $N_{\text{abn}}$ | 残差/背景扣除 | 异常出现次数 | $N_{\text{abn}}=N\{\rho_{\text{res}}^{(m)}>\tau_{\text{res}}\}$ |
| 43 | 谱流量峰值 | $F_{\text{peak}}$ | 瞬态/冲击 | 帧间突变强度 | $F_{\text{peak}}=\max_i \sum_j [P(i,j)-P(i-1,j)]_+$ |
| 44 | TKEO 峰值比 | $R_{\text{tkeo}}$ | 瞬态/冲击 | 局部尖锐度 | $R_{\text{tkeo}}=\dfrac{\max(\Psi[n])}{\mathrm{mean}(\Psi[n])}$ |
| 45 | 残差 TKEO 峰值比 | $R_{\text{res,tkeo}}$ | 瞬态/冲击 | 残差尖锐瞬态 | $R_{\text{res,tkeo}}=\dfrac{\max(\Psi_{\text{res}}[n])}{\mathrm{mean}(\Psi_{\text{res}}[n])}$ |
| 46 | 局部峭度 | $K_{\text{loc}}$ | 瞬态/冲击 | 冲击尖峰 | $K_{\text{loc}}=\dfrac{\frac1N\sum(x-\mu)^4}{\sigma^4}$ |
| 47 | 残差局部峭度最大值 | $K_{\text{res,max}}$ | 瞬态/冲击 | 残差尖峰极值 | $K_{\text{res,max}}=\max_m K_{\text{res}}^{(m)}$ |
| 48 | 谱峭度峰值 | $SK_{\max}$ | 瞬态/冲击 | 瞬态频带定位 | $SK_{\max}=\max_k \hat{SK}(k)$ |
| 49 | 峰均比 | $CF_{\text{res}}$ | 瞬态/冲击 | 尖峰/均方强度 | $CF_{\text{res}}=\dfrac{\max|x_{\text{res}}|}{\mathrm{RMS}(x_{\text{res}})}$ |
| 50 | 高频半衰期 | $T_{\text{half,high}}$ | 瞬态/冲击 | 高频衰减时长 | $T_{\text{half,high}}=t(e_H=0.5e_{H,\max})-t_{\text{peak}}$ |
| 51 | 残差谱质心 | $SC_{\text{res}}(t)$ | 残差演化 | 异常中心频率 | $SC_{\text{res}}(t)=\dfrac{\sum fP_{\text{res}}}{\sum P_{\text{res}}}$ |
| 52 | 残差谱质心斜率 | $k_{\text{res,sc}}$ | 残差演化 | 异常频率下移速度 | $SC_{\text{res}}(t)\approx k_{\text{res,sc}} t+b$ |
| 53 | 残差高低频能量比 | $R_{\text{res,hl}}(t)$ | 残差演化 | 残差高频占比 | $R_{\text{res,hl}}(t)=\dfrac{E_{\text{res,high}}(t)}{E_{\text{res,low}}(t)+\varepsilon}$ |
| 54 | 残差高低频能量比斜率 | $k_{\text{res,hl}}$ | 残差演化 | 残差高频衰减速度 | $R_{\text{res,hl}}(t)\approx k_{\text{res,hl}}t+b$ |
| 55 | 残差包络对称性 | $S_{\text{res,env}}$ | 残差演化 | 残差拖尾对称性 | $S_{\text{res,env}}=\mathrm{corr}(e_{\text{res}},e_{\text{res}}^{rev})$ |
| 56 | 残差前后不对称度 | $\eta_{\text{asym}}$ | 残差演化 | 触发后能量偏置 | $\eta_{\text{asym}}=\dfrac{E[t_0,t_0+\tau]}{E[t_0-\tau,t_0]}$ |
| 57 | 小波包节点能量比 | $R_{\text{wp},i}$ | 小波/多分辨率 | 多尺度能量分布 | $R_{\text{wp},i}=\dfrac{E_{\text{wp},i}}{\sum_j E_{\text{wp},j}}$ |
| 58 | 小波熵 | $H_{\text{wp}}$ | 小波/多分辨率 | 子带分散度 | $H_{\text{wp}}=-\sum_i p_i\ln p_i$ |
| 59 | 高频子带能量突发指数 | $I_{\text{burst}}$ | 小波/多分辨率 | 高频突发性 | $I_{\text{burst}}=\dfrac{\max(E_{\text{wp}}(t))}{\mathrm{median}(E_{\text{wp}}(t))}$ |
| 60 | 高频-低频能差 | $D_{\text{WPT}}$ | 小波/多分辨率 | 高频异常偏置 | $D_{\text{WPT}}=\sum_{i\in HF}p_i-\sum_{i\in Harm}p_i$ |
| 61 | 阻尼原子最大匹配度 | $C_{\text{damp}}$ | 阻尼模型 | 阻尼振荡证据强度 | $C_{\text{damp}}=\dfrac{\max_\theta |\langle r,g_\theta\rangle|}{\|r\|\|g_\theta\|}$ |
| 62 | 残差衰减常数 | $\hat\alpha$ | 阻尼模型 | 指数衰减速度 | $\log e_{\text{res}}(t)\approx -\hat\alpha t+b$ |
| 63 | 矩阵铅笔拟合优度 | $Q_{\text{MP}}$ | 阻尼模型 | 阻尼正弦解释能力 | 阻尼拟合误差下降率 |
| 64 | 加入损伤原子后的误差下降率 | $\Delta J$ | 模型竞争 | 新增损伤分量的解释增益 | $\Delta J=\dfrac{\|x-\hat x_f\|^2-\|x-\hat x_f-\hat x_d\|^2}{\|x\|^2}$ |
| 65 | 字典损伤系数比 | $\eta_{\text{dict}}$ | 稀疏表示 | 损伤字典占比 | $\eta_{\text{dict}}=\dfrac{\|\alpha_d\|_1}{\|\alpha_f\|_1+\|\alpha_d\|_1}$ |

---

## 5. 分类详解

## 5.1 时间与包络类

### 5.1.1 包络峰位比 $r_p$

- 名称：包络峰位比
- 符号：$r_p$
- 物理意义：刻画峰值是否前移。断丝常“早峰值”，标准鼓包常接近中部。
- 计算公式：

$$
r_p=\frac{t_p-t_{on}}{t_{off}-t_{on}}
$$

- 来源公式：
  - $e[n]=|\mathrm{hilbert}(s[n])|$
  - $t_p = n_p/f_s,\ n_p=\arg\max e[n]$
  - $t_{on}, t_{off}$ 来自事件边界检测
- 建议频带：
  - 主推荐：5–60 kHz
  - 低上限设备：5–40 kHz
  - 若只关心包络：5–60 kHz 或 5–50 kHz 均可

### 5.1.2 能量质心位置 $C_E$

- 名称：能量质心位置
- 符号：$C_E$
- 物理意义：衡量主要能量更靠前还是更靠后。
- 计算公式：

$$
C_E=\frac{\sum_t t\,e^2(t)}{\sum_t e^2(t)}
$$

- 来源公式：包络 $e(t)$ 由 Hilbert 解析包络得到。
- 建议频带：
  - 主推荐：5–60 kHz
  - 稳健替代：5–40 kHz

### 5.1.3 包络不对称度 $A_{\text{env}}$

- 名称：包络不对称度
- 符号：$A_{\text{env}}$
- 物理意义：表征“快起慢衰”和“拖尾”程度。
- 计算公式：

$$
A_{\text{env}}=\frac{(t_{off}-t_p)-(t_p-t_{on})}{t_{off}-t_{on}}
$$

- 来源公式：$t_{on},t_{off},t_p$ 均由包络边界与峰值位置得到。
- 建议频带：
  - 主推荐：5–60 kHz
  - 对噪声较大时可先做包络平滑

### 5.1.4 包络对称性 $S_{\text{env}}$

- 名称：包络对称性
- 符号：$S_{\text{env}}$
- 物理意义：鼓包越标准，左右镜像越接近。
- 计算公式：

$$
S_{\text{env}}=\mathrm{corr}(e,e^{rev}),\qquad e^{rev}[n]=e[N-1-n]
$$

- 来源公式：$e[n]$ 见包络公式；$\mathrm{corr}$ 用皮尔逊相关系数。
- 建议频带：
  - 主推荐：5–60 kHz
  - 动水鼓包识别可优先用主分析带

### 5.1.5 包络偏度 $Sk_{\text{env}}$

- 名称：包络偏度
- 符号：$Sk_{\text{env}}$
- 物理意义：反映包络向拖尾方向的偏斜程度。
- 计算公式：

$$
Sk_{\text{env}}=\frac{1}{N}\sum \frac{(e_i-\mu_e)^3}{\sigma_e^3}
$$

- 来源公式：
  - $\mu_e=\mathrm{mean}(e)$
  - $\sigma_e=\mathrm{std}(e)$
- 建议频带：5–60 kHz

### 5.1.6 上升下降时间比 $R_{td}$

- 名称：上升下降时间比
- 符号：$R_{td}$
- 物理意义：断丝常表现为“快起慢衰”。
- 计算公式：

$$
R_{td}=\frac{T_{\text{decay}}}{T_{\text{rise}}}
$$

- 来源公式：
  - $T_{\text{rise}}=(n_p-n_{10,\text{left}})/f_s$
  - $T_{\text{decay}}=(n_{10,\text{right}}-n_p)/f_s$
- 建议频带：
  - 主推荐：5–60 kHz
  - 低流速断丝：可用 5–40 kHz

### 5.1.7 前后能量比 $R_{fb}$

- 名称：前后能量比
- 符号：$R_{fb}$
- 物理意义：峰前/峰后能量不均衡。
- 计算公式：

$$
R_{fb}=\frac{E_{\text{before}}}{E_{\text{after}}}
$$

$$
E_{\text{before}}=\sum_{n=0}^{n_p} e[n]^2\Delta t,\qquad
E_{\text{after}}=\sum_{n=n_p}^{N-1} e[n]^2\Delta t
$$

- 建议频带：5–60 kHz

### 5.1.8 鼓包集中度 $C_{\text{bulge}}$

- 名称：鼓包集中度
- 符号：$C_{\text{bulge}}$
- 物理意义：描述鼓包主体在整段信号中的占比。
- 计算公式：

$$
C_{\text{bulge}}=\frac{T_\eta}{T}
$$

- 来源公式：$T_\eta$ 为归一化包络超过阈值 $\eta$ 的有效持续时间。
- 建议频带：5–60 kHz

### 5.1.9 局部鼓包个数 $N_{\text{bulge}}$

- 名称：局部鼓包个数
- 符号：$N_{\text{bulge}}$
- 物理意义：描述多鼓包背景复杂度。
- 计算方法：对包络做阈值分割与最小间隔约束后计数。
- 来源量：包络 $e[n]$、阈值 $\eta$、最小分离时间 $\tau_{\text{sep}}$。
- 建议频带：5–60 kHz

### 5.1.10 包络拟合误差 $\varepsilon_{\text{env}}$

- 名称：包络拟合误差
- 符号：$\varepsilon_{\text{env}}$
- 物理意义：刻画实际包络偏离“标准鼓包”的程度。
- 计算公式：

$$
\varepsilon_{\text{env}}=\frac{\|e-\hat e\|_2}{\|e\|_2}
$$

推荐拟合模型：

$$
\hat e(t)=
\begin{cases}
A\exp\!\left(-\dfrac{(t-\mu)^2}{2\sigma_l^2}\right), & t\le \mu \\
A\exp\!\left(-\dfrac{(t-\mu)^2}{2\sigma_r^2}\right), & t> \mu
\end{cases}
$$

- 建议频带：5–60 kHz

### 5.1.11 包络左右宽度比 $\eta_{\text{bw}}$

- 名称：包络左右宽度比
- 符号：$\eta_{\text{bw}}$
- 物理意义：衡量峰值左右展宽是否均衡。
- 计算公式：

$$
\eta_{\text{bw}}=\frac{t_{95}-t_p}{t_p-t_{05}}
$$

- 来源量：$t_{05},t_{95}$ 为包络累积或幅值百分位位置。
- 建议频带：5–60 kHz

---

## 5.2 能量与比例类

### 5.2.1 高频低频能量比 $R_{hl}(t)$

- 名称：高频低频能量比
- 符号：$R_{hl}(t)$
- 物理意义：衡量高频成分相对低频成分的占比。
- 计算公式：

$$
R_{hl}(t)=\frac{E_{\text{high}}(t)}{E_{\text{low}}(t)+\varepsilon}
$$

- 来源公式：
  - $E_{\text{high}}(t)=\sum_{f\in[20,40]\text{kHz}}P(t,f)$
  - $E_{\text{low}}(t)=\sum_{f\in[5,15]\text{kHz}}P(t,f)$
- 建议频带：
  - 高频：20–40 kHz
  - 低频：5–15 kHz
  - 若强调高流速背景：高频可改 20–60 kHz

### 5.2.2 高频低频能量比斜率 $k_{hl}$

- 名称：高频低频能量比斜率
- 符号：$k_{hl}$
- 物理意义：描述高频衰减速度。
- 计算公式：

$$
R_{hl}(t)\approx k_{hl} t+b_{hl}
$$

- 来源量：$R_{hl}(t)$。
- 建议频带：同 $R_{hl}(t)$。

### 5.2.3 高频衰减斜率 $\beta_H$

- 名称：高频衰减斜率
- 符号：$\beta_H$
- 物理意义：峰后高频能量下降的快慢。
- 计算公式：

$$
\log(E_H(t)+\varepsilon)\approx \beta_H t+b_H
$$

- 来源公式：$E_H(t)=\sum_{f\in B_H}S(t,f)$。
- 建议频带：
  - 主推荐：30–60 kHz
  - 可替代：25–45 kHz、25–60 kHz
  - 若设备上限不足：20–40 kHz

---

## 5.3 频域/时频演化类

### 5.3.1 谱质心 $SC(t)$

- 名称：谱质心
- 符号：$SC(t)$
- 物理意义：频谱中心频率位置。
- 计算公式：

$$
SC(t)=\frac{\sum_f fP(t,f)}{\sum_f P(t,f)}
$$

- 建议频带：
  - 主推荐：5–60 kHz
  - 低流速断丝：5–40 kHz

### 5.3.2 谱质心衰减斜率 $k_{sc}$

- 名称：谱质心衰减斜率
- 符号：$k_{sc}$
- 物理意义：整体频谱中心向低频迁移的速度。
- 计算公式：

$$
SC(t)\approx k_{sc}t+b_{sc}
$$

- 来源量：谱质心 $SC(t)$。
- 建议频带：5–60 kHz。

### 5.3.3 时频熵 $H_{tf}$

- 名称：时频熵
- 符号：$H_{tf}$
- 物理意义：能量在时频平面的离散程度。
- 计算公式：

$$
H_{tf}=-\sum_{i,j}p_{ij}\ln p_{ij}
$$

- 来源量：$p_{ij}=P[i,j]/\sum P[i,j]$。
- 建议频带：
  - 主推荐：5–60 kHz
  - 对局部窗/残差窗均可单独计算

### 5.3.4 频谱平坦度 $SF$

- 名称：频谱平坦度
- 符号：$SF$
- 物理意义：区分“窄峰+谐波”与“平坦宽带”。
- 计算公式：

$$
SF=\frac{\exp\!\left(\frac{1}{K}\sum\ln P_k\right)}{\frac{1}{K}\sum P_k}
$$

- 建议频带：5–60 kHz 或局部活动带。

### 5.3.5 Renyi 熵 $H_\alpha$

- 名称：Renyi 熵
- 符号：$H_\alpha$
- 物理意义：时频图集中度/复杂度。
- 计算公式：

$$
H_\alpha=\frac{1}{1-\alpha}\log\sum p_{tf}^{\alpha}
$$

- 建议频带：5–60 kHz；适合做补充增强特征。

---

## 5.4 二倍频/谐波/脊线结构类

### 5.4.1 主脊线连续率 $\rho_r$

- 名称：主脊线连续率
- 符号：$\rho_r$
- 物理意义：主结构是否稳定存在。
- 计算公式：

$$
\rho_r=\frac{L_{\text{valid}}}{L_{\text{all}}}
$$

- 来源量：$L_{\text{valid}}$ 为成功跟踪到主脊线的帧数。
- 建议频带：
  - 主脊线搜索：5–30 kHz
  - 谱图主分析带：5–60 kHz

### 5.4.2 脊线缺口率 $G_{\text{gap}}$

- 名称：脊线缺口率
- 符号：$G_{\text{gap}}$
- 物理意义：主结构是否被异常打断。
- 计算公式：

$$
G_{\text{gap}}=\frac{N_{\text{gap}}}{L_{\text{all}}}
$$

- 建议频带：5–30 kHz 主脊线搜索。

### 5.4.3 脊线拟合优度 $R^2_{\text{ridge}}$

- 名称：脊线拟合优度
- 符号：$R^2_{\text{ridge}}$
- 物理意义：主频轨迹是否具有稳定可建模结构。
- 计算公式：

$$
R^2_{\text{ridge}}=R^2(f_1,\hat f_1)
$$

- 建议频带：主脊线 5–30 kHz。

### 5.4.4 拱形轨迹分数 $S_{\text{arch}}$

- 名称：拱形轨迹分数
- 符号：$S_{\text{arch}}$
- 物理意义：是否符合“先升后降”的基频拱形轨迹。
- 计算公式：

$$
q(t)=at^2+bt+c
$$

$$
S_{\text{arch}}=R^2(f_1,q)\cdot\mathbf1(a<0)\cdot\mathbf1(t_v\in[0.2T,0.8T])
$$

- 建议频带：5–30 kHz 主脊线。

### 5.4.5 二倍频存在度 $H2_{\text{ratio}}$

- 名称：二倍频存在度
- 符号：$H2_{\text{ratio}}$
- 物理意义：2:1 关系在时间帧上的稳定出现率。
- 计算公式：

$$
H2_{\text{ratio}}=\frac{N\{|f_2(t)-2f_1(t)|<\Delta f\}}{N_{\text{active}}}
$$

- 建议频带：
  - $f_1$：5–30 kHz
  - $f_2$：10–60 kHz

### 5.4.6 二倍频比 $R_{2:1}$

- 名称：二倍频比
- 符号：$R_{2:1}$
- 物理意义：二倍频能量相对主频能量的强度。
- 计算公式：

$$
R_{2:1}=\frac{\sum_t E(t,2f_1(t)\pm \Delta f)}{\sum_t E(t,f_1(t)\pm \Delta f)+\varepsilon}
$$

- 建议频带：同上。

### 5.4.7 谐波栈能量比 $H_{\text{stack}}$

- 名称：谐波栈能量比
- 符号：$H_{\text{stack}}$
- 物理意义：主频及其多阶谐波的整体组织程度。
- 计算公式：

$$
H_{\text{stack}}=
\frac{\sum_t\sum_{m=1}^M E(t,mf_1(t)\pm\Delta f)}{\sum_{t,f}S(t,f)}
$$

- 建议频带：5–60 kHz；一般 $M=2$ 或 3。

### 5.4.8 谐波能量比 $R_h$

- 名称：谐波能量比
- 符号：$R_h$
- 物理意义：二倍频与基频能量比。
- 计算公式：

$$
R_h=\frac{E_{2nd}}{E_{1st}}
$$

- 来源量：$E_{1st}$、$E_{2nd}$ 为各自邻域积分能量。
- 建议频带：
  - 基频邻域：$f_1\pm 1$ kHz
  - 二倍频邻域：$f_2\pm 1$ kHz

### 5.4.9 二倍频一致性误差 $\varepsilon_{2\times}$ / $C_h$

- 名称：二倍频一致性误差
- 符号：$\varepsilon_{2\times}$ 或 $C_h$
- 物理意义：偏离严格 2:1 谐波关系的程度。
- 计算公式：

$$
\varepsilon_{2\times}=\mathrm{median}\!\left(\frac{|f_2-2f_1|}{f_1}\right)
$$

或

$$
C_h=\mathrm{mean}(|f_2-2f_1|)
$$

- 建议频带：$f_1$ 取 5–30 kHz，$f_2$ 取 10–60 kHz。

### 5.4.10 谐波能量占比 $R_{\text{harm}}$ / 脊线连续性 $Ridge_{\text{coh}}$

- 名称：谐波能量占比/脊线连续性
- 符号：$R_{\text{harm}}$、$Ridge_{\text{coh}}$
- 物理意义：总能量中有多少集中于主频和二倍频结构。
- 计算公式：

$$
R_{\text{harm}}=Ridge_{\text{coh}}=\frac{E_{\text{ridge}}}{E_{\text{total}}}
$$

- 来源公式：

$$
E_{\text{ridge}}=\sum M_h(i,k)P(i,k)
$$

- 建议频带：5–60 kHz。

### 5.4.11 上升帧占比 $\rho_{up}$、下降帧占比 $\rho_{down}$、转折次数 $N_{\text{turn}}$

- 名称：脊线局部漂移统计
- 符号：$\rho_{up}$、$\rho_{down}$、$N_{\text{turn}}$
- 物理意义：适用于“半个鼓包”“多鼓包”以及非完整拱形窗口。
- 计算公式：

$$
\frac{df_1}{dt}\approx \frac{f_1(t_i)-f_1(t_{i-1})}{t_i-t_{i-1}}
$$

$$
\rho_{up}=\frac{N\{df_1/dt>\tau_s\}}{N_{\text{active}}},\qquad
\rho_{down}=\frac{N\{df_1/dt<-\tau_s\}}{N_{\text{active}}}
$$

$$
N_{\text{turn}}=N\{\mathrm{sign}(df_1/dt)\text{发生变化}\}
$$

- 建议频带：5–30 kHz 主脊线。

### 5.4.12 频率跨度 $\Delta f_{\text{span}}$ 与归一化曲率 $C_f$

- 名称：频率跨度、归一化曲率
- 符号：$\Delta f_{\text{span}}$、$C_f$
- 物理意义：表征基频摆动幅度与弯曲程度。
- 计算公式：

$$
\Delta f_{\text{span}}=\max(f_1)-\min(f_1)
$$

$$
C_f=\mathrm{mean}\left(\left|\frac{d^2f_1}{dt^2}\right|\right)
$$

- 建议频带：5–30 kHz 主脊线。

---

## 5.5 残差与背景扣除类

### 5.5.1 谐波可解释能量 $E_{\text{harm}}$

- 名称：谐波可解释能量
- 符号：$E_{\text{harm}}$
- 物理意义：可由标准动水谐波结构解释的能量。
- 计算公式：

$$
E_{\text{harm}}=\sum_{i,k}M_h(i,k)P(i,k)
$$

- 建议频带：5–60 kHz。

### 5.5.2 残差能量 $E_{\text{res}}$

- 名称：残差能量
- 符号：$E_{\text{res}}$
- 物理意义：规则背景之外的异常部分。
- 计算公式：

$$
E_{\text{res}}=E_{\text{total}}-E_{\text{harm}}
$$

- 建议频带：5–60 kHz。

### 5.5.3 非谐波残差占比 $\rho_{\text{res}}$

- 名称：非谐波残差占比
- 符号：$\rho_{\text{res}}$
- 物理意义：异常成分占总能量比例。
- 计算公式：

$$
\rho_{\text{res}}=\frac{E_{\text{res}}}{E_{\text{total}}}
$$

- 建议频带：5–60 kHz。

### 5.5.4 高频残差占比 $R_{\text{high,res}}$

- 名称：高频残差占比
- 符号：$R_{\text{high,res}}$
- 物理意义：高频异常是否显著。
- 计算公式：

$$
R_{\text{high,res}}=\frac{E_{\text{res,high}}}{E_{\text{total}}}
$$

- 来源公式：

$$
E_{\text{res,high}}(t_i)=\sum_{f\in B_{RH}} P_{\text{res}}(t_i,f)
$$

- 建议频带：
  - 主推荐：20–60 kHz
  - 更保守：25–60 kHz
  - 前端上限不足：20–40 kHz

### 5.5.5 流噪声重构误差 $\varepsilon_{\text{rec}}$

- 名称：流噪声重构误差
- 符号：$\varepsilon_{\text{rec}}$
- 物理意义：纯流噪声主结构是否足以解释原信号。
- 计算公式：

$$
\varepsilon_{\text{rec}}=\frac{\|x-\hat x_f\|_2^2}{\|x\|_2^2}
$$

- 来源量：$\hat x_f$ 由谐波掩膜重构得到。
- 建议频带：5–60 kHz。

### 5.5.6 异常子窗个数 $N_{\text{abn}}$

- 名称：异常子窗个数
- 符号：$N_{\text{abn}}$
- 物理意义：异常局部瞬态出现次数。
- 计算公式：

$$
N_{\text{abn}}=N\{\rho_{\text{res}}^{(m)}>\tau_{\text{res}}\}
$$

- 来源量：局部滑窗残差占比 $\rho_{\text{res}}^{(m)}$。
- 建议频带：主分析 5–60 kHz，残差高频 20–60 kHz。

---

## 5.6 瞬态/冲击/高阶统计类

### 5.6.1 谱流量峰值 $F_{\text{peak}}$

- 名称：谱流量峰值
- 符号：$F_{\text{peak}}$
- 物理意义：帧间突变强度，对附加瞬态敏感。
- 计算公式：

$$
Flux(t_i)=\sum_j \max(P(t_i,f_j)-P(t_{i-1},f_j),0)
$$

$$
F_{\text{peak}}=\max_i Flux(t_i)
$$

- 建议频带：5–60 kHz；也可专用于残差谱。

### 5.6.2 TKEO 峰值比 $R_{\text{tkeo}}$

- 名称：TKEO 峰值比
- 符号：$R_{\text{tkeo}}$
- 物理意义：局部尖锐瞬态程度。
- 计算公式：

$$
R_{\text{tkeo}}=\frac{\max(\Psi[n])}{\mathrm{mean}(\Psi[n])}
$$

- 来源公式：$\Psi[n]=x[n]^2-x[n-1]x[n+1]$。
- 建议频带：
  - 主推荐：5–60 kHz
  - 强调冲击尖锐度时可先做 RMS 归一化

### 5.6.3 残差 TKEO 峰值比 $R_{\text{res,tkeo}}$

- 名称：残差 TKEO 峰值比
- 符号：$R_{\text{res,tkeo}}$
- 物理意义：残差中尖锐异常瞬态的强度。
- 计算公式：与 $R_{\text{tkeo}}$ 相同，但输入换为 $x_{\text{res}}[n]$。
- 建议频带：残差重构后时域波形；辅以 20–60 kHz 残差带。

### 5.6.4 局部峭度 $K_{\text{loc}}$ / 残差局部峭度最大值 $K_{\text{res,max}}$

- 名称：局部峭度、残差局部峭度最大值
- 符号：$K_{\text{loc}}$、$K_{\text{res,max}}$
- 物理意义：尖峰性、冲击性。
- 计算公式：

$$
K=\frac{\frac1N\sum(x-\mu)^4}{\sigma^4}
$$

- 建议频带：
  - 局部峭度：5–60 kHz 或高频子带
  - 残差局部峭度：20–60 kHz 残差带优先

### 5.6.5 谱峭度峰值 $SK_{\max}$

- 名称：谱峭度峰值
- 符号：$SK_{\max}$
- 物理意义：定位“哪个频带含有瞬态异常”。
- 计算公式：

$$
\hat{SK}(k)=\frac{\langle |X_m(k)|^4\rangle}{\langle |X_m(k)|^2\rangle^2}-2
$$

$$
SK_{\max}=\max_k \hat{SK}(k)
$$

- 建议频带：
  - 主推荐：全带 5–60 kHz
  - 更聚焦异常：20–60 kHz

### 5.6.6 峰均比 $CF_{\text{res}}$

- 名称：峰均比
- 符号：$CF_{\text{res}}$
- 物理意义：尖峰程度与均方能量的比值。
- 计算公式：

$$
CF_{\text{res}}=\frac{\max |x_{\text{res}}|}{\mathrm{RMS}(x_{\text{res}})}
$$

- 建议频带：残差时域信号。

### 5.6.7 高频半衰期 $T_{\text{half,high}}$

- 名称：高频半衰期
- 符号：$T_{\text{half,high}}$
- 物理意义：高频成分衰减到一半所需时间。
- 计算方法：对 $x_{H1}[n]$ 包络求峰后首次跌至 $50\%$ 峰值时刻。
- 建议频带：
  - 主推荐：20–40 kHz
  - 若需要更高频：25–60 kHz

---

## 5.7 残差演化类

### 5.7.1 残差谱质心 $SC_{\text{res}}(t)$ 与斜率 $k_{\text{res,sc}}$

- 名称：残差谱质心、残差谱质心斜率
- 符号：$SC_{\text{res}}(t)$、$k_{\text{res,sc}}$
- 物理意义：异常分量的中心频率及其下移速度。
- 计算公式：

$$
SC_{\text{res}}(t)=\frac{\sum_f fP_{\text{res}}(t,f)}{\sum_f P_{\text{res}}(t,f)}
$$

$$
SC_{\text{res}}(t)\approx k_{\text{res,sc}}t+b
$$

- 建议频带：残差带 5–60 kHz 或 20–60 kHz。

### 5.7.2 残差高低频能量比 $R_{\text{res,hl}}(t)$ 与斜率 $k_{\text{res,hl}}$

- 名称：残差高低频能量比、残差高低频能量比斜率
- 符号：$R_{\text{res,hl}}(t)$、$k_{\text{res,hl}}$
- 物理意义：异常残差中的高频占比及其衰减速度。
- 计算公式：

$$
R_{\text{res,hl}}(t)=\frac{E_{\text{res,high}}(t)}{E_{\text{res,low}}(t)+\varepsilon}
$$

- 建议频带：
  - 残差低频：5–15 kHz
  - 残差高频：20–60 kHz

### 5.7.3 残差包络对称性 $S_{\text{res,env}}$

- 名称：残差包络对称性
- 符号：$S_{\text{res,env}}$
- 物理意义：异常分量是否呈拖尾型衰减。
- 计算公式：

$$
S_{\text{res,env}}=\mathrm{corr}(e_{\text{res}},e_{\text{res}}^{rev})
$$

- 建议频带：残差重构波形。

### 5.7.4 残差前后不对称度 $\eta_{\text{asym}}$

- 名称：残差前后不对称度
- 符号：$\eta_{\text{asym}}$
- 物理意义：触发后能量是否显著偏向后沿。
- 计算公式：

$$
\eta_{\text{asym}}=\frac{E[t_0,t_0+\tau]}{E[t_0-\tau,t_0]}
$$

- 建议参数：$\tau=0.3$–$1.0$ ms。
- 建议频带：残差重构波形或残差高频带。

---

## 5.8 小波/多分辨率类

### 5.8.1 小波包节点能量比 $R_{\text{wp},i}$

- 名称：小波包节点能量比
- 符号：$R_{\text{wp},i}$
- 物理意义：多尺度子带能量分布。
- 计算公式：

$$
R_{\text{wp},i}=\frac{E_{\text{wp},i}}{\sum_j E_{\text{wp},j}}
$$

- 建议频带：
  - 主推荐：对 $x_B[n]$ 或 $x_{\text{res}}[n]$ 做 WPT
  - 子带宽度建议：2–5 kHz

### 5.8.2 小波熵 $H_{\text{wp}}$

- 名称：小波熵
- 符号：$H_{\text{wp}}$
- 物理意义：多分辨率能量分散度。
- 计算公式：

$$
H_{\text{wp}}=-\sum_i p_i\ln p_i,\qquad p_i=\frac{E_{\text{wp},i}}{\sum_jE_{\text{wp},j}}
$$

- 建议频带：对主分析带或残差做 WPT。

### 5.8.3 高频子带能量突发指数 $I_{\text{burst}}$

- 名称：高频子带能量突发指数
- 符号：$I_{\text{burst}}$
- 物理意义：高频短时突发程度。
- 计算公式：

$$
I_{\text{burst}}=\frac{\max(E_{\text{wp}}(t))}{\mathrm{median}(E_{\text{wp}}(t))}
$$

- 建议频带：高频小波包节点优先。

### 5.8.4 高频-低频能差 $D_{\text{WPT}}$

- 名称：高频-低频能差
- 符号：$D_{\text{WPT}}$
- 物理意义：能量是否从谐波相关子带偏向高频异常子带。
- 计算公式：

$$
D_{\text{WPT}}=\sum_{i\in HF}p_i-\sum_{i\in Harm}p_i
$$

- 建议频带：
  - 高频节点集合：对应 20–60 kHz
  - 谐波节点集合：对应主频/二倍频附近

---

## 5.9 阻尼模型类

### 5.9.1 阻尼原子最大匹配度 $C_{\text{damp}}$

- 名称：阻尼原子最大匹配度
- 符号：$C_{\text{damp}}$
- 物理意义：残差是否像“阻尼振荡”。
- 计算公式：

$$
C_{\text{damp}}=\frac{\max_\theta |\langle r,g_\theta\rangle|}{\|r\|\,\|g_\theta\|}
$$

- 来源公式：$g_\theta(t)=e^{-(t-t_0)/\tau_d}\cos(2\pi f_d(t-t_0)+\phi)u(t-t_0)$。
- 建议参数：
  - $f_d$：8–45 kHz
  - $\tau_d$：0.1–2.0 ms

### 5.9.2 残差衰减常数 $\hat\alpha$

- 名称：残差衰减常数
- 符号：$\hat\alpha$
- 物理意义：指数衰减速度。
- 计算公式：

$$
\log e_{\text{res}}(t)\approx -\hat\alpha t+b
$$

- 来源量：$e_{\text{res}}(t)$ 为残差包络。
- 建议时间窗：触发后 0.3–1.5 ms。

### 5.9.3 矩阵铅笔拟合优度 $Q_{\text{MP}}$

- 名称：矩阵铅笔拟合优度
- 符号：$Q_{\text{MP}}$
- 物理意义：指数阻尼正弦模型对残差的解释程度。
- 计算方式：矩阵铅笔法拟合前后误差下降率。
- 建议频带：候选残差窗；优先在高置信候选上运行。

---

## 5.10 稀疏表示与模型竞争类

### 5.10.1 加入损伤原子后的误差下降率 $\Delta J$

- 名称：加入损伤原子后的误差下降率
- 符号：$\Delta J$
- 物理意义：新加入一个损伤分量后，整体解释是否明显变好。
- 计算公式：

$$
\Delta J=\frac{\|x-\hat x_f\|_2^2-\|x-\hat x_f-\hat x_d\|_2^2}{\|x\|_2^2}
$$

- 来源量：
  - $\hat x_f$：流噪声主结构重构
  - $\hat x_d$：阻尼原子、小字典或稀疏求解得到的损伤分量
- 建议频带：5–60 kHz，必要时局部候选窗。

### 5.10.2 字典损伤系数比 $\eta_{\text{dict}}$

- 名称：字典损伤系数比
- 符号：$\eta_{\text{dict}}$
- 物理意义：稀疏分解中更像“损伤字典”还是“流噪声字典”。
- 计算公式：

$$
\eta_{\text{dict}}=\frac{\|\alpha_d\|_1}{\|\alpha_f\|_1+\|\alpha_d\|_1}
$$

- 来源量：
  - $D_f$：流噪声字典
  - $D_d$：损伤字典
  - $\alpha_f,\alpha_d$：OMP/MP/K-SVD 系数
- 建议频带：5–60 kHz；候选窗优先。

---

## 6. 推荐实现优先级

### 6.1 第一批必须实现

- $A_{\text{env}}$
- $C_E$
- $R_{td}$
- $k_{sc}$
- $k_{hl}$
- $R_{2:1}$
- $\varepsilon_{2\times}$ 或 $C_h$
- $\rho_r$
- $Ridge_{\text{coh}}$ / $R_{\text{harm}}$
- $\varepsilon_{\text{rec}}$
- $R_{\text{high,res}}$
- $SK_{\max}$
- $R_{\text{res,tkeo}}$ 或 $R_{\text{tkeo}}$
- $F_{\text{peak}}$

### 6.2 第二批增强实现

- $S_{\text{arch}}$
- $\rho_{up}, \rho_{down}, N_{\text{turn}}$
- $N_{\text{abn}}$
- $k_{\text{res,sc}}, k_{\text{res,hl}}$
- $C_{\text{damp}}$
- $\Delta J$
- $D_{\text{WPT}}$
- $I_{\text{burst}}$

### 6.3 第三批进阶实现

- $Q_{\text{MP}}$
- $\eta_{\text{dict}}$
- $H_\alpha$
- $SF$
- $K_{\text{res,max}}$

---

## 7. 同义项统一建议

为避免后续编码与汇报混乱，建议统一采用下列口径：

| 原始写法 | 统一建议 |
|---|---|
| 二倍频一致性误差、谐波一致性误差 | 优先统一为 $\varepsilon_{2\times}$，保留 $C_h$ 作为绝对误差版本 |
| 谐波能量占比、脊线连续性 | 在能量意义上可合并描述，优先统一为 $R_{\text{harm}}$ |
| 高频残差占比、残差高频能量比 | 若分母为总能量，统一为 $R_{\text{high,res}}$；若分母为残差低频，统一为 $R_{\text{res,hl}}$ |
| 脊线拟合优度、拱形轨迹分数 | 前者泛指拟合优度 $R^2_{\text{ridge}}$，后者特指“先升后降”约束下的 $S_{\text{arch}}$ |
| TKEO 峰值比、残差 TKEO 峰值比 | 输入不同，公式相同，分别记为 $R_{\text{tkeo}}$ 与 $R_{\text{res,tkeo}}$ |
| 小波包能量差、小波高低频差 | 统一为 $D_{\text{WPT}}$ |

---

## 8. 最终建议的标准化特征集合

若要形成一套统一、可落地、覆盖三份文档主张的标准化特征向量，建议优先采用：

$$
\mathbf{z}_{\text{std}}=
\left[
A_{\text{env}},
\ C_E,
\ R_{td},
\ k_{sc},
\ k_{hl},
\ R_{2:1},
\ \varepsilon_{2\times},
\ \rho_r,
\ R_{\text{harm}},
\ \varepsilon_{\text{rec}},
\ R_{\text{high,res}},
\ SK_{\max},
\ R_{\text{res,tkeo}},
\ C_{\text{damp}},
\ \Delta J
\right]
$$

这组特征同时覆盖：

- 断丝本征瞬态：$A_{\text{env}},R_{td},k_{sc},k_{hl},R_{\text{res,tkeo}},SK_{\max}$
- 流噪声主结构：$R_{2:1},\varepsilon_{2\times},\rho_r,R_{\text{harm}}$
- 背景扣除与残差异常：$\varepsilon_{\text{rec}},R_{\text{high,res}},C_{\text{damp}},\Delta J$

---

## 9. 备注

- 若样本短且窗口常截断，应统一采用“**局部滑窗计算 + 最大值/95%分位数/计数聚合**”。
- 若设备高频响应不足，应优先保留“结构类特征”而弱化“绝对高频能量类特征”。
- 若前端存在削顶或 AGC，应弱化 $K_{\text{loc}}$、$CF_{\text{res}}$、峰值类指标，强化 $\rho_r$、$R_{2:1}$、$\varepsilon_{2\times}$、$S_{\text{arch}}$、$\Delta J$。
- 若后续要编码，建议先将本文第 2 节“基础量”单独实现成公共函数，再派生全部特征。
