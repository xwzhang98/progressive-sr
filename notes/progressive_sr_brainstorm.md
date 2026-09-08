# Lagrangian Progressive 2× Super-Resolution：理论 brainstorm

*给潇文的研究笔记，2026-09-08。目标：把"一个模型每次 2×、autoregressive 地往上 upsample"这个想法，在 Lagrangian（particles on the initial grid）描述下写成一个数学上自洽的问题，并给出从哪里下手。*

---

## 0. 一句话版本

把 N-body 的 Lagrangian 位移场 $\Psi_\ell(\mathbf q)$ 看成一个"尺度上的 Markov 链"：每一级 $\ell\to\ell+1$ 只做两件事，(i) 修正粗网格场本身的 UV 误差（小、近似确定性、可用微扰论刻画），(ii) 生成一个新的 octave 的细节（其随机性恰好等于该 octave 的初条件模式）。线性理论、2LPT 的 $k^2$ 压低、以及 separate-universe/tidal response 告诉我们这个 conditional 在 Lagrangian 坐标下是**局域的、近高斯的、并且在无量纲化之后近似尺度不变的**——这三条性质正是"一个共享权重的 2× 算子 + autoregressive"能成立的数学前提。Wavelet 在这个框架里不是黑箱算子，而是 restriction/prolongation 的选择；FNO 则只适合做线性/谱的那部分，不适合生成新 octave。

---

## 1. 这个想法在文献里的位置

**AI-SR 系列本身。** Paper I（Li et al. 2021, PNAS, arXiv:2010.06608）的 generator 其实已经是 3 级 2× 的 ladder（StyleGAN2 风格，每级注入 noise，LR 先 tri-linear 上采样再拼接），只是中间两级没有被当作"物理上的分辨率层级"来监督，noise 也没有被赋予物理含义。Paper III（arXiv:2305.12222）加了 redshift 条件化，Paper IV（arXiv:2408.09051）证明了：给定 HR 初条件（HRIC）后，SR 可以变成一个 deterministic emulator（U-Net，Lagrangian Charbonnier + Eulerian CIC loss + 弱对抗项）。Paper V 加了 cosmology 条件化。**Progressive 2× 的新意在于**：把 I 的内部 ladder 变成显式的、每级都有物理监督的层级，把 IV 的"noise = 初条件"从一次性 8× 推广到逐 octave，并让所有层级共享同一个算子。

**ML 侧的多尺度生成。** 这一路的理论最成熟的是 Mallat 组：Wavelet Score-based Generative Modeling（Guth, Coste, De Bortoli, Mallat 2022, arXiv:2208.05003）证明按 wavelet 尺度逐级做 conditional diffusion，每一级所需步数与分辨率无关（高斯情形严格证明，$\varphi^4$ 和自然图像数值验证）；Wavelet Conditional RG（Marchand, Ozawa, Biroli, Mallat 2022, arXiv:2207.04941）把这件事写成 RG：逐尺度估计 $p(\text{details}_\ell\mid\text{coarse}_\ell)$ 的能量模型，coarse-to-fine 采样绕过 critical slowing down，并在弱引力透镜图上做了演示；Conditionally Strongly Log-Concave models（Guth, Lempereur, Bruna, Mallat 2023, arXiv:2306.00181）进一步证明：对 $\varphi^4$ 和弱透镜 convergence map，虽然整体分布远非 log-concave，但 wavelet-conditional 分解后每一级的条件分布是强 log-concave 的，采样和参数估计都有保证。工程侧的对应物是 VAR 的 next-scale prediction（Tian et al. 2024, arXiv:2404.02905）、Laplacian multi-scale flow matching（Zhao et al. 2026, arXiv:2602.19461）、以及"多尺度 AR 模型就是伪装的 Laplacian diffusion"（Hong & Belkadi 2025, arXiv:2510.02826）。

**宇宙学侧。** Diffusion SR（Schanz, List, Hahn 2023, arXiv:2310.06929）是 2D Eulerian 的；Cosmo3DFlow（arXiv:2602.10172）在 wavelet 系数空间做 flow matching 但用于 IC 重建、也是 Eulerian 密度场。就我这次检索到的范围，**还没有人在 Lagrangian 位移场上做逐 octave 的 conditional 生成**，这是一个空档。

**FNO 的 zero-shot 超分辨。** 2025–26 有三篇专门泼冷水的：Colagrande et al. 2026（arXiv:2606.00677）指出 FNO 在未见过的离散化上运行时，非线性激活产生的 aliasing 会把伪高频折回已学习频带，直接在细网格推理往往不如"粗网格推理 + Fourier zero-padding 上采样"；"The False Promise of Zero-Shot Super-Resolution"（arXiv:2510.06646）把它归结为 out-of-distribution 问题，且指出模型无法 extrapolate 到没见过的频率信息；Subedi & Tewari（arXiv:2606.00296）从信息论上证明：没有额外结构假设时 zero-shot SR 不可能，只有当输出函数满足 Hölder 连续性（即细网格上没有新信息）时才有保证。对我们的问题，结论很干脆：新的 octave 是**新信息**，恰恰违反那个 Hölder 假设，本来就不该指望 discretization-invariance 变出来（§8）。

---

## 2. 问题的数学表述

### 2.1 Lagrangian 多分辨率层级

盒长 $L$，第 $\ell$ 级每维 $N_\ell=N_0 2^\ell$ 个粒子，网格间距 $h_\ell=L/N_\ell$，粒子质量 $m_\ell=m_0 8^{-\ell}$，Lagrangian 格点 $\Lambda_\ell=h_\ell\mathbb Z^3/L\mathbb Z^3$。场：$\Psi_\ell:\Lambda_\ell\to\mathbb R^3$（位移），$\mathbf v_\ell$（速度）。第 $\ell$ 级的 N-body 解写成一个算子
$$
(\Psi_\ell,\mathbf v_\ell)=\mathcal S_\ell[\delta_{\rm lin}],\qquad \mathcal S_\ell \text{ 带有 } \epsilon_\ell\propto h_\ell \text{ 的力软化},
$$
输入只包含 $|k_i|<k_{\rm Ny,\ell}=\pi/h_\ell$ 的线性模式。

### 2.2 Nested initial conditions 与"octave"

用同一个 white-noise 种子，逐级截断（Li+21 的 LR/HR 就是这样共享大尺度模式的，MUSIC 的 zoom-in IC 也是同一个构造）：
$$
\delta_{\ell+1}=\delta_\ell+\eta_\ell,\qquad \eta_\ell \text{ 只含 } \mathcal O_\ell:=[-k_{{\rm Ny},\ell+1},k_{{\rm Ny},\ell+1}]^3\setminus[-k_{{\rm Ny},\ell},k_{{\rm Ny},\ell}]^3 \text{ 里的模式}.
$$
$\eta_\ell$ 是独立高斯，功率谱已知（$P_{\rm lin}$ 限制在 octave 上）。**自由度计数**：$\mathcal O_\ell$ 含 $7\cdot 8^\ell N_0^3$ 个模式，恰好等于每个粗 cell 新增的 7 个粒子——不论用哪种 wavelet 分解，每级每分量注入的随机数个数都是固定的 $7N_\ell^3$。这个"信息预算"后面会用来约束模型的 noise 维度。

### 2.3 两种不同的 conditional，必须分清

- **(A) 物理 conditional**：$p\big(\mathcal S_{\ell+1}[\delta_\ell+\eta_\ell]\;\big|\;\mathcal S_\ell[\delta_\ell]\big)$，对 $\delta_\ell,\eta_\ell$ 取联合分布。这是 Paper I–V 学的东西（只不过一步 8×）。粗场是一个**真实跑出来的** LR 模拟。
- **(B) 限制 conditional**：取一个线性限制算子 $R:\Lambda_{\ell+1}\to\Lambda_\ell$，学 $p(\Psi_{\ell+1}\mid R\Psi_{\ell+1})$。这是 WSGM / WC-RG / VAR 学的东西。粗场是细场的**确定性函数**。

两者的差是
$$
\varepsilon_\ell:=R\,\mathcal S_{\ell+1}[\delta_\ell+\eta_\ell]-\mathcal S_\ell[\delta_\ell],
$$
即"缺失的 octave 对已分辨尺度的反作用"（UV backreaction，Nishimichi, Bernardeau & Taruya 2016, arXiv:1411.2970 的 response function 量化的就是它）。Progressive 模型必须同时处理 (A) 和 (B)：第一步的输入是真实 LR（属于 A），之后每一步的输入是自己上一步的输出（更接近 B）。**这是 progressive 方案独有的分布漂移问题**，§9 再谈。

---

## 3. 分解定理：wavelet 就是选 $(R,P,W)$

任取线性 $R$ 及其右逆 $P$（$RP=I$，prolongation/插值），令 $Q=PR$，则 $Q$ 是投影，$I-Q$ 是到 $\ker R$ 的投影（因为 $R(I-PR)=R-R=0$）。于是任何细场唯一分解为
$$
\Psi_{\ell+1}=\underbrace{P\,R\Psi_{\ell+1}}_{\text{粗部分的插值}}+\underbrace{(I-PR)\Psi_{\ell+1}}_{=:\,W d_\ell,\ \text{细节}}\,,
$$
$W$ 是 $\ker R$ 的一组基（合成高通），$d_\ell$ 是细节系数。**一个 biorthogonal wavelet 无非就是一组满足 $RP=I,\ RW=0$ 的 $(R,P,W)$**；lifting scheme（Sweldens）是构造它们的通用方法：predict（用粗点插值新点）+ update（保证 $R$ 的矩条件）。所以"用 wavelet 分解"这个提议，落到实处是回答**三个问题**：$R$ 选什么、$P$ 选什么、$W$ 的 range 是否恰好是"物理上新增的自由度"。

把整个 2× 步骤写成
$$
\boxed{\ \hat\Psi_{\ell+1}=P\big(\Psi_\ell+\hat\varepsilon_\ell\big)+W\,\hat d_\ell\ }\qquad
\hat\varepsilon_\ell\sim p(\varepsilon\mid\Psi_\ell),\quad \hat d_\ell\sim p(d\mid \Psi_\ell+\hat\varepsilon_\ell,\ \ldots)
$$
两头结构：**correction head**（修粗场）+ **detail head**（生细节）。这个结构自动保证 $R\hat\Psi_{\ell+1}=\Psi_\ell+\hat\varepsilon_\ell$，即层级之间的一致性由代数保证而不是靠 loss 学出来。

### 3.1 三个候选 $R$，性质完全不同

| $R$ | $P$ | 物理含义 | 线性理论下 $\varepsilon^{(1)}$ | 局域性 |
|---|---|---|---|---|
| Haar：8 个细粒子块平均 | 常数外推 | 粗粒子 = 8 个细粒子的质心，**质量与动量严格守恒** | $\neq0$：粗模式被每维 $\cos(k_ih_{\ell+1}/2)$ 型低通衰减并带半个细格相移（确定性），octave 模式经衰减后折回粗带（**随机**） | 完全局域 |
| 子采样：取角上那个粒子 | Deslauriers–Dubuc 插值 | 粗粒子是细粒子的子集（interpolating scheme） | $\neq0$：octave 模式不衰减地折回粗带（aliasing，**随机**） | 局域 |
| Fourier 截断 $R_F$（各向同性窗更好） | zero-padding（sinc 插值） | 粗场 = 细场的带限部分 | **$=0$**，且 $R_F$ 与线性演化算子对易 | 非局域，halo 附近有 Gibbs 振铃 |

关键观察：只有 Fourier 型的 $R$ 与线性演化对易，此时在 nested IC 下 $R_F\Psi^{(1)}_{\ell+1}=\Psi^{(1)}_\ell$ **恒等成立**，correction 完全来自模式耦合（§5）。Haar 与子采样则在线性阶就给 correction 引入了一个**随机的** aliasing 分量（octave 折回粗带），把一个本来为零的项塞给了网络，并且让 correction head 也必须是生成式的。

我推荐的具体选择是**各向同性的锐截断球**：$W(k)=\mathbb 1[|k|<\alpha\,\pi/h_\ell]$，$\alpha\le1$。理由有两层。一是投影性：$W\in\{0,1\}$ 才有 $RP=I$、$PR=W$ 严格成立（一个平滑 taper 会破坏 $RP=I$，粗、细子空间不再互补；taper 如果要用，只能用在生成模型 source 分布的 partition of unity 里，不能用在 $R$ 上）。二是各向同性：cubic 网格的 octave $\mathcal O_\ell$ 是一个"立方体环"，角落里 $|k|$ 可以到 $\sqrt3\,\pi/h_\ell$，这些模式 LR 模拟本来就分辨不好（Paper I 讨论过 LR 在 Nyquist 附近不可靠）；用球截断等于把粗立方体的角落模式也划归"细节"，于是模型的输入 LR 场先被投影到球内（一个确定性的线性"清洗"），2× 步骤再生成球壳 $\alpha\pi/h_\ell<|k|<\alpha\pi/h_{\ell+1}$ 内的全部内容。链上每一级的场都是自身球带限的，层级之间严格一致。代价是锐截断在实空间有 sinc 型振铃，Phase 0 (a) 里与 Haar 并排比较 $\varepsilon$ 的幅度就能看出这个代价值不值。

### 3.2 数据驱动的"最优 $R$"

$R$ 是建模选择，不是物理定律。可以直接问：哪个线性 $R$ 让 $\varepsilon_\ell$ 最小？在各向同性、平移不变的限制下答案是 Wiener 型滤波
$$
T_\ell(k)=\frac{P_{\Psi_\ell\times\Psi_{\ell+1}}(k)}{P_{\Psi_{\ell+1}}(k)},
$$
直接从一对 LR/HR 模拟测出来。我预期它在 $k\ll k_{\rm Ny,\ell}$ 为 1，在 Nyquist 前就开始滚落。这既是 Phase 0 里最便宜的一个测量，本身也是一个小结果："LR N-body 最像 HR 的什么样的 coarse-graining"。

---

## 4. 线性理论告诉我们的结构

nested IC + $R_F$ 下，线性位移场满足
$$
\Psi^{(1)}_{\ell+1}=P_F\Psi^{(1)}_\ell+\Psi^{(1)}[\eta_\ell],
$$
即**细节与粗场在线性阶完全独立**，$p(d_\ell\mid\Psi_\ell)=p(d_\ell)$ 是一个功率谱已知的高斯。这给出三个直接的设计结论。

**(i) 把线性解析部分硬编码。** 模型输出应写成
$$
d_\ell=\underbrace{G_\ell^{1/2}\,\xi_\ell}_{\text{octave 的线性理论}}+f_\theta\big(\Psi_\ell,\ \xi_\ell;\ s_\ell\big),\qquad \xi_\ell\sim\mathcal N(0,I),
$$
$G_\ell$ 是 octave 的线性位移协方差（Fourier 对角，含 $D(z)$），$f_\theta$ 是非线性修正，高红移时趋于零。这样网络只学"非线性偏离线性理论多少"，和 Paper IV 里"emulator 学 SR 偏离 HR 多少"是同一种残差思想。

**(ii) Noise 有了物理身份。** $\xi_\ell$ 就是 octave 的 white noise，维度恰好 $7N_\ell^3\times3$（§2.2 的信息预算）。于是同一个网络有两种训练模式，理论上应当给出同一个模型：
- *配对/确定性模式*：把训练模拟真实的 $\eta_\ell$ 喂进去，$\xi_\ell=G_\ell^{-1/2}\Psi^{(1)}[\eta_\ell]$，用回归 loss（Charbonnier + CIC）。这就是 Paper IV 的设定，逐级化。
- *生成模式*：$\xi_\ell$ 随机采样，做分布匹配。

理论依据：对确定性模拟器 + nested IC，物理 conditional (A) **精确地**等于高斯 $\mathcal N(0,G_\ell)$ 经映射 $(\delta_\ell,\eta)\mapsto\mathcal S_{\ell+1}[\delta_\ell+\eta]$ 的 pushforward。如果回归学到了这个映射，重新采样 $\xi$ 自动给出正确的条件分布，原则上不需要 GAN 或 diffusion。

**(iii) 但映射在 halo 内部是混沌的。** 位相空间里 $\eta$ 的微小差别在 virialized 区域给出 $O(1)$ 不同的粒子位置（Paper IV 里 95th percentile 位移误差 $\sim1.3\,h^{-1}$Mpc 就是这个），MSE 回归会给出条件均值（模糊）。所以现实方案是：单流区用回归（微扰论准确），多流区用生成（§10）。这个"按 Lagrangian 区域分工"是 progressive + Lagrangian 特有的便利：在 Lagrangian 坐标下多流区就是 $\det(\partial\mathbf x/\partial\mathbf q)$ 变号过的 patch，粗场本身就能给出这个 mask。

---

## 5. 二阶微扰论：correction 为何小、为何局域，细节为何只依赖局部 jet

把线性场分成长短两部分 $\delta=\delta_L+\eta$（$L$ = 粗带，$\eta$ = octave）。2LPT 位移
$$
\Psi^{(2)}(\mathbf k)\propto\frac{i\mathbf k}{k^2}\!\int\! d^3k_1\,\Big[1-\frac{(\mathbf k_1\!\cdot\!\mathbf k_2)^2}{k_1^2k_2^2}\Big]\delta(\mathbf k_1)\delta(\mathbf k_2)\,\delta_D(\mathbf k-\mathbf k_1-\mathbf k_2)
$$
展开后有 $LL$、$L\eta$、$\eta\eta$ 三项。

**$\eta\eta\to$ 低 $k$（UV backreaction）。** 两个 octave 模式 $\mathbf k_1\approx-\mathbf k_2$ 相加落到粗带低 $k$ 处。核在 $k\to0$ 时的展开（我用 sympy 核对过）
$$
1-\frac{(\mathbf k_1\cdot\mathbf k_2)^2}{k_1^2k_2^2}=\frac{k^2}{k_1^2}\,(1-\mu^2)+O(k^3),\qquad \mu=\hat{\mathbf k}\cdot\hat{\mathbf k}_1,
$$
所以 $\Psi^{(2)}_{\eta\eta}(\mathbf k)=O(k)$，密度 $\delta^{(2)}=O(k^2)$，功率 $P\propto k^4$——这就是 Lagrangian 版本的质量与动量守恒（Peebles 的 $k^4$ 尾）。结论：**correction $\varepsilon_\ell$ 在 $k\ll k_{\rm Ny,\ell}$ 处被 $k^2$ 压低，且其领头项是局域算子**（$\propto k^2\Psi^{(1)}$，即 EFTofLSS 的 $c_s^2\nabla\nabla^2\phi$ 型反项，系数依赖被截掉的 octave）。随机部分（EFT 的 stochastic term）压得更狠。因此 correction head 可以做得很小、很局域、近似确定性；只有 $k\sim k_{\rm Ny,\ell}$ 附近它才是 $O(1)$ 并需要完整模型。

我用 nested 2LPT 的合成数据（`phase0_octaves.py --selftest --selftest-dealias`）数值验证了这一点：对谱型 $R$，$P_\varepsilon$ 在 $0.1$–$0.5\,k_{\rm Ny,c}$ 的对数斜率为 $2.2$，散度功率为 $4.15$；Haar 的 $\varepsilon$ 在低 $k$ 反而是平的（$\propto k^{4+n-2}$，来自线性阶的 $\cos$ 滤波失配），幅度高出谱型 $R$ 好几个数量级。**但有一个重要的 caveat**：$k^2$ 压低是对"理想的连续 coarse-graining"成立的。一个真实的 LR N-body 还有离散性误差——在合成实验里，只要把粗一级的 2LPT 直接在粗网格上算（二次项 aliasing，模拟 LR 的离散化），低 $k$ 的 $\varepsilon$ 就多出一个近似白噪声的底（斜率 $\approx0$），它不满足 $k^2$ 压低，虽然幅度很小。真实 LR 的对应物是 PM 网格 aliasing、力软化和时间步误差。所以 correction head 要学的是"$k^2$ 型的物理反作用 + 一个小的离散性白噪底"，后者的大小只能由 Phase 0 (a) 在真实 64/128/256/512 上测出来。

**$L\eta$ 交叉项（tidal coupling）。** $\mathbf k=\mathbf k_L+\mathbf k_\eta$ 基本落在 octave 里，是细节对粗场的领头依赖。在 Lagrangian 坐标下把长模在一个 patch 中心 $\mathbf q_0$ 展开：
$$
\Psi_L(\mathbf q)\approx\Psi_L(\mathbf q_0)+D_{ij}(\mathbf q_0)(q-q_0)_j+\tfrac12\partial_k D_{ij}(q-q_0)_j(q-q_0)_k+\cdots,\qquad D_{ij}:=\partial_i\Psi_{L,j}.
$$
常数项是纯平移，对内部动力学没有影响（Galilean/平移不变性）；$D_{ij}$ 的迹是局部密度（$\delta_L=-\mathrm{tr}D$ 线性阶），无迹部分是潮汐场。Separate-universe 与 anisotropic separate-universe 模拟（Schmidt et al. 2018, arXiv:1803.03274；Stücker et al. 2021, arXiv:2003.06427；Masaki, Nishimichi & Takada 2020）已经系统地测过小尺度结构对这两者的 response。翻译成建模语言：
$$
p(d_\ell\mid\Psi_\ell)\ \approx\ p\big(d_\ell\mid \text{粗场在该 patch 的低阶 jet}:\ D_{ij},\ \partial_kD_{ij},\ \ldots\big),
$$
即**conditional 在 Lagrangian 坐标下是局域的，并且只通过粗场的导数进入**。这直接决定了网络输入应当是什么（§6）。

> 注意：Paper V 试过 11 通道的 Lagrangian 特征堆叠，对 subhalo deficit 没帮助。这里的动机不一样：在 progressive + 权重共享的设定下，用 $D_{ij}$ 而不是 $\Psi$ 本身不是"多给点特征"，而是让算子在不同层级上**量纲一致**的必要条件（§6）——不共享权重时这个必要性不存在，所以 Paper V 的结果不能直接外推到这里。

---

## 6. 尺度条件化与自相似：一个算子跑所有层级的前提

**无量纲化。** 位移有物理单位，不随层级缩放；但如果输入是 $\Psi_\ell/h_\ell$（以格距为单位的位移），粗场的大尺度 bulk flow 会随 $\ell$ 增大而"发散"（跨越越来越多的细格，无界），而细节以格距为单位的幅度 $\sigma_{\Psi,\ell}/h_\ell\propto k_\ell^{(n+3)/2}$ 是有界的、由该级的 $\sigma(h_\ell,z)$ 控制的量（$n>-3$ 时随层级缓慢增大，正是"细一级在其网格尺度上更非线性"）。唯一干净的做法：
- 输入用 **deformation tensor $D_{ij}=\partial_i\Psi_{\ell,j}$**（无量纲，平移不变，在已分辨尺度上幅度与层级无关；壳层交叉对应本征值 $=-1$）以及速度梯度；
- 输出用 **以 $h_\ell$ 为单位的细节位移** $d_\ell/h_\ell$；
- 用一个标量 style 向量 $s_\ell$ 描述"这一级 octave 的非线性状态"。

**Scale-free 宇宙里这是严格的。** $P\propto k^n$、$\Omega_m=1$ 时，无量纲统计量只依赖 $h_\ell/r_{\rm NL}(a)$，$r_{\rm NL}\propto a^{2/(n+3)}$。于是"往下走一级"等价于"时间往前推"（细一级的网格相对 $r_{\rm NL}$ 更小，等价于粗一级在 $r_{\rm NL}$ 长到两倍时的状态）：$(\ell+1,a)\equiv(\ell,a')$，$a'/a=2^{(n+3)/2}$（$n=-2$ 时 $a'=\sqrt2\,a$）。这给出三样东西：一个对权重共享的**严格检验**（在 $(\ell,a)$ 训练的 2× 算子必须在 $(\ell+1,a')$ 上原样工作）、免费的数据增广、以及把模型的"分辨率误差"变成可量化对象（Joyce, Garrison & Eisenstein 2021, MNRAS 504, 3550；Garrison et al., arXiv:2004.07256 用自相似性量化 N-body 分辨率的方法可以直接搬过来用在 AR 链上）。

**ΛCDM 里变成一个 universality 假设。** 条件化变量取 $s_\ell=\big(\sigma(h_\ell,z),\ n_{\rm eff}(k_\ell),\ f(z)\ [\text{或 }\Omega_m(z)]\big)$，前两者类比 halo mass function 的 $\nu$-universality，$f$ 负责速度–位移关系。假设是：**2× 算子在这些无量纲变量下是普适的，对 $(\ell,z,\text{cosmology})$ 的依赖只通过 $s_\ell$**。这个假设可检验（Phase 0 (d)），而且比 Paper III/V 分别对 $z$、对 cosmology 条件化更经济——一旦成立，一个模型天然覆盖所有层级、红移和宇宙学。

---

## 7. 为什么"2× 逐级 + 局域网络"在数学上是对的选择

把 Mallat 组的三个结果翻译到我们的设定：

1. **WSGM**：逐 wavelet 尺度做 conditional 生成，每级的 score 条件数与分辨率无关 → 每级所需的 flow/diffusion 步数固定，总代价随粒子数线性增长。对我们：每级只生成"相对当前输入的最高 octave"，任务在所有层级上是同一个任务——这恰好对冲了神经网络的 spectral bias（永远只学"下一个 octave"，而不是一次学 3 个 octave）。
2. **CSLC**：$\varphi^4$ 与弱透镜图整体高度非高斯，但 $p(\text{details}\mid\text{coarse})$ 是强 log-concave 的。**对 Lagrangian 位移场这一点很可能更好**：§4–5 说明单流区的 conditional 在微扰论下近高斯，非高斯性集中在多流的 Lagrangian halo patch。这是一个可以在训练前直接检验的假设（Phase 0 (c)）。
3. **WC-RG**：每级的条件能量在 wavelet 基下是**局域**相互作用 → 一个 receptive field 以格点为单位固定的 CNN，在第 $\ell$ 级覆盖的物理尺度是 $r\,h_\ell$，随层级收缩，这正是 RG 的图像：长程物理已经编码在粗场里，细节只需要看局部。§5 的 jet 展开给了这个局域性一个宇宙学的理由（tidal response）。

三条合起来，"权重共享的局域 2× 算子 + 尺度条件化 + 每级固定步数的 conditional flow"不是工程上的取巧，而是这类分布的正确参数化。

---

## 8. FNO、wavelet、CNN：我的判断

**FNO。** 它的卖点是 discretization invariance / zero-shot SR，而 2025–26 的三篇文章（§1）已经说明这在原理上做不到：非线性层的 aliasing 把伪高频折回已学习频带，模型无法 extrapolate 到没见过的频率信息。对我们的问题这甚至不是缺陷而是无关：新 octave 是新信息，只能由 $\xi_\ell$ 注入。另外，模数固定为 $K$ 的 FNO 作用在**固定的物理尺度**上，要在层级间共享它必须改成 patch 上的谱卷积（patch 物理尺寸逐级减半），此时它就变成"带周期边界的 patch 内全局卷积"——而 §5 说 conditional 是局域的，全局卷积没有理由更好，还引入 patch 边界伪影。FNO 唯一合适的位置是 **correction head**：它近似线性、谱对角（$\propto k^2$ 反项），一个小 FNO 或者干脆一个可学习的各向同性传递函数就够了。

**Wavelet。** 用作**脚手架**而不是黑箱：它定义 $(R,P,W)$、细节子空间、以及每级注入的自由度。Multiwavelet neural operator（Gupta, Xiao & Bogdan 2021, arXiv:2109.13459）之类的"wavelet 算子"把多分辨率藏在网络里，恰恰放弃了我们最想显式控制的东西。Wavelet Flow（Yu, Derpanis & Brubaker 2020）和 Cosmo3DFlow 是"在 wavelet 系数空间做生成"的先例，但前者是 2D 图像、后者是 Eulerian 密度。

**CNN / 局域 attention（+ 解析的线性部分）。** 线性部分（$R,P,W,G_\ell^{1/2}$）在 Fourier 空间是对角的，应当**精确实现而不是学习**；非线性残差 $f_\theta$ 用局域网络，输入 $D_{ij}$，输出 $d_\ell/h_\ell$，对 cubic group（48 元）做增广或做等变，对 $\Psi$ 的常数平移严格不变（因为只看梯度）。你现有的 DiT/flow-matching 栈可以直接改造：把 token 化的输入换成 $(D_{ij},\ \nabla\mathbf v,\ G_\ell^{1/2}\xi_\ell)$，时间条件 + $s_\ell$ 走 AdaLN。

一个值得试的 flow-matching 变体：**source 分布不用 white noise，而用线性理论 octave** $G_\ell^{1/2}\xi_\ell$，target 是真实细节。这样 OT 路径在高红移几乎为零长度，只在 halo 内部才长——flow 只需要学"非线性演化"这一段，而不是从噪声开始重新发明线性理论。

---

## 9. Autoregressive 的误差累积与一致性

**Markov 性。** 对 (B) 型链，粗场是细场的确定性函数，跨尺度 Markov 性严格成立。对 (A) 型链，$\Psi_{\ell+1}=\mathcal S_{\ell+1}[\delta_\ell+\eta_\ell]$ 原则上决定了 $\delta_\ell+\eta_\ell$（IC 可重建），所以 $p(\Psi_{\ell+2}\mid\Psi_{\ell+1},\Psi_\ell)=p(\Psi_{\ell+2}\mid\Psi_{\ell+1})$ 也成立。链在原理上是自洽的；问题全在近似误差的累积。

**三个抑制手段。**
1. correction head 让每一级都能修上一级在其 Nyquist 附近的错误——链是"自愈"的，这是 progressive 相对一次性 8× 的一个真实优势，不只是代价。
2. 训练时让后续层级看到模型自己的输出（rollout / pushforward trick，Brandstetter, Worrall & Welling 2022），同时第一级用真实 LR、后续级混合真实模拟与模型输出，直面 §2.3 的分布漂移。
3. Ground truth 用 **N-body 收敛序列**：同一 seed 跑 $64^3,128^3,256^3,512^3$，AR 链第 $\ell$ 级的输出应在统计上匹配第 $\ell$ 级的真实模拟（$P(k)$、HMF、单个 halo 的性质随粒子数的收敛曲线）。换句话说，AR 链要复现 N-body 自己的"分辨率收敛行为"，这比只跟最终 HR 比严格得多，也更有物理意义。

---

## 10. Halo 内部：Lagrangian 表示的难点，以及 progressive 给的出路

Paper II/V 的 subhalo deficit 说明 Lagrangian 生成在多流区最吃力：halo 内部的 Lagrangian 映射是折叠的，粒子的最终位置对 $\eta$ 敏感。Progressive 框架给出两个思路：
- **按层级对应 halo 质量。** 第 $\ell$ 级 octave 对应的质量尺度 $\sim m_\ell\times N_{\min}$，即第 $\ell$ 级新增的结构主要是这一质量段的 halo/subhalo。逐级生成让每一级只负责一个质量段，而不是一次生成三个数量级——subhalo 作为小 Lagrangian patch 的内部相干性可能因此更容易学。
- **深层级换参考系。** 当一个 Lagrangian patch 整个落在某个 halo 内部时，conditional 变成"给定粗场的 halo 轮廓，采样一个 virialized 系统的相空间"。由于模型只输出相对 $P\Psi_\ell$ 的残差（§3 的结构），它自然工作在"相对 halo 质心"的局部参考系里；可以进一步对这类 patch 用不同的 $s_\ell$（比如加入局部 $\det(\partial\mathbf x/\partial\mathbf q)$ 的符号/流数）让同一网络切换模式。

---

## 11. 具体怎么开始

### Phase 0：不训练任何模型的"理论数值实验"（1–2 周；脚本 `phase0_octaves.py` 已写好并自测）

现有数据（同一 seed 的 $64^3/128^3/256^3/512^3$，$100\,h^{-1}$Mpc）直接可用；若有多个红移的 snapshot 更好。脚本对每个 2× transition 输出 (a)(b)(c) 三组量和一张总览图，先用 `--offset 0`/`0.5` 各跑一次，nestedness check 给出 $\sim10^{-6}$ 的那个就是 IC 生成器的网格约定：

- **(a) 选 $R$。** 对锐截断球、锐截断立方体、Haar 三种 $R$ 测 $\varepsilon_\ell$ 的功率谱与它同粗场的互相关；验证低 $k$ 的 $k^2$ 压低、Nyquist 附近的 $O(1)$，以及离散性白噪底的大小（§5 caveat）；同时测 Wiener 型 $T_\ell(k)$（§3.2）。这一步决定 correction head 要多大。
- **(b) 回归 vs 生成的比例。** 细节 $d_\ell$ 与线性理论 octave $\Psi^{(1)}[\eta_\ell]$ 的互相关系数 $r(k)$ 随 $z,\ell$ 的变化：$r\approx1$ 的部分是确定性的，$1-r^2$ 就是必须由生成模型负责的 stochasticity（Paper IV 里的 $1-r^2$ 指标直接复用）。
- **(c) 条件高斯性。** 按粗场局部 $D_{ij}$ 的不变量（$\mathrm{tr}$、无迹部分的幅度、单/多流）分箱，测 $d_\ell$ 的方差与峰度；再测把 conditioning 从"局部 jet"扩大到"更宽邻域"时方差被解释掉多少——这同时检验 CSLC 型假设与 §5 的局域性，直接给出网络 receptive field 该多大。
- **(d) Universality。** 在 $\sigma(h_\ell,z)$ 匹配的 $(\ell,z)$ 对之间比较 $d_\ell/h_\ell$ 的统计量；若有 scale-free 模拟（或自己跑一个 $n=-2$ 的），做 §6 的严格自相似检验。

这四项本身就能写成一节"理论动机"，而且每一项都会改变后面的架构决定。

### Phase 1：确定性 progressive emulator（Paper IV 的逐级版）
输入 $(D_{ij},\nabla\mathbf v,\ G_\ell^{1/2}\xi_\ell$ 取真实 octave$)$，输出 $(\hat\varepsilon_\ell,\hat d_\ell)$，固定的 $P,W$ 合成；loss 用现成的 Lagrangian Charbonnier + lag2eul CIC；三对层级 $64\!\to\!128\!\to\!256\!\to\!512$ 共享权重、条件化 $s_\ell$。检验：链式 $64\to512$ vs 直接 HR；自相似检验；对比"不共享权重"的 ablation。

### Phase 2：生成版
把真实 $\xi_\ell$ 换成采样，用 conditional flow matching（source = 线性 octave，§8），每级步数固定；网络复用 Phase 1 的权重加时间输入。评估用 Paper V 的整套 validation（$P(k)$、HMF、SHMF、mean occupation number）外加 §9 的收敛序列比较。

### Phase 3：rollout 训练 + "AI zoom-in"
一切都是 patch-局域的，所以可以**只对选定的 Lagrangian patch 继续往下细化**（halo 的 Lagrangian 区域），这就是 MUSIC 式 nested-grid zoom-in 的 AI 版本：大盒子里对感兴趣的 halo 做任意深度的局部加密。这是 I–V 都做不到、而 progressive 2× 天然能做的应用，值得作为整条线的 headline。

---

## 12. 风险与开放问题

- 第一级输入是真实 LR、后续级是模型输出，两者分布不同（§2.3）；correction head 加混合训练能缓解，但要测。
- 力软化随层级缩放的规则被模型隐式学到；换模拟代码/软化方案会破坏 universality。
- 各向同性谱窗的 $R$ 非局域，halo 附近的 Gibbs 振铃可能污染 correction；需要和 lifting 型的局域 biorthogonal wavelet 对比。
- 多流区的 conditional 是否真的"可局域、可 log-concave"完全是经验问题，Phase 0 (c) 之前不要押注。
- 自由度计数是每分量 $7N_\ell^3$；如果用带 update 步的 wavelet，粗系数本身也会被更新，要确认 $\ker R$ 的维度与注入的 noise 维度一致，否则模型会有多余或不足的随机性。

---

## 参考文献（本次核对过的）

- Li, Ni, Croft, Di Matteo, Bird, Feng 2021, PNAS — AI-assisted SR I, arXiv:2010.06608
- Ni et al. 2021 — AI-assisted SR II, arXiv:2105.01016
- Zhang, Lachance, Ni, Li, Croft, Di Matteo, Bird, Feng 2024, MNRAS 528, 281 — AI-assisted SR III, arXiv:2305.12222
- Zhang, Lachance, Dasgupta, Croft, Di Matteo, Ni, Bird, Li 2024, OJAp — AI-assisted SR IV, arXiv:2408.09051
- Guth, Coste, De Bortoli, Mallat 2022, NeurIPS — Wavelet Score-Based Generative Modeling, arXiv:2208.05003
- Marchand, Ozawa, Biroli, Mallat 2022 — Wavelet Conditional Renormalization Group, arXiv:2207.04941
- Guth, Lempereur, Bruna, Mallat 2023, ICML — Conditionally Strongly Log-Concave Generative Models, arXiv:2306.00181
- Tian et al. 2024, NeurIPS — Visual Autoregressive Modeling (next-scale prediction), arXiv:2404.02905
- Hong & Belkadi 2025 — Multi-scale AR models are Laplacian/discrete/latent diffusion in disguise, arXiv:2510.02826
- Zhao, Molodyk, Xue, Chen 2026 — Laplacian Multi-scale Flow Matching, arXiv:2602.19461
- Schanz, List, Hahn 2023, OJAp — Stochastic SR of cosmological simulations with diffusion, arXiv:2310.06929
- Islam et al. 2026 — Cosmo3DFlow (wavelet flow matching, IC reconstruction), arXiv:2602.10172
- Colagrande, Caillon, Feillet, Allauzen 2026 — Limits of Resolution Equivariance in FNO, arXiv:2606.00677
- "The False Promise of Zero-Shot Super-Resolution in Machine-Learned Operators", arXiv:2510.06646
- Subedi & Tewari 2026 — Is Zero-Shot Super-Resolution Possible in Operator Learning?, arXiv:2606.00296
- Gupta, Xiao, Bogdan 2021, NeurIPS — Multiwavelet-based Operator Learning, arXiv:2109.13459
- Jamieson et al. 2023, ApJ 952, 145 — Field-level neural network emulator (Lagrangian), arXiv:2206.04594；Jamieson et al. 2024, arXiv:2408.07699
- Nishimichi, Bernardeau, Taruya 2016, PLB — Response function of LSS to small-scale inhomogeneities, arXiv:1411.2970
- Schmidt, Cabass, Jasche, Lavaux 2018 — N-body simulations with a large-scale tidal field, arXiv:1803.03274
- Stücker et al. 2021, MNRAS 503, 1473 — Anisotropic separate universe simulations (TreePM), arXiv:2003.06427
- Joyce, Garrison, Eisenstein 2021, MNRAS 504, 3550 — "Good and proper": self-similarity with proper force softening；Garrison et al. — Quantifying resolution using self-similarity, arXiv:2004.07256
- Yu, Derpanis, Brubaker 2020, NeurIPS — Wavelet Flow；Sweldens 1996/1998 — the lifting scheme；Hahn & Abel 2011, MNRAS — MUSIC multi-scale initial conditions；Brandstetter, Worrall, Welling 2022, ICLR — pushforward trick for autoregressive neural PDE solvers
