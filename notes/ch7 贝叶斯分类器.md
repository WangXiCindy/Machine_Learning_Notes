---
Typora-root-url: ../assets/MLpics
---


# 贝叶斯分类器

## 贝叶斯决策论

- 贝叶斯决策论（Bayesian decsion theory）

  - 概率框架下实施决策的基本方法
  - 对分类任务：在所有相关概率都已知的理想情形下，其考虑如何基于这些概率和误判损失来选择最优的类别标记。

- 例子：

  - 假设有N种可能的类别标记，即$\mathcal{Y}=\{ c_1,c_2,...,c_N \}$

  - 在样本x上的**条件风险（conditional risk）**

    - 将样本x分类为$c_i$所产生的期望损失（expected loss）
    - $R(c_i \mid x)= \sum_{j=1}^N \lambda_{ij}P(c_i \mid x)$
    - $\lambda_{ij}$是将一个真实标记为$c_j$的样本误分类为$c_i$所产生的损失
    - $P(c_i \mid x)$后验概率

  - 寻找一个特定准则$h: \mathcal{X} \mapsto \mathcal{Y}$以最小化总体风险$R(h)=\mathbb{E}_{x}[R(h(x) \mid x)]$

    - **贝叶斯判定准则（Bayes decision rule）**最小化总体风险，只需在每个样本上选择那个能使条件风险$R（c \mid x)$最小的类别标记
      - 获得**贝叶斯最优分类器（Bayes optimal classifier）**即$h^*(x)=arg_{c \in \mathcal{Y}}min \ R(c \mid x)$
      - 对应的总体风险$R(h^*)$成为**贝叶斯风险（Bayes risk）**
      - $1-R(h^*)$反映了分类器所能达到的最好性能，（通过ML所能产生的模型精度的理论上限）

  - 若目标是最小化分类错误率，则误判损失$\lambda_{ij}$可以写为

    $\lambda_{i j}= \{\begin{array}{ll}0, & \text { if } i=j \\ 1, & \text { otherwise }\end{array} .$

    - 此时条件风险$R(c \mid x)=1-P(c \mid x)$
    - 于是，最小化分类错误率的贝叶斯最优分类器为$h^*(x)=arg_{c \in \mathcal{Y}}max\ P(c \mid x)$
    - 对每个样本x，选择能使后验概率$P(c \mid x)$最大的类别标记（只是从概率角度分析，实际上很多方法无需计算后验概率）
      - **判别式模型（discriminative models）**：给定x，可通过直接建模$P(c \mid x)$来预测c
        - 例如：决策树，BP神经网络，SVM
      - **生成式模型（generative models）**：先对联合概率分布P(x,c)建模，然后再由此获得$P(c \mid x)$
        - 必须考虑$P(c \mid x)=\frac{P(x,c)}{P(x)}$
        - 基于贝叶斯定理得到公式（7.8）$P(c \mid x)=\frac{P(c) P(x \mid c)}{P(x)}$
        - $P(c)$是**类先验概率**：表达了样本空间中各类样本所占的比例
        - $P(x \mid c)$样本x相对于类标记c的**类条件概率（class-conditional probability）**或**似然（likelihood）**
          - 由于设计x所有属性的联合概率，直接根据样本出现的频率估计艰难，因为有可能很多样本取值在训练集中根本没有出现
        - $P(x)$用于归一化的**证据（evidence）**因子，对给定样本x，证据因子P(x)与类标记无关，因此估计$P(c \mid x)$的问题就转化为如何基于训练数据D来估计先验$P(c)$和似然$P(x \mid c)$

## 极大似然估计

- **极大似然估计（Maximum Likelihood Estimation，简称MLE）**

- 估计类条件概率的一种常用策略
  - 假定其具有某种确定的概率分布形式
  - 基于训练样本对概率分布的参数进行估计
  - 记关于类别c的类条件概率为$P(x \mid c)$，假设$P(x \mid c)$具有确定的形式并且被参数向量$\theta_c$唯一确定
  - 任务：利用训练集D估计参数$\theta_c$，将$P(x \mid c)$记为$P(x \mid \theta_c)$
  
- **参数估计（parameter estimation）**：概率模型的训练过程
  - 频率主义学派（Frequentist）
    - 参数虽然未知，但却是客观存在的固定值
    - 通过优化似然函数等准则来确定参数值
    - 本文介绍本主义学派的MLE，根据数据采样来估计概率分布参数的经典方法
  - 贝叶斯学派（Bayesian）
    - 参数是未观察到的随机变量，其本身也可有分布
    - 假定参数服从一个先验分布（简单来说，后验就是知果求因，先验就是由历史求因）
  
- 推导
  - 令$D_c$表示训练集D中第c类样本组成的集合，假设这些样本是独立同分布的，则参数$\theta_c$对于数据集$D_c$的似然是（式7.9）$P(D_{c} \mid \theta_{c})=\prod_{x \in D_{c}} P(x \mid \theta_{c})$

  - 对$\theta_c$进行极大似然估计，寻找能最大化似然$P(D_c \mid \theta_c)$的参数值$\hat{\theta}_c$

  - 式7.9的连乘操作容易造成下溢，通常使用**对数似然（log-likelihood）**

    $$\begin{aligned}LL(\theta_c)&=log P(D_c \mid \theta_c)\\ &= \sum_{x \in D_c} log P(x \vert \theta_c)\end{aligned}$$

  - 此时参数$\theta_c$的极大似然估计为$\hat{\theta_c}=arg_{\theta_c}max LL(\theta_c)$

- 通过极大似然法得到的

  ​	$$\begin{aligned} \hat{\theta_c}&=arg_{\theta_c}max LL(\theta_c) \\ &=arg_{\theta_c}min -LL(\theta_c) \\ &= arg_{\theta_c}min - \sum_{x \in D_c} log P(x \vert \theta_c)\end{aligned}$$

  - 此时假设概率密度函数$p(x\ \mid c) \sim \mathcal{N}(\mu_{c}, \sigma_{c}^{2})$，(通过正态分布公式）其等价于假设

    $P(x \mid \theta_c)=P(x \mid \mu_c, \sigma_c^2)=\frac{1}{\sqrt{(2 \pi)^d \vert \sum_c \vert}}exp(-\frac{1}{2}(x-\mu_c)^T \sum_c^{-1} (x-\mu_c))$

    - 其中$\sum_c=\sigma_c^2$为对称正定协方差矩阵，$\vert \sum_c \vert$为其行列式
    - 假设$\vert D_c \vert=n$

    $$\begin{aligned}\hat{\mu}_{c}, \hat{\mathbf{\Sigma}}_{c} &=\underset{\mu_{c}, \mathbf{\Sigma}_{c}}{\arg \min }-\sum_{x \in D_{c}} \log  [\frac{1}{\sqrt{(2 \pi)^{d} |\Sigma_{c} |}} \exp  (-\frac{1}{2} (x-\mu_{c} )^{\mathrm{T}} \Sigma_{c}^{-1} (x-\mu_{c} ) ) ] \\ &=\underset{\mu_{c}, \Sigma_{c}}{\arg \min }-\sum_{x \in D_{c}} [-\frac{d}{2} \log (2 \pi)-\frac{1}{2} \log  |\Sigma_{c} |-\frac{1}{2} (x-\mu_{c} )^{\mathrm{T}} \Sigma_{c}^{-1} (x-\mu_{c} ) ] \\ &=\underset{\mu_{c}, \Sigma_{c}}{\arg \min } \sum_{x \in D_{c}} [\frac{d}{2} \log (2 \pi)+\frac{1}{2} \log  |\Sigma_{c} |+\frac{1}{2} (x-\mu_{c} )^{\mathrm{T}} \Sigma_{c}^{-1} (x-\mu_{c} ) ] \\&=\underset{\boldsymbol{\mu}_{c}, \Sigma_{c}}{\arg \min } \sum_{x \in D_{c}} [\frac{1}{2} \log  |\Sigma_{c} |+\frac{1}{2} (x-\mu_{c} )^{\mathrm{T}} \Sigma_{c}^{-1} (x-\mu_{c} ) ] \\ &= \underset{\mu_{c}, \mathbf{\Sigma}_{c}}{\arg \min } \sum_{i=1}^n [\frac{1}{2} \log  |\Sigma_{c} |+\frac{1}{2} (x-\mu_{c} )^{\mathrm{T}} \Sigma_{c}^{-1} (x-\mu_{c} ) ] \\ &=\underset{\mu_{c}, \mathbf{\Sigma}_{c}}{\arg \min }  \frac{n}{2} \log  |\Sigma_{c} |+\sum_{i=1}^n\frac{1}{2} (x-\mu_{c} )^{\mathrm{T}} \Sigma_{c}^{-1} (x-\mu_{c} )  \end{aligned}$$

  - 根据公式$x^TAx=tr(Axx^T),\bar{x}=\frac{1}{n}\sum_{i=1}^n x_i$

    $\hat{\mu}_{c}, \hat{\mathbf{\Sigma}}_{c}=\underset{\mu_{c}, \mathbf{\Sigma}_{c}}{\arg \min } \frac{n}{2} \log |\Sigma_{c}|+\frac{1}{2} \operatorname{tr}[\Sigma_{c}^{-1} \sum_{i=1}^{n}(x_{i}-\overline{x})(x_{i}-\overline{x})^{\mathrm{T}}]+\frac{n}{2}(\mu_{c}-\overline{x})^{\mathrm{T}} \Sigma_{c}^{-1}(\boldsymbol{\mu}_{c}-\overline{x})$

    - 由于$\sum_c，\sum_c^{-1}$均为正定矩阵，所以$\mu_c - \overline{x} \not =0$，最后一项为正定二次型，因此此时其大小仅仅与$\mu_c - \overline{x} $有关，当且仅当$\mu_c - \overline{x}  =0$，取到最小值0

    - 所以$\hat{\mu_c}=\bar{x}=\frac{1}{n}\sum_{i=1}^n x_i$，代入方程

    - $ \hat{\mathbf{\Sigma}}_{c}=\underset{\mu_{c}, \mathbf{\Sigma}_{c}}{\arg \min } \frac{n}{2} \log |\Sigma_{c}|+\frac{1}{2} \operatorname{tr}[\Sigma_{c}^{-1} \sum_{i=1}^{n}(x_{i}-\overline{x})(x_{i}-\overline{x})^{\mathrm{T}}]$

    - 引入引理，设B为p阶正定矩阵，n>0 为实数，在对所有p阶正定矩阵有（当且仅当$\sum=\frac{1}{n}B$时等号成立）

      $\frac{n}{2} \log |\Sigma|+\frac{1}{2} tr[\Sigma^{-1} B] \geq \frac{n}{2} \log |B|+\frac{p n}{2}(1-\log n)$

    - 所以当且仅当$\Sigma_{c}=\frac{1}{n} \sum_{i=1}^{n}(x_{i}-\overline{x})(x_{i}-\overline{x})^{\mathrm{T}}$取到最小值

  - 正态分布均值就是样本均值

  - 方差就是$(x-\hat{\mu}_c)(x-\hat{\mu}_c)^T$的均值

- 这种参数化方法虽然能使类条件概率估计变得相对简单，但估计结果的准确性严重依赖于所假设的概率分布形式是否符合潜在的真实数据分布

## 朴素贝叶斯分类器

- 基于贝叶斯公式7.8来估计后验概率$P(c \mid x)$的主要困难

  - 类条件概率$P(x \mid c)$是所有属性上的联合概率，难以从有限的训练样本直接估计而得

- **朴素贝叶斯分类器（naive Bayes classifeier）**

  - 采用**属性条件独立性假设（attribute conditional independence assumption）**：对已知类别，假设所有属性相互独立地对分类结果发生影响
  - 基于该理论，式7.8可以重写为$P(c \mid x)=\frac{P(c) P(x \mid c)}{P(x)}=\frac{P(c)}{P(x)} \prod_{i=1}^dP(x_i \mid c)$
    - 其中d为属性数目
    - $x_i$为x在第i个属性上的取值
  - 基于贝叶斯判定准则得到朴素贝叶斯分类器的表达式：$h_{nb}(x)=arg_{c \in \mathcal{Y}}max P(c) \prod_{i=1}^d P(x_i \mid c)$

- 训练过程

  - 基于训练集D估计类先验概率P(c)，并为每个属性估计条件概率$P(x_i \mid c)$
    - 令$D_c$表示训练集D中第c类样本组成的集合，若有充足的独立同分布样本，则可容易估计出类先验概率（式7.16）$P(c)=\frac{\vert D_c \vert}{\vert D \vert}$
    - 离散属性
      - 令$D_{c, x_i}$表示$D_c$中在第i个属性上取值为$x_i$的样本组成的集合
      - 则条件概率（式7.17）$P(x_i \mid c)=\frac{\vert D_{c, x_i} \vert}{\vert D_c \vert}$
    - 连续属性
      - 概率密度函数
      - 假定$p(x_{i} \mid c) \sim \mathcal{N}(\mu_{c, i}, \sigma_{c, i}^{2})$
      - $\mu_{c, i}, \sigma_{c, i}^{2}$分别是第c类样本在第i个属性上取值的均值和方差
      - 式7.18：$p(x_{i} \mid c)=\frac{1}{\sqrt{2 \pi} \sigma_{c, i}} \exp (-\frac{(x_{i}-\mu_{c, i})^{2}}{2 \sigma_{c, i}^{2}})$

- 例子：使用西瓜数据集3.0训练一个朴素贝叶斯分类器

  - 如下图

    <img src=" /T95.png" style="zoom:50%;" />

  - 先估计类先验概率P(c)，显然有

    $P($ 好瓜 $=$ 是 $)=\frac{8}{17} \approx 0.471 .$

    $P($ 好瓜 $=$ 否 $)=\frac{9}{17} \approx 0.529 .$

  - 然后为每个属性估计条件概率$P(x_i \mid c)$，如

    $P($ 青绿 $=$ 是 $)=P($色泽$=$青绿$ \mid $好瓜$=$是$)=\frac{3}{8}=0.375$

    $$\begin{aligned} p_{\text {密度: } 0.697 \mid \text { 是 }} &=p(\text { 密度 }=0.697 \mid \text { 好瓜 }=\text { 是 }) \\ &=\frac{1}{\sqrt{2 \pi} \cdot 0.129} \exp (-\frac{(0.697-0.574)^{2}}{2 \cdot 0.129^{2}}) \approx 1.959 \end{aligned}$$

  - 所以

    <img src=" /T96.png" style="zoom:50%;" />

  - 由于$0.038>6.80 \times 10^{-5}$，所以判别为“好瓜”

  - 在估计概率值时通常要进行**平滑（smoothing）**

    - 解决问题：如果出现$P(清脆 \mid 是 )=P($敲声$=$清脆$ \mid $好瓜$=$是$)=\frac{0}{8}=0$，会直接得出好瓜=否

    - 常用**拉普拉斯修正（Laplacian correction）**

    - 令N表示训练集D中可能的类别数，$N_i$表示第i个属性可能的取值数，则7.16和7.17分别修正为

      - 7.19：$\hat{P}(c)=\frac{\vert D_c \vert +1}{\vert D \vert + N}$
      - 7.20：$\hat{P}(x_i \mid c)=\frac{\vert D_{c,x_i} \vert +1}{\vert D_c \vert + N_i}$

    - 因此类先验概率为

      $\hat{P}($ 好瓜 $=$ 是 $)=\frac{8+1}{17+2} \approx 0.474 .$

      $\hat{P}($ 好瓜 $=$ 否 $)=\frac{9+1}{17+2} \approx 0.526 .$

    - 所以：$P(清脆 \mid 是 )=P($敲声$=$清脆$ \mid $好瓜$=$是$)=\frac{0+1}{8+3}=0.091$

    - 避免了由于训练集样本不充分导致的概率估值为0的问题，并且在训练集变大时，修正过程所引入的先验的影响也会逐渐变得可忽略，使得估值渐趋向于实际概率值

- 一些其他的小知识

  - **懒惰学习（lazy learning）**：先不进行任何训练，待收到预测请求时再根据当前数据集进行概率估值
  - **增量学习**：在现有估值基础上，仅对新增样本的属性值所涉及的概率估值进行计数修正

## 半朴素贝叶斯分类器

- 为了降低贝叶斯公式中估计后验概率的困难，朴素贝叶斯分类器采用了属性条件独立性假设，但现实中往往很难成立

- **半朴素贝叶斯分类器（semi-naive Bayes classifiers）**：对属性条件独立性假设进行一定程度的放松。

- 基本想法

  - 适当考虑一部分属性间的相互依赖信息
  - 既不需进行完全联合概率计算
  - 又不会彻底忽略了比较强的属性依赖关系

- **独依赖估计（One-Dependent Estimator，简称ODE）**

  - 半朴素贝叶斯分类器最常用的一种策略

  - 独依赖：假设每个属性在类别之外最多仅依赖于一个其他属性

    - $P(c \mid x) \propto P(c) \prod_{i=1}^{d} P(x_{i} \mid c, p a_{i})$
      - $pa_i$为属性$x_i$所依赖的属性，称为$x_i$的父属性
      - 对每个属性$x_i$，若其父属性$pa_i$已知，则可以采用类似7.20的方法来估计概率值$P(x_i \mid c,pa_i)$

  - 如何确定每个属性的父属性

    - **SPODE（Super-Parent ODE)**假设所有属性都依赖于同一个属性，称为**超父（superparent）**，然后通过交叉验证等模型选择方法来确定超父属性

    - **TAN（Tree Augmented naive Bayes）**：在**最大带权生成树（maximum weighted spanning tree）**的基础上

      - 步骤

        - 计算任意两个属性之间的**条件互信息（conditional mutual information）**

          $I(x_i,x_j \mid y)=\sum_{x_i,x_j;c \in \mathcal{Y}}P(x_i,x_j \mid c)log \frac{P(x_i,x_j \mid c)}{P(x_i \mid c)P(x_j \mid c)}$

          - 刻画了属性$x_i,x_j$在已知类别的情况下的相关性

        - 以属性为结点构建完全图，任意两个结点之间的边的权重设为$I(x_i,x_j \mid y)$

        - 构建此完全图的最大带权生成树，挑选根变量，将边置为有向

        - 加入类别结点y，增加从y到每个属性的有向边

      - TAN仅保留了强相关属性之间的依赖性

    - 如图：

      <img src=" /T97.png" style="zoom:50%;" />

    - **AODE（Averaged One-Dependent Estimator）**

      - 基于集成学习体制

      - 尝试将每个属性作为超父来构建SPODE，然后将有足够训练数据支撑的SPODE集成起来作为最终结果

        $P(c \mid x) \propto \sum_{i=1,\mid D_{x_{i}}\mid \ge m}P(c,x_i) \prod_{j=1}^{d} P(x_{j} \mid c,x_{i})$

        - 其中$D_{x_i}$是在第i个属性上取值为$x_i$的样本的集合，m'为阈值函数

      - $\hat{P}(c,x_i)=\frac{\vert D_{c,x_i} \vert +1}{\vert D \vert + N_i}$

      - $\hat{P}(x_j \mid c,x_i)=\frac{\vert D_{c,x_i,x_j} \vert +1}{\vert D_{c,x_i} \vert + N_j}$

      - $N_i$是第i个属性可能的取值数

      - $D_{c,x_i}$是类别为c且在第i个属性上，取值为$x_i$的样本集合

      - $D_{c,x_i}$是类别为c且在第i和j个属性上，取值为$x_i$和$x_j$的样本集合

    - 例如，对瓜瓜数据3.0

      <img src=" /T98.png" style="zoom:50%;" />

    - 训练过程：同朴素贝叶斯

      - 计数，在训练数据集上对符合条件的样本进行计数的过程
      - 无需模型选择
      - 优点：
        - 通过预计算节省预测时间
        - 采取懒惰学习方式在预测时再进行计数
        - 易于实现增量学习

  - 如果我们考虑属性间的高阶依赖来进一步提升泛化性能，如将$pa_i$替换为包含k个属性的集合$pa_i$，ODE---->kDE

    - 随着k的增加，估计概率$P(x_{i} \mid c, p a_{i})$所需要的训练样本数量将以指数级增加
    - 训练数据非常充分，泛化性能才有可能提升

## 贝叶斯网

- **贝叶斯网（Bayesian network）**，别名**信念网（belief network）**

  - 借助**有向无环图（Directed Acyclic Graph，简称DAG）**来刻画属性之间的依赖关系

  - 使用**条件概率表（Conditional Probability Table，简称CPT）**来描述联合概率分布

  - 一个贝叶斯网B由结构G和参数$\Theta$两部分构成，则$B=<G,\Theta>$

    - G是一个有向无环图，其每个结点对应于一个属性
    - 若两个属性有直接依赖关系，则它们由一条边连接起来
    - 假设属性$x_i$在G中的父结点集为$\pi_i$，则参数$\Theta$包含了每个属性的条件概率表$\theta_{x_i \mid \pi_i}=P_B(x_i \mid \pi_i)$

  - 例如

    <img src=" /T99.png" style="zoom:50%;" />

    - 色泽直接依赖于好瓜和甜度
    - 根蒂直接依赖于甜度
    - 从条件概率表能得到根蒂对甜度量化依赖关系，如$P(根蒂=硬挺 \mid 甜度=高)=0.1$

- 结构

  - 有效表达了属性间的条件独立性

  - 给定父结点集，贝叶斯网假设每个属性与它的非后裔属性独立

  - $B=<G,\Theta>$将属性$x_1,x_2,...,x_d$的联合概率分布定义为（式7.26）

    $P_B(x_1,x_2,...,x_d)=\prod_{i=1}^d P_B(x_i \mid \pi_i)=\prod_{i=1}^d \theta_{x_i \mid \pi_i}$

  - 图7.2的联合概率分布定义为$P(x_1,x_2,x_3,x_4,x_5)=P(x_1)P(x_2)P(x_3 \mid x_1)P(x_4 \mid x_1,x_2) P(x_5 \mid x_2)$

    - 注意，$x_3,x_4$在给定$x_1$取值时独立，简记为$x_{3} \perp x_{4} \mid x_{1}$（$x_4,x_5$对$x_2$同理）

  - 贝叶斯网三种变量之间典型依赖关系

    <img src=" /T100.png" style="zoom:50%;" />

    - **同父（common parent）**：给定父结点的取值，则$x_3,x_4$条件独立

    - **顺序**：给定x的取值，y与z条件独立

    - **V型结构（V-structure）**，别名**冲撞**结构：给定子结点$x_4$的取值，$x_1,x_2$必不独立

      - **边际独立性（marginal independence）**：若$x_4$的取值未知，则$x_1,x_2$却相互独立

      - 验证

        $$\begin{aligned}P(x_1,x_2)&=\sum_{x_4}P(x_1,x_2,x_4) \\ &=\sum_{x_4}P(x_4 \mid x_1,x_2)P(x_1)P(x_2) \\ &=P(x_1)P(x_2)\end{aligned}$$

      - 记为<img src=" /T101.png" style="zoom:50%;" />

  - 为了分析有向图变量间的条件独立性，可使用**有向分离（Desparation）**

    - 找出有向图中的所有V型结构，在V型结构的两个父结点之间加上一条无向边
    - 将所有有向边改为无向边
    - 产生的无向图称为**道德图（moral graph）**，令父结点相连的过程称为**道德化（moralization）**

  - 假定道德图中有变量x，y和变量集合z

    - x和y被z有向分离：从道德图中将变量集合z去除后，x和y分属两个连通分支，$x \perp y \mid z$   成立

      <img src=" /T102.png" style="zoom:50%;" />

- 学习

  - 若网络和属性间的依赖关系已知，则贝叶斯网学习过程相对简单，只需对训练样本计数，估计出每个结点的条件概率表即可

    - 现实生活中并不知晓网络结构

  - 任务：根据训练数据集来找出结构最“恰当的”贝叶斯网

  - 方法：评分搜索

    - **评分函数（score function）**：评估贝叶斯网与训练数据的契合程度
    - 基于这个评分函数来寻找结构最优的贝叶斯网
    - 评分函数引入了归纳偏好

  - 常用评分函数

    - 基于信息论准则
    - 将学习问题看作一个数据压缩任务
    - 学习目标：找到一个能以最短编码长度描述训练数据的模型
    - 编码长度
      - 描述模型自身所需的字节长度
      - 使用该模型描述数据所需的字节长度

  - **最小描述长度（Minimal Description Length，简称MDL）**

    - 模型就是一个贝叶斯网
    - 每个贝叶斯网描述了一个在训练数据上的概率分布
    - 经常出现的样本有更短的编码
    - 选择综合编码长度（包括描述网络和编码数据）最短的贝叶斯网

  - 公式推导

    - 给定训练集D，贝叶斯网$B=<G,\Theta>$在D上的评分函数可以写为（式7.28）

      $s(B \mid D)=f(\theta) \vert B \vert -LL(B \mid D)$

      - $\vert B \vert$是贝叶斯网的参数个数
      - $f(\theta)$表示描述每个参数$\theta$所需的字节数
      - $LL(B \mid D)=\sum_{i=1}^m log P_B(x_i)$是贝叶斯网B的对数似然（式7.29）
      - 公式第一项是计算编码贝叶斯网B所需的字节数
      - 第二项是计算B所对应的概率分布$P_B$需要多少字节来描述D

    - 学习任务转化为一个优化任务：寻找B使评分函数最小

      - 若$f(\theta)=1$，得到**AIC（Akaike Information Criterion）**评分函数

        $AIC(B \mid D)=\vert B \vert -LL(B \mid D)$

      - 若$f(\theta)=\frac{1}{2}log_m$，得到**BIC（Bayesian Information Criterion）**评分函数

        $BIC(B \mid D)=\frac{log \  m}{2}\vert B \vert -LL(B \mid D)$

      - 若$f(\theta)=0$，即不计算对网络进行编码的长度，则评分函数退化为负对数似然，学习任务退化为极大似然估计

    - 若贝叶斯网B的网络结构G固定，则评分函数$s(B \mid D)$的第一项为常数，此时最小化$s(B \mid D)$等价于对参数$\Theta$的极大似然估计

    - 由式7.29和7.26可知，参数$\theta_{x_i \mid \pi_i}=\hat{P}_D(x_i \mid \pi_i)$

      - $\hat{P}_D(·)$是D上的经验分布
      - 为了最小化评分函数，只需要对网络结构进行搜索
      - 候选结构的最优参数可直接在训练集上计算得到

  - 从所有可能的网络结构空间搜索最优贝叶斯网结构是NP难问题

    - 贪心法：从某个网络结构出发，每次调整一条边（增删或改方向），直到评分函数值不再降低
    - 通过给网络结构施加约束来削减搜索空间

- 推断

  - 概念

    - **查询（query）**：通过一些属性变量的观测值来推测其他属性变量的取值
    - **推断（inference）**：通过已知变量观测值来推测待查询变量的过程
    - **证据（evidence）**：已知变量观测值

  - 直接根据贝叶斯网定义的联合概率分布来精确计算后验概率（但NP难，在网络结点较多，连接稠密时，难以进行精确推断）

  - 近似推断：降低精度要求，在有限时间内求得金丝街

  - **吉布斯采样（Gibbs sampling）**：随机采样方法，进行贝叶斯网的近似推断

    - 令$Q=\{ Q_1, Q_2,...,Q_n \}$表示待查询变量，$E=\{ E_1,E_2,...,E_k \}$为证据变量，已知其取值为$e=\{ e_1, e_2, ...,e_k \}$

    - 目标：计算后验概率$P(Q=q \mid E=e)$

    - $q=\{ q_1,q_2,...,q_n \}$是待查询变量的一组取值

    - 例如

      - $Q=\{ 好瓜，甜度 \}$
      - $E=\{色泽，敲声，根蒂 \}$，其取值为$e=\{ 青绿，浊响，蜷缩\}$
      - 目标值$q=\{是，高\}$（这是好瓜且甜度高的概率）

    - 吉布斯采样算法先随机产生一个与证据$E=e$一致的样本$q^0$作为初始点，然后每步从当前样本出发产生下一个样本

    - 在第t次采样中，算法先假设$q^t=q^{t-1}$

    - 对非证据变量逐个进行采样改变其取值

    - 采样概率根据贝叶斯网V和其他变量的当前取值计算获得

    - 假定经过T次采样得到的与q一致的样本共有$n_q$个，则可近似估算出后验概率（式7.33）

      $P(Q=q \mid E=E) \simeq \frac{n_{q}}{T}$

  - 吉布斯采样是在贝叶斯网所有变量的联合状态空间与证据$E=e$一致的子空间中进行**随机漫步（random walk）**

    - 每一步仅依赖于前一步的状态，是一个**马尔可夫链（Markov chain）**
    - 在一定条件下，无论从什么初始状态开始，马尔可夫链第t步的状态分布在$t \rightarrow \infty$时必收敛于一个**平稳分布（stationary distribution）**
    - 对于吉布斯采样来说，这个分布恰好是$P(Q \mid E=e)$
    - 在T很大时，吉布斯采样相当于根据$P(Q \mid E=e)$采样
    - 保证了式7.33收敛于$P(Q=q \mid E=E)$

    <img src=" /T103.png" style="zoom:50%;" />

- 注意

  - 由于马尔可夫链需要很长时间才能趋于平稳分布，所以吉布斯采样算法收敛速度较慢
  - 若贝叶斯网中存在极端概率0 or 1，不能保证马尔可夫链存在平稳分布，此时吉布斯采样会给出错误的估计结果

## EM算法

- 前面我们一直假设训练样本的所有属性变量的值都已被观测到，即训练样本是完整的，但现实生活中往往会遇到不完整的训练样本

  - 比如由于西瓜的根蒂已经脱落，无法看出是“蜷缩”还是“硬挺”，即根蒂属性值位置

- **隐变量（latent variable）**：未观测变量的学名

  - 令X表示已观测变量集

  - Z表示隐变量集

  - $\Theta$表示模型参数

  - 若欲对$\Theta$做极大似然估计，则应最大化对数似然

    $LL(\Theta \mid X,Z)=ln P(X,Z \mid \Theta)$

  - 由于Z无法直接求解，所以可通过对Z求期望，来最大化已观测数据的对数**边际似然（marginal likelihood）**

    $LL(\Theta \mid X)=ln P(X \mid \Theta)=ln \sum_z P(X,Z \mid \Theta)$

- **EM（Expectation-Maximization)**

  - 常用的估计参数隐变量的利器

  - 是一种迭代式的方法

  - 基本想法：

    - 若参数$\Theta$已知，则可根据训练数据推断出最优隐变量Z的值（E步）
    - 若Z的值已知，则可方便地对参数$\Theta$做极大似然估计（M步）

  - 步骤

    - 以$\Theta^0$为起点，对式7.35.可迭代执行以下步骤直到收敛
    - 基于$\Theta^t$推断隐变量Z的期望，记为$Z^t$
    - 基于已观测变量X和$Z^t$对参数$\Theta$做极大似然估计，记为$\Theta^{t+1}$

  - 如果不取Z的期望，而是基于$\Theta^t$计算隐变量Z的概率分布$P(Z \mid X,\Theta^t)$，则EM算法的两个步骤是

    - E步（Expectation）：以当前参数$\Theta^t$推断隐变量分布$P(Z \mid X,\Theta^t)$，并计算对数似然$LL(\Theta \mid X,Z)$关于Z的期望

      $Q(\Theta \mid \Theta^t)=\mathbb{E}_{Z \mid X,\Theta^t} LL(\Theta \mid X,Z)$

    - M步（Maximization）：寻找参数最大化期望似然，即

      $\Theta^{t+1}=arg_{\Theta}max \ Q(\Theta \mid \Theta^t)$

    - 也就是说，先做E步，然后最大化M步，然后将心得到的参数值重新用于E步，直到收敛到局部最优解

  - EM是一种非梯度优化，可以克服由于求和项数随着隐变量数目以指数级上升的梯度计算的麻烦
