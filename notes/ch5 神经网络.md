---
Typora-root-url: ..
---
# 神经网络

## 神经元模型

- **神经网络（neural networks）**是具有适应性的简单单元组成的广泛并行互连的网络，它的组织能够模拟生物神经系统对真实世界物体所作出的交互反应

- 成分

  - **神经元（neuron）**：最基本的，相当于定义中的“简单单元”

- **M-P神经元模型**：

  ​	<img src="/assets/MLpics/T64.png" style="zoom:50%;" />

  - 神经元接收到来自n个其他神经元传递过来的输入信号
  - 这些输入信号通过带权重的连接进行传递
  - 神经元接收到的总输入值将与神经元的阈值进行比较
  - 通过**激活函数（activation function）**处理以产生神经元的输出

- 激活函数

  - 理想：阶跃函数

    - 1对应神经元兴奋
    - 0对应神经元抑制
    - 但具有不连续，不光滑等不佳性质

  - 实际：Sigmoid（别名挤压函数（squashing function））

    <img src="/assets/MLpics/T65.png" style="zoom:50%;" />

## 感知机与多层网络

- **感知机（Perceptron）**

  - 两层神经元组成

    <img src="/assets/MLpics/T66.png" style="zoom:50%;" />

  - 输入层接收外界输入信号传递给输出层

  - 输出层是M-P神经元，亦称**阈值逻辑单元（threshold logic unit）**

    - 感知机只有输出层神经元进行激活函数处理
    - 只有一层**功能神经元（functional neuron）**

  - $y=f(\sum_i w_i x_i-\theta)=f(w^Tx-\theta)$（假设f为阶跃函数）

  - 可进行逻辑与、或、非运算

    - 均属于**线性可分（linearly separable）**问题

    - 与运算（x1^x2）

      - 令$w_1=w_2=1,\theta=2,y=f(1*x_1+1*x_2-2)$
      - 仅$x_1=x_2=1,y=1$

    - 或运算（x1 v x2）

      - 令$w_1=w_2=1,\theta=0.5,y=f(1*x_1+1*x_2-0.5)$
      - 仅$x_1=1 \  or \  x_2=1,y=1$

    - 非运算<img src="/assets/MLpics/T67.png" style="zoom:50%;" />

      - 令$w_1=-0.6,w_2=0,\theta=-0.5,y=f(-0.6*x_1+0.5)$
      - 仅$x_1=1,y=0$
      - $x_1=0,y=1$

    - 若两类模式是线性可分的，即存在一个线性超平面能将它们分开

      - n维空间的超平面方程为$w_1x_1+...w_nx_n=w^Tx+b=0$
      - $w^Tx-\theta$可以看作是n维空间中的一个超平面，将n维空间划分为
        - $w^Tx-\theta \ge 0$ 样本模型输出值为1
        - $w^Tx-\theta <0$ 样本模型输出值为0

      <img src="/assets/Mlpics/T68.png" style="zoom:50%;" />

    - 感知机的学习过程一定会收敛（converge）而求得适当的权向量w

    - 否则学习过程会发生振荡（fluctuation），w难以稳定下来，如上图中的异或问题

  - 学习

    - 给定训练数据集，权重$w_i(i=1,2,...,n)$以及阈值$\theta$可以通过学习得到
      - 阈值作为一个固定输入为-1.0的**哑结点（dummy node）**所对应的权重$w_{n+1}$
      - 权值和阈值的学习统一为权重的学习
    - 损失函数
      - 易得$(y-\hat{y})(w^T x-\theta) \ge 0$恒成立
      - $L(w,\theta)=\sum_{x \in M}(y-\hat{y})(w^T x-\theta)$
        - M为误分类样本集合
        - 该公式连续可导
          - 没有误分类点，损失函数值为0
          - 误分类点越少，则误分类点离超平面越近，损失函数值越小
      - 求参数
        - 极小化$min_{w,\theta}L(w,\theta)=min_{w,\theta}\sum_{x_i \in M}(\hat{y_i}-{y_i})(w^T x_i-\theta)$
        - 把阈值看为哑结点，简化$w^Tx_i-\theta=w^Tx_i$
          - $min_{w}L(w,\theta)=min_{w}\sum_{x_i \in M}(\hat{y_i}-y_i)w^T x_i$
        - 假设M固定，求梯度
          - $\Delta_wL(w)=\sum_{x_i \in M}(\hat{y_i}-y_i)x_i$
      - 对训练样例（x，y），若当前感知机的输出为$\hat{y}$
        - $w_i \leftarrow w_i+\Delta w_i$
        - 参数更新公式：$\Delta w_i=\eta(y-\hat{y})x_i$
    - **学习率（learning rate）**：$\eta \in (0,1)$
    - 权重调整
      - 若$\hat{y}=y$，感知机不发生变化
      - 根据错误的程度进行权重调整

  - 多层功能神经元

    - 解决非线性可分问题

      <img src="/assets/MLpics/T69.png" style="zoom:50%;" />

    - **隐（含）层（hidden layer）**：也是拥有激活函数的功能神经元

    - **多层前馈神经网络（multi-layer feedforward neural networks）**

      - A别称两层网络（本书称为单隐层网络）

        <img src="/assets/MLpics/T70.png" style="zoom:50%;" />

        - 输入层：接收外界输入
        - 隐藏层，输出层（功能神经元）：对信号进行加工
        - 输出层：输出结果

      - 学习过程就是根据训练数据来调整神经元之间的**“连接权”（connection weight）**以及每个功能神经元的阈值

## 误差逆传播算法

- 误差逆传播（error BackPropagation，简称BP）

  - 训练多层网络的学习算法
  - 虽然可用于其他类型神经网络训练，但通常指的是多层前馈

- 算法思想

  - 给定训练集$D=\{ (x_1,y_1),...(x_m,y_m)\},x_i \in R^d y_i \in R^l$

    - 输入示例由d个属性描述，输出l维实值向量

  - 下图有d个输入神经元，l个输出，q个隐藏层神经元

    <img src="/assets/MLpics/T71.png" style="zoom:50%;" />

    - $\theta_j$：输出层第j个神经元的阈值
    - $\gamma_h$：隐藏层第h个神经元的阈值
    - $v_{ih}$：输入层第i个神经元与隐藏层第h个神经元之间的连接权
    - $w_{hj}$：隐藏层第h个神经元与输出层第j个神经元之间的连接权
    - $b_h$：隐藏层第h个神经元的输出
    - 使用sigmoid
    - 第h个隐藏层神经元的输入和第j个输出神经元的输入见上图

  - 对训练样例$(x_k,y_k)$，假定$\hat{y_k}=(\hat{y_1}^k,\hat{y_2}^k...\hat{y_l}^k)$

    - $\hat{y_j}^k=f(\beta_j-\theta_j)$

    - 则网络在样例上的均方误差为

      $E_k=\frac{1}{2}\sum_{j=1}^l(\hat{y_j}^k-y_j^k)^2$

  - 上图的网络中有 (d+l+1)q+l个参数需要确定

    - 输入层到隐藏层：dq
    - 隐藏层：q
    - 隐藏层到输出层：ql
    - 输出层：l

  - BP算法公式推导

    - 迭代学习

    - 基于梯度下降，以目标的负梯度方向对参数进行调整

    - 在每一轮中采用广义感知机对参数进行更新估计：$v \leftarrow v+\Delta v$

    - 以上图中$w_{hj}$为例

      - $\Delta w_{hj}=- \eta \frac{\delta E_k}{\delta w_{hj}}$

      - 先影响到第j个输出层神经元的输入值$\beta_j$，再影响到其输出值$\hat{y_j}^k$

        - $\frac{\delta E_k}{\delta w_{hj}}=\frac{\delta E_k}{\delta \hat{y_j}^k}·\frac{\delta  \hat{y_j}^k}{\delta \beta_j}·\frac{\delta \beta_j}{\delta w_{hj}}$

        - $\frac{\delta \beta_j}{\delta w_{hj}}=b_h$

        - sigmoid函数的性质$f'(x)=f(x)(1-f(x))$

          $$\begin{aligned}g_j &=-\frac{\delta E_k}{\delta \hat{y_j}^k}·\frac{\delta  \hat{y_j}^k}{\delta \beta_j} \\ &=-(\hat{y_j}^k-y_j^k)f'(\beta_j-\theta_j) \\ &=\hat{y_j}^k(1-\hat{y_j}^k)(y_j^k-\hat{y_j}^k) \end{aligned}$$

        - $\Delta w_{hj}=- \eta g_j b_h$

        - 同理可得（不懂推导的童鞋可以参考南瓜书）

          - $\Delta \theta_j=-\eta g_j$

            - 个人的简单理解：和上方$w_{hj}$同公式，但$\theta$和输入值无关，所以不需要$b_h$

          - $\Delta v_{ih}=\eta e_h x_i$

            - 非常类似于$w_{hj}$的推导，只是这次为求输入层和隐藏层的连接权

          - $\Delta \gamma_h=-\eta e_h$

            - 同$\theta$

          - 注意$e_h$

            $$\begin{aligned} e_h &=-\frac{\delta E_k}{\delta b_h}·\frac{\delta b_h}{\delta \alpha_h} \\ &=-\sum_{j=1}^l \frac{\delta E_k}{\delta \beta_j}·\frac{\delta \beta_j}{\delta b_h} f'(\alpha_h-\gamma_h) \\ &=\sum_{j=1}^l w_{hj}g_j f'(\alpha_h-\gamma_h) \\ &=b_h(1-b_h)\sum_{j=1}^l w_{hj}g_j\end{aligned}$$

          - 注意$\eta$

            - 学习率控制者每一轮迭代中的更新步长，
              - 太大容易振荡
              - 太小收敛速度会过慢
            - 精细调节
              - $w_{hj},\theta$和$v_{ih},\gamma_h$使用的$\eta$可以不同

  - BP算法的工作流程

    - BP算法的目标：最小化训练集D上的累积误差$E=\frac{1}{m}\sum_{k=1}^m E_k$
    - 标准BP算法：每次仅针对一个训练样例更新连接权和阈值
      - 也就是说下图的更新规则是基于单个$E_k$推导得到
      - 参数更新频繁
      - 对不同样例可能会出现抵消现象
      - 为了达到同样的累积误差最小点，需要更多次的迭代
      - 对应**随机梯度下降（stochastic gradient descent，简称SGD）**

    <img src="/assets/MLpics/T72.png" style="zoom:50%;" />

    结果示例：

    <img src="/assets/MLpics/T73.png" style="zoom:50%;" />

  - **累积误差逆传播（accumulated error backpropagation）**

    - 基于累积误差最小化的更新规则
    - 在读取整个训练集D一遍后才对参数进行更新
    - 更新频率更低
    - 但下降到一定程度之后，进一步下降会非常缓慢，这时标准BP会更快获得较好的解（在D非常大时更明显）
    - 对应**标准梯度下降**

- **试错法（trial-by-error）**

  - **只需要一个包含足够多神经元的隐藏层，多层前馈神经网络就能以任意精度逼近任意复杂度的连续函数**
  - 如何设置隐藏层神经元的个数

- 过拟合缓解

  - **早停（early stopping）**
    - 将数据氛围训练集和验证集
    - 训练集：计算梯度、更新连接权和阈值
    - 验证集：估计误差
    - 训练集误差降低但验证集误差升高，则停止训练，返回连接权和阈值
  - **正则化（regularization）**
    - 在误差目标函数中增加一个用于描述网络复杂度的部分
    - 例如连接权与阈值的平方和
    - $E=\lambda \frac{1}{m}\sum_{k=1}^mE_k+(1-\lambda)\sum_i w_i^2$
      - $\lambda \in (0,1)$
        - 使用交叉验证进行估计
        - 用于对经验误差与网络复杂度这两项进行折中

## 全局最小与局部极小

- E表示训练集上的误差，训练过程为在参数空间中，寻找一组最优参数使得E最小

  - 若存在$w^*,\theta^*,\epsilon>0$使得

    $\forall (w;\theta) \in \{ (w;\theta) \vert \ \vert\vert (w;\theta)-(w^*;\theta^*) \vert\vert \le \epsilon \}$

- **局部极小（local minimum）**
  - 对上式都有$E(w;\theta) \ge E(w^*;\theta^*)$成立，$(w^*;\theta^*)$为局部最小解
  - 其邻域点的误差函数值均不小于该点的函数值
  - 对应的$E(w^*;\theta^*)$为局部极小值
  - 梯度为0，误差函数值小于邻点
  - 可能存在多个
- **全局最小（global minimum）**
  - 对参数空间的任意$(w;\theta)$都有$E(w;\theta) \ge E(w^*;\theta^*),(w^*;\theta^*)$为全局最小解
  - 所有点的误差函数值均不小于该点的函数值
  - 对应的$E(w^*;\theta^*)$为全局最小值
  - 只能存在一个
  - 一定是局部极小

<img src="/assets/MLpics/T74.png" style="zoom:50%;" />

- 基于梯度的搜索
  - 最广泛的参数寻优
  - 思路
    - 从初始解出发，迭代寻找最优参数
    - 每次迭代中
      - 先计算误差函数在当前点的梯度
      - 根据梯度决定搜索方向
        - 负梯度方向是函数值下降最快的方向
      - 若当前点的梯度为0，则达到局部最小，参数的迭代在这里停止
    - 找到全局最小（大多启发式，理论上缺乏保障）
      - 只有一个局部极小=全局最小
      - 多个局部极小，则试图“跳出”局部极小
        - 方法1：以多组不同参数初始化多个神经网络，取误差最小的解最为最终参数（需要陷入不同的局部极小）
        - 方法2：**模拟退火（simulated annealing）**，每一步都以一定的概率接受比当前解更差的结果
          - 在每步迭代的过程中，接受“次优解”的概率要随着时间的推移而逐渐降低
        - 方法3：随机梯度下降，在计算梯度时加入了随机因素，陷入局部极小点时，它计算出的梯度仍可能不为0
        - 方法4：**遗传算法（genetic algorithms）**

## 其他神经网络

- **RBF（Radial Basis Function，径向基函数）**网络

  - 单隐层前馈神经网络

  - 使用径向基函数作为隐层神经元激活函数

  - 输出层：对隐层神经元的线性组合

  - 假定输入d维向量x

    $\phi(x)=\sum_{i=1}^q w_i \rho(x,c_i)$

    - q为隐层神经元个数

    - $c_i,w_i$分别为第i个隐层神经元对应的中心和权重

    - $\rho(x,c_i)$为径向基函数（某种沿径向对称的标量函数）

      - 样本x到数据中心$c_i$之间欧式距离的单调函数

      - 常用高斯径向基函数

        $\rho(x,c_i)=e^{-\beta_i \vert\vert x-c_i \vert\vert^2}$

  - 步骤

    - 确定神经元中心$c_i$
      - 方式：随机采样，聚类
    - 利用BP算法等来确定参数$w_i,\beta_i$

- **ART（Adaptive Resonance Theory，自适应谐振理论）网络**

  - **竞争型学习（competitve learning）**
    - 无监督学习策略
    - **胜者通吃（winner-take-all）**原则
      - 网络的输出神经元相互竞争
      - 每个时刻仅有一个竞争获胜的神经元被激活，其它的被抑制
    - ART是其中的重要代表
  - 组成
    - 比较层：接收输入样本，并将其传递给识别层神经元
    - 识别层：每个神经元对应一个模式类，神经元数目可在训练过程中动态增长增加新的模式类
      - 模式类可以认为是某个类比的子类
    - 识别阈值
    - 重置模块
  - 过程
    - 识别层神经元收到比较层的输入信号
    - 神经元之间相互竞争产生获胜神经元
      - 计算输入向量与每个识别层神经元所对应的 模式类的 代表向量之间的距离
      - 距离最小者胜
    - 获胜神经元向其他识别层神经元发送信号，抑制激活
    - 输入向量与获胜神经元所对应的代表向量之间的相似度大于识别阈值
      - 当前输入样本被归为该代表向量所属类别
      - 网络连接权更新
      - 以后再接收到相似输入样本时该模式类会计算出更大的相似度，从而该获胜神经元更大可能获胜
    - 相似度不大于识别阈值
      - 重置模块会在识别层增设一个新的神经元
      - 代表向量为当前输入向量
  - 识别阈值对性能影响
    - 识别阈值较高：输入样本将会分成比较多的、精细的模式类
    - 识别阈值较低：产生比较少、比较粗略的模式类
  - **可塑性-稳定性窘境（stability-plasticity dilemma）**
    - 可塑性：神经网络要有学习新知识的能力
    - 稳定性：神经网络在学习新知识时要保持对旧知识的记忆
    - ART的优点：可以进行**增量学习（incremental learning）**或**在线学习（online learning）**

- **SOM（Self-Organizing （Feature） Map，自组织（特征）映射）**网络

  - 属于竞争学习型的无监督神经网络

  - 功能

    - 将高维输入数据映射到低维空间（通常为二维）

    - 同时保持输入数据在高维空间的拓扑结构

    - 也就是将高维空间中相似的样本点映射到网络输出层中的邻近神经元，从而保持拓扑结构

      <img src="/assets/MLpics/T75.png" style="zoom:50%;" />

  - 过程

    - 接收到一个训练样本
    - 每个输出层神经元会计算该样本与自身携带的权向量之间的距离
    - **最佳匹配单元（best matching unit）**：距离最近的神经元成为竞争获胜者
    - 最佳匹配单元及其邻近神经元的权向量将被调整，使得这些权向量与当前输入样本的距离缩小
    - 不断迭代直至收敛

- **级联相关（Cascade-Correlation）**网络

  - 属于**结构自适应（亦称 构造性（constructive））**神经网络
    - 不同于一般的网络结构事先固定，将网络结构也当作学习目标之一

  - 训练

    ​	<img src="/assets/MLpics/T76.png" style="zoom:50%;" />

    - 级联：建立层次连接的层级结构
      - 开始时，网络只有输入层和输出层，处于最小拓扑结构
      - 新的隐层神经元逐渐加入
        - 输入端连接权值是冻结固定的
    - 相关：最大化新神经元的输出与网络误差之间的相关性来训练参数

  - 优劣势

    - 无需设置网络层数，隐层神经元网络，训练速度较快
    - 在数据较小时容易陷入过拟合

- Elman网络

  - 属于**递归神经网络（recurrent neural networks）**
    - 允许网络中出现环形结构
    - 可以让一些神经元的输出反馈回来作为输入信号
    - 因此网络在t时刻的输出状态不仅与t时刻的输入状态和t-1时刻的网络状态有关，能够处理与时间有关的动态变化
  - 与前馈区别：隐层神经元的输出被反馈回来，与下一时刻输入层神经元提供的信号一起作为隐层神经元在下一时刻的输入
  - 隐层神经元：Sigmoid激活函数
  - 网络训练：BP算法

- Boltzmann机（亦称“平衡态（equilibrium）或平稳分布（stationary distribution））

  - **基于能量的模型（energy-based model）**

    - 为网络状态定义一个能量，其最小化时网络达到理想状态，网络训练就是在最小化这个能量函数

  - 神经元

    - 分为两层
      - 显层：表示数据的输入与输出
      - 隐层：数据的内在表达
    - 类型：布尔型的
      - 只能取0和1两种状态
        - 0：抑制
        - 1：激活

  - 能量定义

    - 令向量$s \in \{ 0,1  \}^n$表示n个神经元的状态

    - $w_{ij}$表示神经元i与j之间的连接权

    - $\theta_i$表示神经元i的阈值

    - 状态s对应的Boltzmann机能量的定义为

      $E(s)=-\sum_{i=1}^{n-1}\sum_{j=i+1}^n w_{ij} s_i s_j - \sum_{i=1}^n \theta_i s_i$

    - 推导

      - 能量值越大，当前状态越不稳定（物理学上），能量值最小系统处于稳定态

      - Boltzmann机本质上引入了隐变量的无向图模型，能量为

        $E_{graph}=E_{edges}+E_{nodes}$

      - 分别代表图、边和结点的能量

      - 边能量：两连接结点的值及其权重的乘积$E_{edge_{ij}}=-w_{ij}s_is_j$

      - 结点能量：结点的值及其阈值的乘积$E_{nodes_i}=-\theta_is_i$

      - $E_{edges}=\sum_{i=1}^{n-1}\sum_{j=i+1}^n E_{edge_{ij}}=-\sum_{i=1}^{n-1}\sum_{j=i+1}^n w_{ij}s_is_j$

      - $E_{nodes}=\sum_{p=1}^n E_{node_i}=-\sum_{p=1}^n \theta_i s_i$

      - $E_{graph}=-\sum_{i=1}^{n-1}\sum_{j=i+1}^n w_{ij}s_is_j-\sum_{p=1}^n \theta_i s_i$

  - Boltzmann分布

    - 若网络中的神经元以任意不依赖于输入值的顺序进行更新，最终网络将达到Boltzmann分布

    - 状态s出现的概率将仅由其能量与所有可能状态向量的能量确定

    - 推导

      - 注意相关知识在14.2节

      - 无向图网络联合概率分布

        - k为无向图中的极大团个数
        - $c_i$为极大团的节点集合
        - $x_{c_i}$为极大团所对应的节点变量
        - $\Phi_i$为势函数
        - Z为规范化因子
        - $P(s)=\frac{1}{Z}\prod_{i=1}^k \Phi_i(s_{c_i})$

      -  Boltzmann机的极大团只有一个（为全连接网络），结点集合为$c=\{ s_1,s_2,...,s_n \}$

        - 联合概率分布为

          $P(s)=\frac{1}{Z} \Phi(s_c)$

      - 势函数$\Phi(s_c)$一般定义为指数型函数，所以其一般形式为

        $\Phi(s_c)=e^{-E(s_c)}$

      - 其中$s_c=s$

        - 状态集合T中某个状态s的概率定义：状态s的联合概率分布与所有可能的状态的联合概率分布的比值

        - 则状态s下的联合概率分布为

          $P(s)=\frac{1}{Z} e^{-E(s)}$

        - $P(s)=\frac{e^{-E(s)}}{\sum_{t \in T} e^{-E(t)}}$

      - $P(s)=\frac{e^{-E(s)}}{\sum_t e^{-E(t)}}$            （公式5.21）

  - 训练过程

    - 将每个样本视为一个状态向量，使其出现的概率尽可能大

  - 分类

    ​		<img src="/assets/MLpics/T77.png" style="zoom:50%;" />

    - 标准的Boltzmann机：是一个全连接图，训练网络的复杂度很高，难以解决现实任务
    - **受限Boltzmann机（Restricted Boltzmann Machine，简称RBM）**：仅保留显层与隐层之间的连接，从而将其结构由完全图转为二部图
      - 使用**对比散度（Contrastive Divergence，简称CD）**算法
      - 假定网络中有d个显层神经元和q个隐层神经元
      - 令v和h分别表示显层与隐层的状态向量，则由于同一层内不存在连接
      - $P(v \vert h) = \prod_{i=1}^d P(v_i \vert h)$
      - $P(h \vert v) = \prod_{j=1}^q P(h_j \vert v)$
      - CD算法对每个训练样本v
        - 先根据上式计算出隐层神经元状态的概率分布
        - 根据这个概率分布采样得到h
        - 类似地根据上上式从h产生$v'$，再从$v'$产生$h'$
        - 连接权的更新公式：$\Delta w=\eta(vh^T-v'h'^T)$（推导见本章附录1）

## 深度学习

- 诞生原因（复杂模型的缺陷和当下为何可用）

  - 计算能力大幅提高，缓解训练低效性
  - 训练数据的大幅增加可降低过拟合风险

- 提高容量

  - 容量越大，能完成更复杂的学习任务
  - 增加隐层神经元的数目
  - 增加隐层的数目（更有效）
    - 增加拥有激活函数的神经元数目
    - 增加激活函数嵌套的层数
    - 多隐层（三个以上隐层）神经网络，难以用经典算法（如标准BP）进行训练，因为会**发散（diverge）**而不能收敛到稳定态

- **无监督逐层训练（unsupervised layer-wise training）**

  - 多隐层网络训练的有效手段

  - 思想

    - 预训练+微调：可视为将大量参数分组，对每组先找到局部较优，然后基于这些结果进行全局寻优

      - 目的
        - 利用模型大量参数所提供的自由度时间
        - 节省训练开销
      - **预训练（pre-training）**
        - 每次训练一层隐结点
        - 训练时将上一层隐结点的输出作为输入
        - 本层隐结点的输出作为下一层隐结点的输入
      - **微调（finetuning）**

    - **权共享（weight sharing）**

      - 让一组神经元使用相同的连接权

      - 用于CNN

        - **特征映射（feature map）**：一个由多个神经元构成的平面

        - **RELU**：将sigmoid激活函数替换为修正线性函数

          $\begin{equation}
          f(x)=\left\{\begin{array}{ll}
          0, & \text { if } x<0 \\
          x, & \text { otherwise }
          \end{array}\right.
          \end{equation}$

        - **采样层（亦称汇合层）（pooling）**：基于局部相关性原理进行亚采样，减少数据量，保留有用信息

        - 可用BP训练

- 理解

  - **特征学习（feature learing）或表示学习（representation learning）**
  - 对输入信号进行逐层加工，把初始输入转化为和输出目标联系更密切的表示，让最后一层输出映射成为可能
    - 逐渐将初始的低层特征转化为高层特征表示后
    - 用简单模型即可完成复杂的分类等学习任务

- **特征工程（feature engineering）**

  - 用于描述样本的特征
  - 现在一般为人工设计
  - 特征学习通过ML技术产生好特征

## 附录1

- 受限Boltzmann机连接权更新公式推导

- 本推导来自南瓜书

  <img src="/assets/MLpics/T78.png" style="zoom:50%;" />

  <img src="/assets/MLpics/T79.png" style="zoom:50%;" />

  <img src="/assets/MLpics/T80.png" style="zoom:50%;" />

  <img src="/assets/MLpics/T81.png" style="zoom:50%;" />

