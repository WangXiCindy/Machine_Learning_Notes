---
Typora-root-url: ../assets/MLpics
---

# 支持向量机

- **支持向量机（Support Vector Machine，简称SVM）**

## 间隔与支持向量

- 分类学习：找到一个划分超平面，如何找到最合适的一个？

  - 如下图所示，直观看需要找到正中间最粗的那条超平面

    - 该划分超平面对训练样本局部扰动的“容忍性”最好
    - 容忍性：例如由于训练集的局限性或噪声，训练集外的样本可能比图中的训练样本更接近两个类的分隔界，而中间的超平面受影响最小
      - 也就是其分类结果是最鲁棒的，对未见示例的泛化能力最强

    <img src="/T84.png" style="zoom:50%;" />

- 划分超平面线性方程$w^Tx+b=0$

  - 其中$w=(w_ 1;w_ 2;...;w_ d)$为法向量，决定了超平面的方向

  - b为位移，决定了超平面与原点之间的距离

  - 样本空间中任意点x到超平面（w,b）的距离可写为$r=\frac{\vert w^Tx+b \vert}{\vert \vert w \vert \vert}$

  - 假设超平面（w,b）能将训练样本正确分类，即对于$(x_ i,y_ i) \in D,若 y_ i=+1，则w^Tx_ i+b <0$

  - 式（6.3）：

    $$ y=\left\{\begin{array}{cl}w^Tx_ i+b \ge +1, & y_ i=+1 \\ w^Tx_ i+b \le -1 & y_ i=-1  \end{array}\right. $$

  - 如下图所示

    - 距离最近的训练样本使上式等号成立，即为**支持向量（supprot vector）**
    - **间隔（margin）**：两个异类支持向量到超平面的距离为$\gamma=\frac{2}{\vert \vert w \vert \vert}$
      - 间隔与w有关也与b有关，因为b通过约束隐式地影响着w的取值，进而对间隔产生影响

    <img src="/T85.png" style="zoom:50%;" />

    - **最大间隔（maximum margin）**：找到满足6.3的w和b使得$\gamma$最大

      - （6.5）

        $max_ {w,b} \frac{2}{\vert \vert w \vert \vert}$

        $s.t. \  y_ i(w^Tx_ i+b) \ge 1  \ \ i=1,2,...m$

      - 最大化$\vert \vert w \vert \vert ^{-1}$，等价于最小化$\vert \vert w \vert \vert ^2$

      - 重写得到SVM的基本型（6.6）

        $max_ {w,b} \frac{\vert \vert w \vert \vert ^2}{2}$

        $s.t. \  y_ i(w^Tx_ i+b) \ge 1  \ \ i=1,2,...m$

## 对偶问题

- 求解6.6得到大间隔划分超平面对应的模型$f(x)=w^Tx+b$

- 方法1：

  - 式子6.6本身是一个**凸二次规划（convex quadratic programming）**，可以直接用优化计算包求解

- 方法2:

  - 使用拉格朗日乘子法得到**对偶问题（dual problem）**

  - 对式6.6的每条约束添加拉格朗日乘子$\alpha_ i \ge 0$，则拉格朗日函数可写为

    （6.8）$L(w,b,\alpha)=\frac{1}{2} \vert \vert w \vert \vert ^2+\sum_ {i=1}^m \alpha_ i(1-y_ i(w^Tx_ i+b))$

  - 其中$\alpha=(\alpha_ 1;\alpha_ 2;...;\alpha_ m)$

  - 令$L(w,b,\alpha)$对w和b的偏导为0可得

    （6.9）$w=\sum_ {i=1}^m \alpha_ iy_ ix_ i$

    （6.10）$0=\sum_ {i=1}^m \alpha_ iy_ i$        （对w求导再对b求导得到）

  - 把6.9代入6.8，可以将w和b消除，考虑6.10的约束，就得到对偶问题（6.11）

    $max_ {\alpha} \sum_ {i=1}^m \alpha_ i-\frac{1}{2}\sum_ {i=1}^m \sum_ {j=1}^m \alpha_ i \alpha_ j y_ i y_ j x_ i^T x_ j$

    $s.t. \ \  \sum_ {i=1}^m \alpha_ i y_ i=0$

    $\alpha_ i \ge 0,i=1,2,...,m$

  - 解出$\alpha$（拉格朗日乘子），$\alpha_ i$对应$(x_ i,y_ i)$

    - 求出w和b（6.12）

      $f(x)=w^Tx+b=\sum_ {i=1}^m \alpha_ i y_ i x_ i^Tx+b$

  - 6.6中有不等式约束，因此上述过程需要满足**KKT（Karush-Kuhn-Tucker）**

    - 证明见附录B.1

    $$ \left\{\begin{array}{cl}\alpha_ i \ge 0; \\ 1-y_ if(x_ i) \le 0; \\ \alpha_ i(y_ i f(x_ i)-1)=0  \end{array}\right. $$

    - 若$\alpha_ i=0$，则该样本不会在6.12中出现，不会对f(x)有任何影响
    - 若$\alpha_ i>0$，则必有$y_ i f(x_ i)=1$，对应样本点位于最大间隔边界上
    - SVM性质：训练完成后，大部分的训练样本都不需要保留，最终模型仅与支持向量有关

  - 求解6.11:

    - 二次规划问题（见附录B.2）

    - 但该问题的规模正比于训练样本数，会造成很大开销

    - **SMO（Sequential Minimal Optimization）**算法

      - 先固定$\alpha_ i$之外的所有参数，然后求$\alpha_ i$上的极值

      - 由于存在约束$\sum_ {i=1}^m \alpha_ i y_ i=0$

      - 若固定$\alpha_ i$之外的其他变量，则其可由其他变量导出

      - SMO每次选择$\alpha_ i,\alpha_ j$，并固定其他参数，进行初始化

      - SMO不断进行以下步骤进行收敛

        - 选取一对需要更新的变量$\alpha_ i,\alpha_ j$
        - 固定$\alpha_ i,\alpha_ j$以外的参数，求解6.11获得更新后的$\alpha_ i,\alpha_ j$

      - SMO先选取违背KKT条件程度最大的变量

        - 只需选取的$\alpha_ i,\alpha_ j$中有一个不满足KKT，目标函数就会在迭代后减小
        - KKT条件违背的程度越大，则变量更新后可能导致的目标函数值减幅越大

      - 再选取一个使目标函数值见效最快的变量

        - 减幅的复杂度过高，因此SMO采用了启发式
          - 选取的两变量所对应样本之间的间隔最大
          - 这样的两个变量会有很大的差别，与对两个相似的变量进行更新对比，对它们进行更新会带给目标函数值更大的变化

      - SMO高效的原因

        - 在固定其他参数后，仅优化两个参数的过程能做到非常高效

        - 仅考虑$\alpha_ i,\alpha_ j$，6.11中的约束可以重写为

          $\alpha_ iy_ i+\alpha_ jy+j=c，\alpha_ i \ge 0，\alpha_ j \ge 0$

        - 其中$c=-\sum_ {k \not=i,j}\alpha_ ky_ k$是使得$\sum_ {i=1}^m \alpha_ i y_ i=0$成立的常数

        - 用$\alpha_ iy_ i+\alpha_ jy_ j=c$消去6.11中的变量$\alpha_ j$，得到一个关于$\alpha_ i$的单变量二次规划问题，仅有约束$\alpha_ i \ge 0$

        - 这样的二次规划问题具有闭式解，于是不必调用数值优化算法即可高效计算出更新后的$\alpha_ i,\alpha_ j$

      - 确定偏移项b，对任意$(x_ s,y_ s)$都有$y_ sf(x_ s)=1$

        - （6.17）：$y_ s(\sum_ {t \in S} \alpha_ i y_ i x_ i^T x_ s+b)=1$
        - $S=\{i \vert \alpha_ i >0,i=1,2,...,m\}$为所有支持向量的下标集
        - 理论上，可选取任意支持向量并通过求解6.17获得b
        - 现实上，使用更鲁棒的做法
          - 使用所有支持向量求解的平均值
          - $b=\frac{1}{\vert S \vert} \sum_ {s \in S}(y_ s-\sum_ {i \in S} \alpha_ i y_ i x_ i^T x_ s)$

## 核函数

- 现实任务中，原始样本空间内也许并不存在一个能正确划分两类样本的超平面

  - 例如异或问题

  <img src="/T86.png" style="zoom:50%;" />

- 如何解决

  - 将样本从原始空间映射到一个更高维的特征空间，使得样本在这个空间内线性可分，如上图，将二维空间映射到三维
  - 如果原始空间是有限维（属性数有限），一定存在一个高维特征空间使样本可分

- 公式推导

  - $\phi(x)$表示将x映射后的特征向量

  - 超平面所对应的模型$f(x)=w^T\phi(x)+b$

  - w和b是模型参数，类似式6.6，有

    $min_ {w,b}=\frac{1}{2} \vert \vert w \vert \vert ^2$

    $s.t. \ \ y_ i(w^T\phi(x_ i)+b) \ge 1,\ i=1,2,...,m$

  - 对偶问题是（6.21）

    $max_ {\alpha} \sum_ {i=1}^m \alpha_ i-\frac{1}{2}\sum_ {i=1}^m \sum_ {j=1}^m \alpha_ i \alpha_ j y_ i y_ j \phi(x_ i)^T \phi(x_ j)$

    $s.t. \ \  \sum_ {i=1}^m \alpha_ i y_ i=0$

    $\alpha_ i \ge 0,i=1,2,...,m$

  - 求解6.21

    - $\phi(x_ i)^T\phi(x_ j)$是样本$x_ i,x_ j$映射到特征空间之后的内积

    - 为避免由于特征空间维数过高导致的计算困难，设计函数

      $\kappa(x_ i,x_ j) = \ <\phi(x_ i),\phi(x_ j)> \ =\phi(x_ i)^T\phi(x_ j)$

    - 即$x_ i,x_ j$在特征空间的内积等于它们在原始样本空间中通过函数$k(·,·)$计算的结果

    - 因此6.21可以重写为

      $max_ {\alpha} \sum^m_ {i=1}\alpha_ i-\frac{1}{2}\sum_ {i=1}^m\sum_ {j=1}^m \alpha_ i \alpha_ j y_ i y_ j k(x_ i,x_ j)$

      $s.t. \ \ \sum_ {i=1}^m \alpha_ i y_ i=0$

      $\alpha_ i \ge 0,i=1,2,...,m$

    - 求解得到式6.24

      $\begin{aligned} f(x)&=w^T\phi(x)+b \\ &= \sum_ {i=1}^m \alpha_ i y_ i \phi(x_ i)^T \phi(x)+b \\ &=\sum_ {i=1}^m \alpha_ i y_ i k(x,x_ i)+b\end{aligned}$

    - 这里的函数$\kappa(·,·)$就是**“核函数（kernel function）”**

    - 6.24显示出模型最优解可通过训练样本的核函数展开，这一展式亦称**“支持向量展式”（support vector expansion）**

- **核函数**

  - 当$\chi$为输入空间，$\kappa(·,·)$是定义在$\chi \times \chi$上的对称函数

  - 则$\kappa$是核函数当且仅当对于任意数据$D= \{x_ 1,x_ 2,...,x_ m\}$

  - **“核矩阵”（kernel matrix）K**总是半正定的

    $$\mathbf{K}=\left[\begin{array}{ccccc}\kappa\left(\boldsymbol{x}_ {1}, \boldsymbol{x}_ {1}\right) & \cdots & \kappa\left(\boldsymbol{x}_ {1}, \boldsymbol{x}_ {j}\right) & \cdots & \kappa\left(\boldsymbol{x}_ {1}, \boldsymbol{x}_ {m}\right) \\ \vdots & \ddots & \vdots & \ddots & \vdots \\ \kappa\left(\boldsymbol{x}_ {i}, \boldsymbol{x}_ {1}\right) & \cdots & \kappa\left(\boldsymbol{x}_ {i}, \boldsymbol{x}_ {j}\right) & \cdots & \kappa\left(\boldsymbol{x}_ {i}, \boldsymbol{x}_ {m}\right) \\ \vdots & \ddots & \vdots & \ddots & \vdots \\ \kappa\left(\boldsymbol{x}_ {m}, \boldsymbol{x}_ {1}\right) & \cdots & \kappa\left(\boldsymbol{x}_ {m}, \boldsymbol{x}_ {j}\right) & \cdots & \kappa\left(\boldsymbol{x}_ {m}, \boldsymbol{x}_ {m}\right)\end{array}\right]$$

  - 只要一个对称函数所对应的核矩阵半正定，它就能作为核函数使用

  - 对于一个半正定核矩阵，总能找到一个与之对应的映射$\phi$

  - 换言之，任何一个核函数都隐式定义了一个**“再生核希尔伯特空间”（Reproducing Kernel Hilbert Space，简称RKHS）**的特征空间

  - 我们希望样本在特征空间内线性可分，因此特征空间的好坏对支持向量机的性能至关重要

- 核函数选择

  - SVM的最大变数

  - 常用核函数

    <img src="/T87.png" style="zoom:50%;" />

  - 函数组合

    - 若$\kappa_ 1，\kappa_ 2$为核函数，则对于任意正数$\gamma_ 1，\gamma_ 2$，其线性组合$\gamma_ 1 \kappa_ 1+\gamma_ 2 \kappa_ 2$也是核函数
    - 若$\kappa_ 1，\kappa_ 2$为核函数，则核函数的直积$\kappa_ 1 \otimes \kappa_ (x,z)=\kappa_ 1(x,z)\kappa_ 2(x,z)$也是核函数
    - 若$\kappa_ 1$为核函数，则对于任意函数$g(x)$，$\kappa(x,z)=g(x)\kappa_ 1(x,z)g(z)$也是核函数

## 软间隔与正则化

- 现实中很难确定合适的核函数使得训练样本在特征空间中线性可分

- 即使找到也很难判断是否是由于过拟合造成的

- **硬间隔（hard margin）**：要求所有样本均满足约束（6.3），前面提到的SVM都是如此

- **软间隔（soft margin）**：允许SVM在一些样本上出错

  - 式6.28：$y_ i(w^Tx_ i+b) \ge 1$

  <img src="/T88.png" style="zoom:50%;" />

- **软间隔支持向量机**

  - 最大化间隔的同时，不满足约束的样本应该尽可能少，因此优化目标为

  - 式（6.29）：$min_ {w,b} \frac{1}{2} \vert \vert w \vert \vert ^2+C\sum_ {i=1}^m l_ {0/1}(y_ i(w^Tx_ i+b)-1)$ 

  - 其中$C >0$是一个常数$l_ {0/1}$是0/1损失函数

    $$ l_ {0/1}(z)=\left\{\begin{array}{cl}1, &if \  z<0; \\ 0, &otherwise  \end{array}\right. $$

  - 当C为无穷大时，6.29迫使所有样本均满足约束（6.28），于是式6.29等价于6.6

  - 当C取有限值时，6.29允许一些样本不满足约束

  - **替代损失（surrogate loss）**

    - 由于$l_ {0/1}$非凸、非连续，数学性质不好，使得式6.29不易直接求解

    - 需要用其他一些函数替代$l_ {0/1}$

    - 常用的替代损失函数

      - hinge损失：$l_ {hinge}(z)=max(0,1-z)$
      - 指数损失（exponential loss）：$l_ {exp}(z)=exp(-z)$
      - 对率损失（logistic loss）：$l_ {log}(z)=log(1+exp(-z))$
        - 对率损失函数通常表示为$l_ {log}(·)$而非$ln(·)$

      <img src="/T89.png" style="zoom:50%;" />

    - 若采用hinge损失，则6.29变成

      ​	$min_ {w,b} \frac{1}{2} \vert \vert w \vert \vert ^2+C\sum_ {i=1}^m max(0,1-y_ i(w^Tx_ i+b))$ 

      - $max(0,1-y_ i(w^Tx_ i+b))=\xi_ i$

      - 引入**松弛变量（slack variables）**$\xi_ i \ge 0$，可将上式重写

        - 当$1-y_ i(w^Tx_ i+b)>0$，$1-y_ i(w^Tx_ i+b)=\xi_ i$
        - 当$1-y_ i(w^Tx_ i+b) \le 0$，$\xi_ i=0$

      - 得到软间隔SVM（6.35）

        $min_ {w,b,\xi_ i} \frac{1}{2} \vert \vert w \vert \vert ^2+C \sum_ {i=1}^m \xi_ i$

        $s.t. \ \ y_ i(w^Tx_ i+b) \ge 1- \xi_ i$

        $\xi_ i \ge 0,i=1,2,...,m$

      - 每一个样本对应松弛变量，表征该样本不满足约束6.28的程度

      - 仍是一个二次规划问题，通过拉格朗日乘子法

        $$\begin{aligned}L(w,b,\alpha,\xi,\mu) =&\frac{1}{2} \vert \vert w \vert \vert ^2+C \sum_ {i=1}^m \xi_ i \\ &+\sum_ {i=1}^m \alpha_ i(1-\xi_ i-y_ i(w^Tx_ i+b))-\sum_ {i=1}^m \mu_ i \xi_ i \end{aligned}$$

        - $\alpha_ i \ge 0,\mu_ i \ge 0$是拉格朗日乘子

        - 令$L(w,b,\alpha,\xi,\mu)$对$w,b,\xi_ i$的偏导为0可得

          式6.37：$w=\sum_ {i=1}^m \alpha_ i y_ i x_ i$

          式6.38：$0=\sum_ {i=1}^m \alpha_ i y_ i$

          式6.39：$C=\alpha_ i+\mu_ i$

        - 将式6.37-6.39代入式6.36即可得到式6.35的对偶问题（6.40）

          $max_ {\alpha} \sum_ {i=1}^m \alpha_ i -\frac{1}{2}\sum_ {i=1}^m \sum_ {j=1}^m \alpha_ i \alpha_ j y_ i y_ j x_ i^T x_ j$

          $s.t. \ \ \sum_ {i=1}^m \alpha_ i y_ i=0$

          $0 \le \alpha_ i \le C,i=1,2,...,m$

        - 将上式与软间隔下的对偶问题6.11只有对偶变量的约束不同，所以采用同样的算法求解，因此KKT

          $$ \left\{\begin{array}{cl}\alpha_ i \ge 0, & \mu_ i \ge 0, \\ 1-\xi_ i-y_ if(x_ i) \le 0; \\ \alpha_ i(y_ i f(x_ i)-1+\xi_ i)=0 \\ \xi_ i \ge 0, &\mu_ i \xi_ i=0 \end{array}\right. $$

          - 对任意训练样本$(x_ i,y_ i)$,总有$\alpha_ i=0$或$y_ if(x_ i)=1-\xi_ i$
          - 若$\alpha_ i=0$，则该样本不会对$f(x)$有任何影响
          - 若$\alpha_ i>0$则必有$y_ i f(x_ i)=1-\xi_ i$，即该样本是支持向量
          - 由6.39可知，
            - 若$\alpha_ i<C$，则$\mu_ i > 0$，进而$\xi_ i = 0$，则该样本恰在最大间隔边界上
            - 若$\alpha_ i=C$，则$\mu_ i = 0$，此时
              - 若$\xi_ i \le 1$则该样本落在最大间隔内部
              - 若$\xi_ i > 1$则该样本被错误分类

      - 软间隔SVM的最终模型仅与支持向量有关

        - hinge函数有一处“平坦”的0区域，使得解具有稀疏性

    - 使用对率损失函数，几乎就得到了对率回归模型

      - 和对率回归的异同
        - SVM与对率回归的优化目标相近，一般性能相当
        - 对率回归的优势在于其输出具有自然的概率意义；SVM输出不具有概率意义，欲得到概率输出需进行特殊处理
        - 对率回归能直接用于多分类任务；SVM为此则需要进行推广
      - 光滑的单调递减函数，不能导出类似支持向量的概念，依赖于更多的训练样本，预测开销大

      

- **正则化（regularization）**

  - 可理解为对不希望得到的结果进行惩罚，从而使得优化过程趋向于希望目标
    - 从贝叶斯估计角度看，正则化项是提供了模型的先验概率
  - 损失函数的共性：优化目标中的第一项用来描述划分超平面的“间隔”大小，另一项用于表示训练集上的误差，可以写为（6.42）$min_ {f} \Omega(f)+C\sum_ {i=1}^m l(f(x_ i),y_ i)$
    - $\Omega(f)$称为**结构风险（structural risk）**，描述模型f的某些性质
    - $\sum_ {i=1}^m l(f(x_ i),y_ i)$称为**经验风险（empirical risk）**，描述模型与训练数据的契合程度
    - C用于对二者进行折中
    - 某种意义上6.42可以称为正则化问题
      - $\Omega(f)$称为**正则化项**
      - C称为**正则化常数**
        - $L_ p$范数是常用的正则化项
          - $L_ 2（\vert \vert w \vert \vert_ 2）$范数倾向于w的分量取值尽量均衡（非0分量个数尽量稠密
          - $L_ 0$范数$（\vert \vert w \vert \vert_ 0）$和$L_ 1（\vert \vert w \vert \vert_ 1）$$范数则倾向于w的分量尽量稀疏，即非0分量个数尽量少

## 支持向量回归

- 与传统模型不同，**支持向量回归（Support Vector Regression，简称SVR）**假设能容忍$f(x),y$之间最多有$\epsilon$的偏差

  - 如下图所示，以$f(x)$为中心，构建了一个宽度为$2\epsilon$的间隔带，若落入间隔带，则认为预测正确

  <img src="/T90.png" style="zoom:50%;" />

- SVR问题形式化（6.43）

  $min_ {w,b} \frac{1}{2} \vert \vert w \vert \vert ^2+C \sum_ {i=1}^m l_ \epsilon(f(x_ i)-y_ i)$	

  - C为正则化常数

  - $l_ \epsilon$是**$\epsilon$-不敏感损失（$\epsilon$-insensitive loss）**函数

    $$l_ \epsilon(z)=\left\{\begin{array}{cl}0, & if \vert z \vert \le \epsilon; \\ \vert z \vert - \epsilon & otherwise.  \end{array}\right. $$

    <img src="/T91.png" style="zoom:50%;" />

  - 引入松弛变量$\xi_ i，\hat{\xi_ i}$，可以将6.43重写为

    $min_ {w,b,\xi_ i,\hat{\xi_ i}} \frac{1}{2} \vert \vert w \vert \vert ^2+C \sum_ {i=1}^m (\xi_ i+\hat{\xi_ i})$

    $s.t. \ \ f(x_ i)-y_ i \le \epsilon+\xi_ i$

    $y_ i-f(x_ i) \le \epsilon+\hat{\xi_ i}$

    $\xi_ i \ge 0,\hat{\xi_ i} \ge 0,i=1,2,...,m$	

  - 引入拉格朗日乘子$\mu_ i \ge 0,\hat{\mu_ i} \ge 9,\alpha_ i \ge 0,\hat{\alpha_ i} \ge 0$，得到式（6.46）

    $L(\boldsymbol{w}, b, \boldsymbol{\alpha}, \hat{\boldsymbol{\alpha}}, \boldsymbol{\xi}, \hat{\boldsymbol{\xi}}, \boldsymbol{\mu}, \hat{\boldsymbol{\mu}})$
    $=\frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_ {i=1}^{m}\left(\xi_ {i}+\hat{\xi}_ {i}\right)-\sum_ {i=1}^{m} \mu_ {i} \xi_ {i}-\sum_ {i=1}^{m} \hat{\mu}_ {i} \hat{\xi}_ {i}$
    $+\sum_ {i=1}^{m} \alpha_ {i}\left(f\left(\boldsymbol{x}_ {i}\right)-y_ {i}-\epsilon-\xi_ {i}\right)+\sum_ {i=1}^{m} \hat{\alpha}_ {i}\left(y_ {i}-f\left(\boldsymbol{x}_ {i}\right)-\epsilon-\hat{\xi}_ {i}\right)$

  - 将式$f(x)=w^Tx+b$代入，对w，b，$\xi_ i，\hat{\xi_ i}$偏导为0可得

    $$\begin{aligned} \boldsymbol{w} &=\sum_ {i=1}^{m}\left(\hat{\alpha}_ {i}-\alpha_ {i}\right) \boldsymbol{x}_ {i} \\ 0 &=\sum_ {i=1}^{m}\left(\hat{\alpha}_ {i}-\alpha_ {i}\right) \\ C &=\alpha_ {i}+\mu_ {i} \\ C &=\hat{\alpha}_ {i}+\hat{\mu}_ {i} \end{aligned}$$

  - 上面几式代入6.46，得到SVR的对偶问题（6.51）

    $$\begin{aligned} \max _ {\boldsymbol{\alpha}, \hat{\boldsymbol{\alpha}}} & \sum_ {i=1}^{m} y_ {i}\left(\hat{\alpha}_ {i}-\alpha_ {i}\right)-\epsilon\left(\hat{\alpha}_ {i}+\alpha_ {i}\right) \\ &-\frac{1}{2} \sum_ {i=1}^{m} \sum_ {j=1}^{m}\left(\hat{\alpha}_ {i}-\alpha_ {i}\right)\left(\hat{\alpha}_ {j}-\alpha_ {j}\right) \boldsymbol{x}_ {i}^{\mathrm{T}} \boldsymbol{x}_ {j} \\ \text { s.t. } & \sum_ {i=1}^{m}\left(\hat{\alpha}_ {i}-\alpha_ {i}\right)=0 \\ & 0 \leqslant \alpha_ {i}, \hat{\alpha}_ {i} \leqslant C \end{aligned}$$

  - 上述过程需要满足KKT条件

    $$\left\{\begin{array}{l}
    \alpha_ {i}\left(f\left(\boldsymbol{x}_ {i}\right)-y_ {i}-\epsilon-\xi_ {i}\right)=0 \\
    \hat{\alpha}_ {i}\left(y_ {i}-f\left(\boldsymbol{x}_ {i}\right)-\epsilon-\hat{\xi}_ {i}\right)=0 \\
    \alpha_ {i} \hat{\alpha}_ {i}=0, \xi_ {i} \hat{\xi}_ {i}=0 \\
    \left(C-\alpha_ {i}\right) \xi_ {i}=0,\\ \left(C-\hat{\alpha}_ {i}\right) \hat{\xi}_ {i}=0
    \end{array}\right.$$

    - 当且仅当$f(\boldsymbol{x}_ {i})-y_ {i}-\epsilon-\xi_ {i}=0$（不落入间隔带·中）时$\alpha_ i$和$\hat{\alpha_ i}$能取非0值
    - $f(\boldsymbol{x}_ {i})-y_ {i}-\epsilon-\xi_ {i}=0$和$f(\boldsymbol{x}_ {i})-y_ {i}-\epsilon-\hat{\xi_ {i}=0}$不能同时成立，$\alpha_ i,\hat{\alpha_ i}$至少有一个为0

  - 将$w =\sum_ {i=1}^{m}(\hat{\alpha}_ {i}-\alpha_ {i}) x_ {i}$代入$f(x)=w^Tx+b$，则SVR的解形如

    $f(x)=\sum_ {i=1}^m(\hat{\alpha_ i}-\alpha_ i)x_ i^Tx+b$

    - 能使上式中的$(\hat{\alpha_ i}-\alpha_ i) \not=0$的样本即为SVR的支持向量，必落在间隔带之外

    - 间隔带中的样本均满足$\alpha_ i$和$\hat{\alpha_ i}$均为0

      - 其解仍然具有稀疏性

    - 由KKT条件，若$0 < \alpha_ i <C$必有$\xi_ i=0$，进而

      $b=y_ i+\epsilon-\sum_ {i=1}^m(\hat{\alpha_ i}-\alpha_ i)x_ i^Tx$

      - 在求解6.51得到$\alpha_ i$后，理论上可以任意选取满足$0 < \alpha_ i <C$的样本求得b
      - 只见中采用更鲁棒的方法，选取多个满足条件的样本求b后求平均值

  - 若考虑特征映射形式（6.19），则

    $w =\sum_ {i=1}^{m}(\hat{\alpha}_ {i}-\alpha_ {i}) \phi(x_ {i})$

    - SVR可以表示为

      $f(x)=\sum_ {i=1}^{m}(\hat{\alpha}_ {i}-\alpha_ {i}) \kappa(x,x_ i)+b$

      $\kappa(x,x_ i)=\phi(x_ i)^T\phi(x_ j)$为核函数

## 核方法

- **表示定理（representer theorem）**

  - 令$\Bbb{H}$为核函数$\kappa$对应的再生核希尔伯特空间，$\vert \vert h \vert \vert_ {\Bbb{H}}$表示$\Bbb{H}$空间中关于h的范数，对于任意单调递增函数$\Omega:[0,\infin] \rarr \Bbb{R}$和任意非负损失函数$l:\Bbb{R}^m \rarr [0,\infin]$，优化问题（式6.57）

    $\min _ {h \in \mathbb{H}} F(h)=\Omega\left(\|h\|_ {\mathbb{H}}\right)+\ell\left(h\left(\boldsymbol{x}_ {1}\right), h\left(\boldsymbol{x}_ {2}\right), \ldots, h\left(\boldsymbol{x}_ {m}\right)\right)$

    的解可总写为

    $h^{*}(\boldsymbol{x})=\sum_ {i=1}^{m} \alpha_ {i} \kappa\left(\boldsymbol{x}, \boldsymbol{x}_ {i}\right)$

  - 表示定理对$\Omega$要求单调递增，不需要是凸函数

    - 对于一般的损失函数和正则化项，优化问题6.57的最优解$h^*(x)$都可以表示为核函数$\kappa(x,x_ i)$的线性组合

- **核方法（kernel methods）**

  - 基于核函数的学习方法

- **线性判别分析（Kernelized Linear Discriminant Analysis，简称KLDA）**

  - 通过**核化**（引入核函数）来为线性学习器拓展为非线性

  - 过程

    - 假设通过$\phi:\chi \rarr \Bbb{F}$将样本映射到一个特征空间$\Bbb{F}$，执行线性判别分析，得到

      $h(x)=w^T \phi(x)$

    - KLDA的学习目标

      $max_ w J(w)=\frac{w^T S_ b^ \phi w}{w^T S_ w^ \phi w}$

      - $S_ b ^\phi,S_ w^\phi$分别为训练样本在特征空间中的类间散度矩阵和类内散度矩阵

    - $X_ i$表示第$i \in \{0,1\}$在类样本的集合，其样本数为$m_ i$，总样本数为$m=m_ 0+m_ 1$，第i类样本在特征空间的均值为

      $\mu_ i ^ \phi=\frac{1}{m_ i}\sum_ {x \in X_ i} \phi(x)$

    - 两个散度矩阵分别为

      $\mathbf{S}_ {b}^{\phi} =\left(\boldsymbol{\mu}_ {1}^{\phi}-\boldsymbol{\mu}_ {0}^{\phi}\right)\left(\boldsymbol{\mu}_ {1}^{\phi}-\boldsymbol{\mu}_ {0}^{\phi}\right)^{\mathrm{T}}  $

      $\mathbf{S}_ {w}^{\phi} =\sum_ {i=0}^{1} \sum_ {\boldsymbol{x} \in X_ {i}}\left(\phi(\boldsymbol{x})-\boldsymbol{\mu}_ {i}^{\phi}\right)\left(\phi(\boldsymbol{x})-\boldsymbol{\mu}_ {i}^{\phi}\right)^{\mathrm{T}} $

    - 一般难以知道映射的具体形式，使用核函数来隐式表达映射和特征空间

    - $J(w)$作为6.57的损失函数可得到式（6.64）

      $h(\boldsymbol{x})=\sum_ {i=1}^{m} \alpha_ {i} \kappa\left(\boldsymbol{x}, \boldsymbol{x}_ {i}\right)$

      $w=\sum_ {i=1}^m \alpha_ i \phi(x_ i)$

    - 令$K \in \Bbb{R}^{m \times m}$为核函数对应的核矩阵，$(K)_ {ij}=\kappa(x_ i,x_ j)$

    - $1_ i \in \{1,0\}^{m \times 1}$为第i类样本的指示向量

      - $1_ i$的第j个分量为1当且仅当$x_ j \in X_ i$，否则为0

        $\hat{\mu_ 0}=\frac{1}{m_ 0}K1_ 0$

        $\hat{\mu_ 1}=\frac{1}{m_ 1}K1_ 1$

        $M=(\hat{\mu_ 0}-\hat{\mu_ 1})(\hat{\mu_ 0}-\hat{\mu_ 1})^T$

        $N=KK^T-\sum_ {i=0}^1 m_ i\hat{\mu_ i}\hat{\mu_ i}^T$

      - 于是KLDA的学习目标等价为

        $max_ \alpha J(\alpha)=\frac{\alpha^T M \alpha}{\alpha^T N \alpha}$

      - 求得$\alpha$之后可以由6.64得到投影函数$h(x)$

