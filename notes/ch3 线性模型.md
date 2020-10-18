---
Typora-root-url: ../assets/MLpics
---
# 线性模型

## 基本形式

- $f(x)=w _ {1}x _ 1+w _ 2x _ 2+...+w _ dx _ d+b$
- 向量形式$f(x)=w^Tx+b$
- 许多非线性模型可以在线性模型的基础上通过引入层级结构或高维影射得到
- 由于w直观表达了各属性在预测中的重要性，因此线性模型有很好的**可解释性**

## 线性回归

- 数据集$D={(x _ 1,y _ 1),(x _ 2,y _ 2)...,(x _ m,y _ m)}$

- $x _ i=(x _ i1;x _ i2;...;x _ id)$

- 线性回归试图学得一个线性模型以尽可能的预测实值并输出标记

- 最简单的情形：输入属性的数目只有一个，也就是说第二条的$x _ i=x _ {i1}$

  - $D=\{(x _ i,y _ i)\} _ {i=1}^m$

  - 对于离散属性

    - 若属性值之间存在序关系
      - 可通过连续化将其转化为序列值
      - 如高、矮可转化为$\{1.0,0.0\}$
    - 若属性值之间不存在序关系
      - 如果有k个属性值，则通常转化为k维向量
      - 如西瓜、南瓜可转化为$(1,0),(0,1)$

  - 线性回归试图学得$f(x _ i)=wx _ i+b,\text{使得}f\left(x _ {i}\right) \simeq y _ {i}$

    - 如何求得w和b

      - 在性能度量一节中，均方误差是回归任务最常用的，为凸函数（说明详见西瓜书P54注解）

      - 试图让均方误差最小化$$ \begin{aligned}
        \left(w^{ * }, b^{ * }\right) &=\underset{(w, b)}{\arg \min } \sum _ {i=1}^{m}\left(f\left(x _ {i}\right)-y _ {i}\right)^{2}  \\ 
        &=\underset{(w, b)}{\arg \min } \sum _ {i=1}^{m}\left(y _ {i}-w x _ {i}-b\right)^{2}
        \end{aligned} $$

      - 在2.3节中提到，均方误差的几何意义对应欧式距离，基于均方误差最小化来进行模型求解的方法为**“最小二乘法”**

      - 线性回归模型的最小二乘“参数估计”：求解$w,b$使得均方误差最小化的过程

      - 我们分别对$w,b$求导

        - $\frac{\partial E _ {(w, b)}}{\partial w}=2\left(w \sum _ {i=1}^{m} x _ {i}^{2}-\sum _ {i=1}^{m}\left(y _ {i}-b\right) x _ {i}\right)$

        - $\frac{\partial E _ {(w, b)}}{\partial b}=2\left(m b-\sum _ {i=1}^{m}\left(y _ {i}-w x _ {i}\right)\right)$

        - 令上两式为0得到$w,b$的最优解的闭式解

        - **闭式解**

          - 又名解析解，针对一些严格的公式，给出任意的自变量就可以求出其因变量,也就是问题的解, 他人可以利用这些公式计算各自的问题。
  - 除此之外，**数值解**是采用某种计算方法,如数值逼近,插值的方法 得到的解。别人只能利用数值计算的结果, 而不能随意给出自变量并求出计算值。
    
  - $b=\frac{1}{m} \sum _ {i=1}^{m}\left(y _ {i}-w x _ {i}\right)$
    
  - $w=\frac{\sum _ {i=1}^{m} y _ {i}\left(x _ {i}-\bar{x}\right)}{\sum _ {i=1}^{m} x _ {i}^{2}-\frac{1}{m}\left(\sum _ {i=1}^{m} x _ {i}\right)^{2}}$
          
    - 求解上方求导式（注意带入b）：
      
      - $w \sum _ {i=1}^{m} x _ {i}^{2} =\sum _ {i=1}^{m} y _ {i} x _ {i}-\sum _ {i=1}^{m}b x _ {i}$
      
      - $b=\frac{1}{m} \sum _ {i=1}^{m}\left(y _ {i}-w x _ {i}\right)$
      
        - $\frac{1}{m} \sum _ {i=1}^{m}y _ {i}=\bar{y}$
      
        - $\frac{1}{m} \sum _ {i=1}^{m}w x _ {i}=w\bar{x}$
      
        - $b=\bar{y}-w\bar{x}$
      
          $$ \begin{aligned} w \sum _ {i=1}^{m} x _ {i}^{2} &=\sum _ {i=1}^{m} y _ {i} x _ {i}-\sum _ {i=1}^{m}(\bar{y}-w \bar{x}) x _ {i}  \\ 
          w \sum _ {i=1}^{m} x _ {i}^{2} &=\sum _ {i=1}^{m} y _ {i} x _ {i}-\bar{y} \sum _ {i=1}^{m} x _ {i}+w \bar{x} \sum _ {i=1}^{m} x _ {i} \end{aligned}$$
      
        - 可得
      
          $$\begin{aligned}w\left(\sum _ {i=1}^{m} x _ {i}^{2}-\bar{x} \sum _ {i=1}^{m} x _ {i}\right) &=\sum _ {i=1}^{m} y _ {i} x _ {i}-\bar{y} \sum _ {i=1}^{m} x _ {i} \\ w=&\frac{\sum _ {i=1}^{m} y _ {i}\left(x _ {i}-\bar{x}\right)}{\sum _ {i=1}^{m} x _ {i}^{2}-\frac{1}{m}\left(\sum _ {i=1}^{m} x _ {i}\right)^{2}}\end{aligned}$$

- 更一般的情形是$f(x _ i)=wx _ i^T+b,\text{使得}f\left(x _ {i}\right) \simeq y _ {i}$，称之为多元线性回归

  - 同样适用最小二乘法进行估计

  - 把$w,b$放入向量形式

  - 把数据集D表示为一个$m\times(d+1)$大小的矩阵X

    - 其中每行对应一个示例

    - 该行前d个元素对应于示例的d个属性值，最后一个元素恒为1

      $$\mathbf{X}=\left(\begin{array}{ccccc}
      x _ {11} & x _ {12} & \ldots & x _ {1 d} & 1  \\ 
      x _ {21} & x _ {22} & \ldots & x _ {2 d} & 1  \\ 
      \vdots & \vdots & \ddots & \vdots & \vdots  \\ 
      x _ {m 1} & x _ {m 2} & \ldots & x _ {m d} & 1
      \end{array}\right)=\left(\begin{array}{cc}
      \boldsymbol{x} _ {1}^{\mathrm{T}} & 1  \\ 
      \boldsymbol{x} _ {2}^{\mathrm{T}} & 1  \\ 
      \vdots & \vdots  \\ 
      \boldsymbol{x} _ {m}^{\mathrm{T}} & 1
      \end{array}\right) $$

    - 把标记写成向量形式：$\hat{\boldsymbol{w}}^{*}=\underset{\hat{\boldsymbol{w}}}{\arg \min }(\boldsymbol{y}-\mathbf{X} \hat{\boldsymbol{w}})^{\mathrm{T}}(\boldsymbol{y}-\mathbf{X} \hat{\boldsymbol{w}})$

    - 令$E _ {\hat{\boldsymbol{w}}}=(\boldsymbol{y}-\mathbf{X} \hat{\boldsymbol{w}})^{\mathrm{T}}(\boldsymbol{y}-\mathbf{X} \hat{\boldsymbol{w}})=\boldsymbol{y}^{\mathrm{T}} \boldsymbol{y}-\boldsymbol{y}^{\mathrm{T}} \mathbf{X} \hat{\boldsymbol{w}}-\hat{\boldsymbol{w}}^{\mathrm{T}} \mathbf{X}^{\mathrm{T}} \boldsymbol{y}+\hat{\boldsymbol{w}}^{\mathrm{T}} \mathbf{X}^{\mathrm{T}} \mathbf{X} \hat{\boldsymbol{w}}$
    
      - 对$E _ {\hat{\boldsymbol{w}}}$求导
  - 矩阵微分公式$\frac{\partial \boldsymbol{a}^{\mathrm{T}} \boldsymbol{x}}{\partial \boldsymbol{x}}=\frac{\partial \boldsymbol{x}^{\mathrm{T}} \boldsymbol{a}}{\partial \boldsymbol{x}}=\boldsymbol{a}, \frac{\partial \boldsymbol{x}^{\mathrm{T}} \mathbf{A} \boldsymbol{x}}{\partial \boldsymbol{x}}=\left(\mathbf{A}+\mathbf{A}^{\mathrm{T}}\right) \boldsymbol{x}$
      - $\frac{\partial E _ {\hat{\boldsymbol{w}}}}{\partial \hat{\boldsymbol{w}}}=0-X^Ty-X^Yy+(X^TX+X^TX)w \\  \frac{\partial E _ {\hat{\boldsymbol{w}}}}{\partial \hat{\boldsymbol{w}}}=2 \mathbf{X}^{\mathrm{T}}(\mathbf{X} \hat{\boldsymbol{w}}-\boldsymbol{y})$
      
    - 令上式为0得到$\hat{\boldsymbol{w}}$的最优解的闭式解

      - 当$\mathbf{X} ^T\mathbf{X} $为**满秩矩阵/正定矩阵**时

        - 补充：

          - 满秩矩阵：设A是n阶矩阵, 若r（A） = n, 则称A为满秩矩阵。但满秩不局限于n阶矩阵。

            - 在线性代数中，一个矩阵A的列秩是A的线性独立的纵列的极大数目。类似地，行秩是A的线性无关的横行的极大数目。

            - 如下图

              <img src="/T24.png" style="zoom:50%;" />

          - 正定矩阵

            - 设M是n阶方阵，如果对任何非零向量z，都有$z^TMz>0$，就称M为正定矩阵。

        - 令求导式为0得到

          - $\hat{\boldsymbol{w}}^{*}=\left(\mathbf{X}^{\mathrm{T}} \mathbf{X}\right)^{-1} \mathbf{X}^{\mathrm{T}} \boldsymbol{y}$
      - 令$\hat{\boldsymbol{x _ i}}=(x _ i,1)$，最终学得的多元线性回归模型为
        
          - $f(\hat{\boldsymbol{x _ i}})= \hat{\boldsymbol{x _ i}}^T\left(\mathbf{X}^{\mathrm{T}} \mathbf{X}\right)^{-1} \mathbf{X}^{\mathrm{T}} \boldsymbol{y}$
    
  - 现实生活中$\mathbf{X} ^T\mathbf{X} $往往不是满秩矩阵，此时可以解出多个$\hat{\boldsymbol{w}}$，他们都能使均方误差最小化，选择哪一个解作为输出将由学习算法的归纳偏好决定
    
      - 常见做法为引入**正则化(regularization)**

- **广义线性模型**

  - 对数线性回归
    - 我们得到了线性回归模型之后，可否令模型预测值逼近$y$的衍生物
    - 譬如我们认为示例所对应的输出标记是在指数尺度上变化
      - $lny=w^Tx+b$
      - 虽然形式上仍然是线性回归，但实质上已经是在求取空间到输出空间的非线性函数映射
      - <img src="/T25.png" style="zoom:50%;" />
      - 这里对数函数起到了将线性回归模型的预测值与真实标记联系起来的作用
  - 广义线性模型
    - 考虑单调可微函数$g(·)$（称之为联系函数）
    - $y=g^{-1}(w^Tx+b)$
    - 对数线性回归是广义线性模型在$g(·)=ln(·)$时的特例

## 对数几率回归

- 先决条件

  - 二分类任务
  - 输出标记为$y\in\{0,1\}$
  - 线性回归模型产生的预测值$z=w^Tx+b$是实际值
  - 我们需要将z转换为0/1值

- $y=\frac{1}{1+e^{-(w^Tx+b)}}$  公式推导

  - 单位阶跃函数（unit-step function）$$ y=\left\{\begin{array}{cl}
  0, & z<0  \\ 
    0.5, & z=0  \\ 
    1, & z>0    \end{array}\right. $$
  
  - 若预测值z大于0就判为正例，小于0则判为反例，预测值为临界值0则可以任意判别

    <img src="/T26.png" style="zoom:50%;" />

  - 从图中可以看出，单位阶跃函数不连续，因此不能直接用作广义线性模型中的$g^-(·)$

    - 找出一个替代函数，并且单调可微---对数几率函数

      - $y=\frac{1}{1+e^{-z}}$

      - 对数几率函数是一个**Sigmoid函数**，将z转化成一个接近0或1的y值

      - 将其作为$g^-(·)$得到

        $y=\frac{1}{1+e^{-(w^Tx+b)}}$

      - 上式可变化为$ln{\frac{y}{1-y}}=w^Tx+b$

      - 将y视为样本x作为正例的可能性，1-y为其反例可能性，$\frac{y}{1-y}$称之为**几率（odds）**，将其取对数称之为**对数几率（log odds/logit）**

  - 虽然$y=\frac{1}{1+e^{-(w^Tx+b)}}$叫对数几率回归，但实际上却是一种分类学习方法
  - 优点
      - 对分类可能性进行建模，无需假设数据分布，避免了假设分布不准确所带来的问题
      - 不是仅预测出“类别”，而是可得到近似概率预测，适合许多需要利用概率辅助决策的任务
      - 对率函数是任意阶可导的凸函数，现有的许多数值优化算法可以直接用于求取最优解
  
- 确定$w,b$

  - 若将y视作**类后验概率估计**p($y=1 \vert x$)，则

    $$\begin{aligned} ln{\frac{y}{1-y}}=&w^Tx+b  \\  ln\frac{p(y=1 \vert x)}{p(y=0 \vert x)}=&w^Tx+b  \\  p(y=1 \vert x)=&\frac{e^{w^Tx+b}}{1+e^{w^Tx+b}}  \\  p(y=0 \vert x)=&\frac{1}{1+e^{w^Tx+b}}\end{aligned}$$

    （最后两公式为3.23和3.24）

    - 通过极大似然法

      - 补充知识

        <img src="/T27.png" style="zoom:30%;" />

      - 给定数据集$\{(x _ i,y _ i)\}^m _ {i=1}$，对率回归模型最大化“对数似然”

        $\ell(\boldsymbol{w}, b)=\sum _ {i=1}^{m} \ln p\left(y _ {i} \mid \boldsymbol{x} _ {i} ; \boldsymbol{w}, b\right)$								(公式3.25)

      - 令每个样本属于其真实标记的概率越大越好

        - 令$\beta=(w;b),\hat{x}=(x;1)$，则$w^Tx+b=\beta^T\hat{x}$

        - $p _ 1(\hat{x};\beta)=p(y=1 \vert \hat{x};\beta)$

        - $p _ 0(\hat{x};\beta)=p(y=0 \vert \hat{x};\beta)=1-p(y=1 \vert \hat{x};\beta)$

        - 公式3.25的似然项为

          $$\begin{aligned}p(y _ i \vert x _ i;w,b)=&p(y=1 \vert \hat{x};\beta)+p(y=0 \vert \hat{x};\beta)  \\  p(y _ i \vert x _ i;w,b)=&y _ ip _ 1(\hat{x _ i};\beta)+(1-y _ i)p _ 0(\hat{x _ i};\beta)\end{aligned}$$

        - 将上式带入3.25
    
          - 注意
        
        $$\begin{aligned}p _ 1(\hat{x _ i};\beta)=&\frac{e^{\beta^T\hat{x _ i}}}{1+e^{\beta^T\hat{x _ i}}}  \\  p _ 0 (\hat{x _ i};\beta)=&\frac{1}{1+e^{\beta^T\hat{x _ i}}}\end{aligned}$$
    
  - 根据公式3.23和3.24：最大化似然函数等价于最小化似然函数的相反数（下式中倒数第二个推导出的公式为极大似然估计的似然函数）
              
      $$\begin{aligned}\ell(\beta)=& \sum _ {i=1}^mln(y _ ip _ i(\hat{x _ i};\beta)+(1-y _ i)p _ 0(\hat{x _ i};\beta))  \\  =&\sum _ {i=1}^mln(\frac{y _ ie^{\beta^T\hat{x _ i}}+1-y _ i}{1+e^{\beta^T\hat{x _ i}}})  \\  =&\sum _ {i=1}^{m}(ln(y _ ie^{\beta^T\hat{x _ i}}+1-y _ i)-ln(1+e^{\beta^T\hat{x _ i}}))  \\  \ell(\beta)=&\left\{\begin{array}{cl}
            \sum _ {i=1}^{m}(-ln(1+e^{\beta^T\hat{x _ i}})), & y _ i=0  \\ 
        \sum _ {i=1}^{m}(\beta^T\hat{x _ i}-ln(1+e^{\beta^T\hat{x _ i}})), & y _ i=1
            \end{array}\right.  \\  \ell(\beta)=&\sum _ {i=1}^{m}\left(y _ {i} \boldsymbol{\beta}^{\mathrm{T}} \hat{\boldsymbol{x}} _ {i}-\ln \left(1+e^{\boldsymbol{\beta}^{\mathrm{T}} \hat{\boldsymbol{x}} _ {i}}\right)\right)  \\  \ell(\beta)=&\sum _ {i=1}^{m}\left(-y _ {i} \boldsymbol{\beta}^{\mathrm{T}} \hat{\boldsymbol{x}} _ {i}+\ln \left(1+e^{\boldsymbol{\beta}^{\mathrm{T}} \hat{\boldsymbol{x}} _ {i}}\right)\right)\end{aligned} $$
            

  - 上式是关于$\beta$的高阶可导连续凸函数，根据凸优化，通过**梯度下降、牛顿法**都可求其最优解

      - $\boldsymbol{\beta}^{*}=\underset{\boldsymbol{\beta}}{\arg \min } \ell(\boldsymbol{\beta})$

      - 做求导


      ​        

## 线性判别分析

- 线性判别分析（Linear Discriminant Analysis，简称LDA，亦称Fisher判别分析）

- 思想

  - 给定训练样例集，设法将样例投影到一条直线上，使得同类样例的投影点尽可能接近、异类样例的投影点尽可能远离
  - 在对新样本进行分类时，将其投影到同样的这条直线上，再根据投影点的位置来确定新样本的类别
  - LDA与Fisher判别分析稍有胡同，前者假设了各类样本的协方差矩阵相同并且满秩

- 二维示意图

  <img src="/T28.png" style="zoom:50%;" />

- 公式推导

  - 给定数据集$D=\{(x _ i,y _ i)\}^m _ {i=1}$

  - 令$X _ i,\mu _ i,\sum _ i$分别表示第$i\in\{0,1\}$类示例的集合、均值向量、协方差矩阵

  - 若将数据投影到直线$w$上，则两类样本的中心在直线上的投影分别为$w^T\mu _ 0$和$w^T\mu _ 1$

  - 若将所有样本点都投影到直线上，则两类样本的协方差分别为$w^T\sum _ 0w$和$w^T\sum _ 1w$

    - 知识补充：

      - **协方差**：协方差是对两个随机变量联合分布线性相关程度的一种度量。两个随机变量越线性相关，协方差越大，完全线性无关，协方差为零。

      <img src="/T29.png" style="zoom:30%;" />

      - **样本协方差**：对于多维随机变量$Q(x1,x2,x3,…,xn)$，样本集合为$$x _ {ij}=[x _ {1j},x _ {2j},…,x _ {nj}] (j=1,2,…,m) $$，m为样本数量，在$ a,b（a,b=1,2…n）$两个维度内:

        $\operatorname{cov}\left(\mathrm{x} _ {\mathrm{a}}, \mathrm{x} _ {\mathrm{b}}\right)=\frac{\sum _ {j=1}^{m}\left(x _ {a j}-\bar{x} _ {a}\right)\left(x _ {b j}-\bar{x} _ {b}\right)}{m-1}$

      - **协方差矩阵**：对于多维随机变量$Q(x1,x2,x3,…,xn)$我们需要对任意两个变量$(xi,xj)$求线性关系，即需要对任意两个变量求协方差矩阵$$ \operatorname{cov}\left(x _ {i}, x _ {j}\right)=\left[\begin{array}{ccccc}
        \operatorname{cov}\left(x _ {1}, x _ {1}\right) & \operatorname{cov}\left(x _ {1}, x _ {2}\right) & \operatorname{cov}\left(x _ {1}, x _ {3}\right) & \cdots & \operatorname{cov}\left(x _ {1}, x _ {n}\right)  \\ 
        \operatorname{cov}\left(x _ {2}, x _ {1}\right) & \operatorname{cov}\left(x _ {2}, x _ {2}\right) & \operatorname{cov}\left(x _ {2}, x _ {3}\right) & \cdots & \operatorname{cov}\left(x _ {2}, x _ {n}\right)  \\ 
        \operatorname{cov}\left(x _ {3}, x _ {1}\right) & \operatorname{cov}\left(x _ {3}, x _ {2}\right) & \operatorname{cov}\left(x _ {3}, x _ {3}\right) & \cdots & \operatorname{cov}\left(x _ {3}, x _ {n}\right)  \\ 
        \vdots & \vdots & \vdots & \ddots & \vdots  \\ 
        \operatorname{cov}\left(x _ {m}, x _ {1}\right) & \operatorname{cov}\left(x _ {m}, x _ {2}\right) & \operatorname{cov}\left(x _ {m}, x _ {3}\right) & \cdots & \operatorname{cov}\left(x _ {m}, x _ {n}\right)
        \end{array}\right] $$

  - 欲最大化

    - 欲使同类样例的投影点尽可能接近，可以让同类样例点的协方差$w^T\sum _ 0w+w^T\sum _ 1w$尽可能小

    - 欲使异类样例的投影点尽可能远离，可以让类中心之间的距离尽可能大，即$\|w^T\mu _ 0-w^T\mu _ 1\| _ 2^2$

    - 同时考虑二者，则可以得到欲最大化的目标
      
      $$ \begin{aligned}
      J &=\frac{\left\|\boldsymbol{w}^{\mathrm{T}} \boldsymbol{\mu}_{0}-\boldsymbol{w}^{\mathrm{T}} \boldsymbol{\mu}_{1}\right\|_{2}^{2}}{\boldsymbol{w}^{\mathrm{T}} \boldsymbol{\Sigma}_{0} \boldsymbol{w}+\boldsymbol{w}^{\mathrm{T}} \boldsymbol{\Sigma}_{1} \boldsymbol{w}} \\
  &=\frac{\boldsymbol{w}^{\mathrm{T}}\left(\boldsymbol{\mu}_{0}-\boldsymbol{\mu}_{1}\right)\left(\boldsymbol{\mu}_{0}-\boldsymbol{\mu}_{1}\right)^{\mathrm{T}} \boldsymbol{w}}{\boldsymbol{w}^{\mathrm{T}}\left(\boldsymbol{\Sigma}_{0}+\boldsymbol{\Sigma}_{1}\right) \boldsymbol{w}}
      \end{aligned} $$
    
    - 定义**类内散度矩阵**

      $$ S _ w=\sum _ 0+\sum _ 1 \\ \begin{aligned}=\sum _ {\boldsymbol{x} \in X _ {0}}\left(\boldsymbol{x}-\boldsymbol{\mu} _ {0}\right)\left(\boldsymbol{x}-\boldsymbol{\mu} _ {0}\right)^{\mathrm{T}}+\sum _ {\boldsymbol{x} \in X _ {1}}\left(\boldsymbol{x}-\boldsymbol{\mu} _ {1}\right)\left(\boldsymbol{x}-\boldsymbol{\mu} _ {1}\right)^{\mathrm{T}}\end{aligned}$$

    - 定义**类间散度矩阵**

      $S _ b=(\mu _ 0-\mu _ 1)(\mu _ 0-\mu _ 1)^T$

    - 则J可重写为

      $J=\frac{w^TS _ bw}{w^TS _ ww}$

    - LDA欲最大化的目标：$S _ b$和$S _ w$的**广义瑞利商（generalized Rayleigh quotient）**
  
  - 确定w
  
  - 由于J的解与w的长度无关，只与其方向有关，不失一般性，令$w^TS _ ww=1$，则J的重写式等价于$$\begin{array}{ll}
      \min  _ {\boldsymbol{w}} & -\boldsymbol{w}^{\mathrm{T}} \mathbf{S} _ {b} \boldsymbol{w}  \\ 
    \text { s.t. } & \boldsymbol{w}^{\mathrm{T}} \mathbf{S} _ {u} \boldsymbol{w}=1
      \end{array}$$

    - 由拉格朗日乘子法

      - 补充说明：拉格朗日乘数法是一种寻找变量受一个或多个条件所限制的多元函数的极值的方法。这种方法将一个有n 个变量与k 个约束条件的最优化问题转换为一个有n + k个变量的方程组的极值问题，其变量不受任何约束。这种方法引入了一种新的标量未知数，即拉格朗日乘数：约束方程的梯度的线性组合里每个向量的系数，从而找到能让设出的隐函数的微分为零的未知数的值。

        - 设给定二元函数z=ƒ(x,y)和附加条件φ(x,y)=0，为寻找z=ƒ(x,y)在附加条件下的极值点，先做拉格朗日函数 

          <img src="/T30.png" style="zoom:50%;" /> 

          (其中λ为参数)

      - 推导：

        $L(w,\lambda)=-w^TS _ bw+\lambda(w^TS _ ww-1)$

        对$w$求偏导

        $\frac{\partial L(w,\lambda)}{\partial w}=-(S _ b+S _ b^T)w+\lambda (S _ w+S _ w^T)w$

        因为$S _ b=S _ b^T,S _ w=S _ w^T$

        $\frac{\partial L(w,\lambda)}{\partial w}=-2S _ bw+2\lambda S _ ww=0$

        $S _ bw=\lambda S _ ww$

        由于拉格朗乘子具体取值多少无所谓，我们像求解的只有$w$，所以我们可以任意设定$\lambda$来配合我们求解$w$

        如果我们令$\lambda=(\mu _ 0-\mu _ 1)^Tw$，那么$S _ bw=\lambda(\mu _ 0-\mu _ 1)$

        则$w=S _ w^{-1}(\mu _ 0-\mu _ 1)$
  
      - 考虑到数值解的稳定性，实际上是对$S _ w$做奇异值分解
  
        - 补充知识：**奇异值分解**
          - 假设M是一个m×n阶矩阵，其中的元素全部属于域 K，也就是实数域或复数域。如此则存在一个分解使得$M=U\sum V^T$
        - 其中U是m×m阶酉矩阵（$U^TU=UU^T=I _ A/U^T=U^{-1}$）；$\sum$是半正定m×n阶对角矩阵；$V^*$，(V的共轭转置)，是n×n阶酉矩阵。这样的分解就称作M的奇异值分解。Σ对角线上的元素Σi，其中Σi即为M的奇异值。
        - $S _ w=U\sum V^T$
      - $\sum$是一个实对角矩阵，其对角线上的元素是$S _ w$的奇异值
        
    - $S _ w^{-1}=V\sum^{-1}U^T$
      
- LDA贝叶斯决策理论阐释
  
  - 可证明，当两类数据同先验、满足高斯分布且协方差相等时，LDA可达到最优分类
  
- LDA推广到多分类任务重
  
    - 假定存在N个类，且第i类示例数为$m _ i$
  
  - **全局散度矩阵**
  
    - $S _ t=S _ b+S _ w  =\sum^{m} _ {i=1}(x _ i-\mu)(x _ i-\mu)^T$
  
    - 其中$\mu$是所有示例的均值向量，将类内散度矩阵$S _ w$重定义为每个类别的散度矩阵之和$$ \begin{aligned}S _ w=\sum^N _ {i=1}S _ {w _ i}\end{aligned}$$
  
      - 所以说$$ \begin{aligned}S _ {w _ i}=\sum _ {\boldsymbol{x} \in X _ {i}}\left(\boldsymbol{x}-\boldsymbol{\mu} _ {i}\right)\left(\boldsymbol{x}-\boldsymbol{\mu} _ {i}\right)^T\end{aligned}$$
  
      $$ \begin{aligned}S _ b=&S _ t-S _ w  \\  =&\sum^{m} _ {i=1}(x _ i-\mu)(x _ i-\mu)^T-\sum^N _ {i=1}\sum _ {x \in X _ {i}}(x-\mu _ {i})(x-\mu _ {i})^T               \\ =&\sum^N _ {i=1}(\sum _ {x \in X _ {i}}((x-\mu)(x-\mu)^T-(x-\mu _ {i})(x-\mu _ {i})^T))  \\ =&\sum^N _ {i=1}(\sum _ {x \in X _ {i}}((x-\mu)(x^T-\mu^T)-(x-\mu _ {i})(x^T-\mu _ {i}^T)))                           \\ =& \sum^N _ {i=1}(\sum _ {x \in X _ {i}}(-\mu x^T-x\mu^T +\mu\mu^T+\mu _ i^Tx+\mu _ ix^T-\mu _ i\mu _ i^T))                          \\ =& \sum^N _ {i=1}(-m _ i\mu \mu _ i^T-m _ i\mu _ i\mu^T +m _ i\mu\mu^T+m _ i\mu _ i^T\mu _ i)      \\ =& \sum^N _ {i=1}m _ i(\mu _ i\mu _ i^T-\mu \mu _ i^T-\mu _ i\mu^T +\mu\mu^T) \\ =&\sum^N _ {i=1}m _ i(\mu _ i-\mu)(\mu _ i-\mu)^T\end{aligned}$$
  
    - 多分类LDA可以有多种实现方法
    
      - 采用优化目标$$\max  _ {\mathbf{W}} \frac{\operatorname{tr}\left(\mathbf{W}^{\mathrm{T}} \mathbf{S} _ {b} \mathbf{W}\right)}{\operatorname{tr}\left(\mathbf{W}^{\mathrm{T}} \mathbf{S} _ {w} \mathbf{W}\right)}$$
    
        - $\mathbf{W}=(w _ 1,w _ 2...w _ i,...w _ {N-1}) \in \mathbb{R}^{d \times(N-1)}$
        
        - $tr(·)$表示矩阵的迹，一个n×n矩阵A的主对角线（从左上方至右下方的对角线）上各个元素的总和。
          
          $$ \left\{ \begin{aligned}
          \operatorname{tr}\left(\mathbf{W}^{\mathrm{T}} \mathbf{S} _ {b} \mathbf{W}\right) &=\sum_{i=1}^{N-1} \boldsymbol{w} _ {i}^{\mathrm{T}} \mathbf{S} _ {b} \boldsymbol{w} _ {i}  \\ 
          \operatorname{tr}\left(\mathbf{W}^{\mathrm{T}} \mathbf{S} _ {w} \mathbf{W}\right) &=\sum _ {i=1}^{N-1} \boldsymbol{w} _ {i}^{\mathrm{T}} \mathbf{S} _ {w} \boldsymbol{w}_{i}
          \end{aligned} \right. $$
          
        - 可以变形为$\max  _ {\mathbf{W}} \frac{\sum _ {i=1}^{N-1} \boldsymbol{w} _ {i}^{\mathrm{T}} \mathbf{S} _ {b} \boldsymbol{w} _ {i}}{\sum _ {i=1}^{N-1} \boldsymbol{w} _ {i}^{\mathrm{T}} \mathbf{S} _ {w} \boldsymbol{w} _ {i}}$
        
        - 为$J=\frac{w^TS _ bw}{w^TS _ ww}$的推广式
        
        - 求解：$S _ bW=\lambda S _ w W$
  
- 若将W视为一个投影矩阵
      
  - 则多分类$LDA$将样本投影到N-1维空间，N-1通常远小于数据原有的属性数
        - 可以通过这个投影来减小样本点的维数，且投影过程中使用了类别信息
            - LDA通常也被视为一种经典的监督降维技术

## 多分类学习

- 有些二分类学习方法可直接推广到多分类

- 二分类学习器

  - 考虑N个类别$C _ 1,C _ 2,...,C _ N$
  - 多分类学习器的基本思路是**拆解法**
    - 将多分类任务拆为若干个二分类任务求解
    - 先对问题进行拆分
    - 为拆出的每个二分类任务训练一个分类器
    - 测试时：对这些分类器的预测结果进行集成
    - 重点为**拆分和集成**

- 拆分策略

  - **“一对一”（OvO）**

    - 给定数据集$D=\{(x _ 1,y _ 1),(x _ 2,y _ 2),...,(x _ m,y _ m)\},y _ i \in \{C _ 1,C _ 2,...,C _ N\}$
    - 将这N个类别两两配对，从而产生N(N-1)/2个二分类任务
    - 例如OvO为区别$C _ i,C _ j$训练一个分类器
      - 该分类器把D中的$C _ i$类样例作为正例
      - 把D中的$C _ j$类样例作为反例
    - 测试阶段，新样本将同时提交给所有分类器，所以获得N(N-1)/2个分类结果
    - 被预测得最多的类别作为最终分类结果

  - **“一对其余”（OvR）**

    - 只训练N个分类器

    - 训练

      - 每次将一个类的样例作为正例
      - 所有其他类的样例作为反例

    - 测试时若仅有一个分类器预测为正类，对应的类别标记为最终分类结果

      <img src="/T31.png" style="zoom:33%;" />

    - 比起OvO的优缺点

      - 优点：存储开销和测试时间开销更小
      - 缺点：OvR的每个分类器均适用全部训练样例，而OvO仅需要用到两个类的样例
        - 在类别很多时，OvO的训练时间开销通常比OvR更小

  - **“多对多”（MvM）**

    - 每次将若干个类作为正类，若干个其他类作为反类

    - OvO和OvR是MvM的特例

    - MvM的正反类构造需要有特殊设计

      - 最常用的MvM技术**纠错输出码 （Error Correcting Output Codes，简称ECOC）**

        - 编码

          - 对N个类做M次划分，每次划分将一部分类别划为正类，一部分类划为反类，从而形成一个二分类训练集
          - 一共产生M个训练集，可以训练出M个分类器

        - 解码

          - M个分类器分别对测试样本进行预测，这些预测标记组成一个编码
          - 将这个预测编码与每个类别各自的编码进行比较
          - 最终预测结果：距离最小的类别

        - 类别划分：**”编码矩阵（coding matrix）**

          - 二元码

            - 将每个类别分别指定为正类和反类
            - 如下图，分类器$f _ 2$将$C _ 1$类和$C _ 3$类的样例作为正例，$C _ 2$类和$C _ 4$类的样例作为反例

          - 三元码

            - 在正、反类之外，还可以指定“停用类”
            - 如下图，分类器$f _ 6$将$C _ 1$类和$C _ 3$类的样例作为正例，$C _ 2$类的样例作为反例

            <img src="/T32.png" style="zoom:50%;" />

          - 在解码阶段

            - 各分类器的预测结果联合起来形成了测试示例的编码
            - 将距离最小的编码所对应的类别作为预测结果，如上图二元码结果为$C _ 3$

        - 修正

          - 在测试阶段，EOOC编码对分类器的错误有一定的容忍和修正能力
          - 如上图a中，如果$f _ 2$出错导致错误编码（-1，-1，+1，-1，+1），但结果仍然正确
          - 同一个学习任务，ECOC编码越长
            - 纠错能力越强
            - 所需训练的分类器越多，计算存储开销增大
            - 对有限类别数，可能的组合数目是有限的，码长超过一定范围后就失去了意义
          - 同等长度的编码
            - 任意两个类别之间的编码距离越远，纠错能力越强
            - NP难问题
              - 码长较小时可以根据上一条的原则计算出理论最优编码
              - 码长稍大就难以有效确定最优编码（NP难问题）
              - 但通常我们不需要获得理论最优编码

## 类别不平衡问题

- 引入

  - 前面的分类学习方法共同的基本假设：
    - 不同类别的训练样例数目相当
    - 如果不同类别的训练样例数目差别很大，则会造成困扰
      - 如998个反例，正例只有2个，学习方法之返回一个永远将新样本预测为返例的学习器就能达到99.8%的精度

- 类别不平衡

  - 分类任务中不同类别的训练样例数目差别很大的情况，本节假定**正类样例较少，反类样例较多**
  - 发生情况：
    - 通过拆分法解决多分类问题时，即使原始问题不同类别的样例数目相当
      - 在使用OvR、MvM策略后
        - 虽然每个类进行了相同的处理，其拆解出的二分类任务中类别不平衡的影响会相互抵消
        - 但也仍然可能出现类别不平衡

- **再缩放（rescaling）**

  - 类别不平衡学习的一个基本策略

  - 线性分类器的角度

    - 用预测出的y值与一个阈值进行比较
    - 当训练集中正、反例数目相同
      - 例如阈值为0.5，y>0.5时判为正例，否则为反例
      - y表达了正例的可能性，$\frac{y}{1-y}$表达了正例可能性与反例可能性之比值
      - 阈值0.5表明分类器认为真实正、反例可能性相同
      - 若$\frac{y}{1-y}>1$时，预测为正例
    - 当训练集中正、反例数目不同，$m^+、m^-$表示正、反例数目
      - 观测几率是$\frac{m^+}{m^-}$
      - 假设训练集是真实样本总体的**无偏采样**（真实样本总体的类别比例在训练集中得以保持），观测几率=真实几率
      - 若$\frac{y}{1-y}>\frac{m^+}{m^-}$，预测为正例
        - 将上式转换$\frac{y'}{1-y'}=\frac{y}{1-y}\times \frac{m^-}{m^+}$ => $\frac{y'}{1-y'}>1$
        - 这就是再缩放

  - 如何做

    - 训练集是真实样本总体的无偏采样这个假设往往**不成立**

    - 做法1：直接对训练集的反类样例做**”欠采样“（undersampling）**（又称下采样）

      - 去除一些反例使得正、反例数目接近
      - 可能会丢弃一些重要信息
      - **EasyEnsemble**利用集成学习，将反例划分为若干个集合供不同学习器使用
        - 全局来看没有丢失重要信息

    - 做法2：对训练集里的正类样例做**"过采样"（oversampling）**（又称“上采样”）

      - 增加一些正例使得正、反例数目接近，然后再进行学习
      - 不能简单的对初始正例样本进行重复采样，否则会导致过拟合
      - **SMOTE**通过对训练集里的正例进行插值来产生额外的正例

    - 做法3：直接基于原始训练集进行学习

      - **“阈值移动”**：在进行预测时，将$\frac{y'}{1-y'}=\frac{y}{1-y}\times \frac{m^-}{m^+}$嵌入到其决策过程中

    - 比较

      | 方法     | 优点                   | 缺点                   |
      | -------- | ---------------------- | ---------------------- |
      | 欠采样法 | 时间开销小（丢弃反例） | 丢失重要信息           |
      | 过采样法 |                        | 时间开销大（增加正例） |
