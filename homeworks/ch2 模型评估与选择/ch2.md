## 部分习题解释

- 2.1

  - 需要1000个样本，700个训练集，300个测试集
  - 500个正例和500个反例平均分配
  - 700训练集：350正/350反，300测试集：150正/150反
  - $C _ {500}^{350}C _ {500}^{350}$

- 2.2 

  - 10折交叉验证
    - 分成10个大小相似的互斥子集（5个一组）
    - 9个子集为训练集，1个测试集
    - 有10种选择求均值
    - 正反例数目相同
    - 预测判断为正反例的概率相同
    - 错误率50%
  - 留一法
    - 每一个都是一个子集
    - 99个子集为训练集，1个测试集
    - 有100种选择求均值
    - 可能结果：正/反
    - 正：预测错误，反：预测错误 错误率100%

- 2.3

  - $F 1=\frac{2 \times P \times R}{P+R}$
  - 事实上在P同的情况下，R越大$F1$越大，反之亦然
  - 而P越大，R越小
  - P-R图通常是非单调，不平滑的，在很多局部有上下波动
  - A和B的F1值相同和BEP值二者没有绝对相关。
  - BEP点前后很可能同时出现更大的P和更大的R，因此现实中BEP并不实用。

- 2.4

  - 公式
    - $\mathrm{TPR}=\frac{T P}{T P+F N}$
    - $\mathrm{FPR}=\frac{F P}{T N+F P}$
    - $P=\frac{T P}{T P+F P}$
    - $R=\frac{T P}{T P+F N}$
    - $TPR=R$    查全率=真正例率

- 2.5

  - 证明$\mathrm{AUC}\approx1-\ell _ {\text {rank}}$

  - ROC曲线如下（已经乘$m^+m^-$）

    <img src="/assets/MLpics/T4.png" style="zoom:50%;" />

  - 以上图为例，一共有4个正例4个反例$m^+=m^-=4$

  - 理论向证明

    - 暂时不考虑$f(x^{+})=f(x^{-})$，$AUC=\sum _ {i=1}^{m-1}\left(x _ {i+1}-x _ {i}\right)y _ {i}$
      - 乘$m^+m^-$（如上图），可发现$x _ {i+1}-x _ {i}=\{0,1\}$
      - 当$x _ {i+1}-x _ {i}=1$时，$(x _ {i+1},y _ {i+1})$是一个反例样本，排在它前面的有$y _ {i}$个正例
      - 也就是说$AUC=\frac{1}{m^{+} m^{-}} \sum _ {x^{+} \in D^{+}} \sum _ {x^{-} \in D^{-}}(\mathbb{I}(f(x^{+})>f(x^{-}))$
      - 此时$\ell _ {r a n k}=\frac{1}{m^{+} m^{-}} \sum _ {x^{+} \in D^{+}} \sum _ {x^{-} \in D^{-}}\left(\mathbb{I}\left(f\left(x^{+}\right)<f\left(x^{-}\right)\right)\right)$
      - $\mathrm{AUC}=1-\ell _ {\text {rank}}$
    - 考虑$f(x^{+})=f(x^{-})$
      - 正例预测值和反例预测值都相同，如图中的（x2，y3）和（x3，y4）
      - 在AUC面对这种情况时，会多余的加上一个三角形，需要减去
      - 对于$\ell _ {r a n k}$来说就是加上这个三角（1/2）

  - 数据向证明

    - $\ell _ {r a n k}=\frac{1}{m^{+} m^{-}} \sum _ {x^{+} \in D^{+}} \sum _ {x^{-} \in D^{-}}\left(\mathbb{I}\left(f\left(x^{+}\right)<f\left(x^{-}\right)\right)+\frac{1}{2} \mathbb{I}\left(f\left(x^{+}\right)=f\left(x^{-}\right)\right)\right)$
      - $f(x^{+})<f(x^{-})$
        - （0，y2）点前有1个反例样本
        - （x2，y3）点前有2.5个反例样本
        - （x4，y5）点前有3个反例样本
      - $f(x^{+})=f(x^{-})$
        - （x2，y3）和（x3，y4）预测值相等，记0.5
    - 由上式可得$\ell _ {r a n k}=$5.5/16
    - $\mathrm{AUC}=\frac{1}{2} \sum _ {i=1}^{m-1}\left(x _ {i+1}-x _ {i}\right) \cdot\left(y _ {i}+y _ {i+1}\right)$（总面积）
    - 由上式可得$AUC=(1+2+1/2+3+4)/16=10.5/16$
    - 所以$\mathrm{AUC}=1-\ell _ {\text {rank}}$

- 2.6

  - $\mathrm{TPR}=\frac{T P}{T P+F N}$，针对$D^+$情况

  - $\mathrm{FPR}=\frac{F P}{T N+F P}$，针对$D^-$情况

    所以

    $$ \begin{aligned}
    E(f ; D ; \cos t)&= \frac{1}{m}(\sum _ {\boldsymbol{x} _ {i} \in D^{+}} \mathbb{I}\left(f\left(\boldsymbol{x} _ {i}\right) \neq y _ {i}) \times \operatorname{cost} _ {01}
    +\sum _ {\boldsymbol{x} _ {i} \in D^{-}} \mathbb{I}\left(f\left(\boldsymbol{x} _ {i}\right) \neq y _ {i}\right) \times \operatorname{cost} _ {10}\right)
     \\ &= \frac{1}{m}( \vert D^+ \vert (1-TPR) cost _ {01}+FPR \vert D^- \vert cost _ {10}) \\ \end{aligned} $$

- 2.10

  - 一个为卡方检验：主要针对具有相同频数的数据组的一致性（在2.34中用于比较不同算法之间的差异）
  - 一个为基于卡方检验的F检验：主要针对两组数值差异性（在2.35中基于卡方检验进行数据集之间的差异性）
  - F检验比起卡方检验更好的是考虑到了不同数据集所带来的影响（卡方检验直接xN）
