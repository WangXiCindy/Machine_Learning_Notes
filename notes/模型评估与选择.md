# 模型评估与选择

## 经验误差与过拟合

- 概念
  - 错误率：分类错误的样本数占样本总数的比例
  - 精度：1-错误率
  - 误差：学习器的实际预测输出与样本的真实输出之间的差异
    - **训练/经验误差**：学习器的实际预测输出与样本的真实输出
    - **泛化误差**：在新样本上的误差
      - 如何获得更好的泛化误差：尽可能在训练样本中学出适用于所有潜在样本的“普遍规律”
      - 过拟合：把训练样本学习“太好”，把训练样本自身的一些特点当作了所有潜在样本都会具有的一般性质。原因：学习能力过于强大。无法避免，只能缓解。
      - 欠拟合：对训练样本的一般性质尚未学好。原因：学习能力低下。方法：决策树中扩展分支，神经网络中增加训练轮数等。

## 评估方法

- 测试集：测试学习器对新样本的判别能力
  - 概念
    - **测试误差**：泛化误差的近似
    - 尽可能与训练集互斥（测试样本尽量不在训练集中出现）
  - 常见得到测试集做法
    - **留出法**：将数据集D划分为两个互斥的集合，一个集合作为训练集S，另一个作为测试集T
      - 训练/测试集的划分要尽可能保持数据分布的一致性，避免因数据划分过程中引入额外的偏差而对最终结果产生影响
      - 分层采样：保留类别比例的采样方式
      - 在给定样本比例后，仍然存在多种划分方式对数据集进行分割，比如是前350个放入训练集还是后350个
      - 所以使用留出法时，一般要采用若干次随机划分，重复进行实验评估，最后取平均值
    - **交叉验证法**（k折交叉验证）
      - 将数据集D划分为k个大小相似的互斥子集，也存在多种划分方式，一般要随机使用不同的划分重复p次，最终评估结果是这p次k折交叉验证结果的均值
      - 每次用k-1个子集的并集作为训练集，余下的子集作为测试集
      - 可以进行k次训练和测试，最后返回这k个测试结果的均值
      - k最常用的取值是10/5/20
      - 特例：**留一法**，D中包含m个样本，k=m
        - m个样本只有唯一的方式划分为m个子集
        - 优点：与用D训练出来的模型很相似，往往被认为较为准确
        - 缺点：
          - 在数据集比较大时，训练m个模型的计算开销可能是难以忍受的
          - 没有免费的午餐仍然适用
      - **自助法**：为了减少训练样本规模不同造成的影响
        - 以自助采样法为基础
        - 每次从D中随机挑选一个样本，将其拷贝放入D‘，然后再将该样本放回D，使得下一次采样时仍有可能被采到
        - 样本在m次采样中始终不被采到的概率取极限$\lim  _ {m \mapsto \infty}\left(1-\frac{1}{m}\right)^{m} \mapsto \frac{1}{e} \approx 0.368$
        - 我们说D中约有36.8%的样本未出现在D‘中，D\D‘即可作为测试集
        - 这样的测试结果，成为：包外估计“
        - 优点：
          - 在数据集较小，难以有效划分训练/测试集时很有用
          - 能从初始数据集中产生多个不同的训练集，对集成学习等有较大好处
        - 缺点：改变了初始数据集分布，会引入估计偏差，所以数据量足够时，留出法和交叉验证更为常用
      - 调参与最终模型
        - 学习算法的很多参数是在实数范围内取值，因此，对每种参数配置都训练出模型来是不可行的
        - 常用做法：选定范围和变化步长
          - 非最佳，但是在计算开销和性能估计之间折中的结果
        - 最终模型！！！：一开始只适用一部分数据训练模型，所以在模型选择完成之后，需要将D重新训练模型

## 性能度量

- 概念：衡量模型泛化能力的评价标准

  - **均方误差**：回归任务最常用

    - $E(f ; D)=\frac{1}{m} \sum _ {i=1}^{m}\left(f\left(\boldsymbol{x} _ {i}\right)-y _ {i}\right)^{2}$

    - 对于数据分布$\mathcal{D}$和概率密度函数$p(·)$，均方误差

      $E(f ; \mathcal{D})=\int _ {\boldsymbol{x} \sim \mathcal{D}}(f(\boldsymbol{x})-y)^{2} p(\boldsymbol{x}) \mathrm{d} \boldsymbol{x}$

    - 几何意义：对应了欧式距离

      - 基于其最小化来进行模型求解的方法为最小二乘法
      - 最小二乘法就是找到一条直线，使所有样本到直线上的欧式距离和最小

- 错误率与精度

  - 错误率：

    - $E(f ; D)=\frac{1}{m} \sum _ {i=1}^{m} \mathbb{I}\left(f\left(\boldsymbol{x} _ {i}\right) \neq y _ {i}\right)$

    - 对于数据分布$\mathcal{D}$和概率密度函数$p(·)$，

      $E(f ; \mathcal{D})=\int _ {\boldsymbol{x} \sim \mathcal{D}} \mathbb{I}(f(\boldsymbol{x}) \neq y) p(\boldsymbol{x}) \mathrm{d} \boldsymbol{x}$

  - 精度
    
      $ \begin{aligned} \operatorname{acc}(f ; D) =\frac{1}{m} \sum _ {i=1}^{m} \mathbb{I}\left(f\left(\boldsymbol{x} _ {i}\right)=y _ {i}\right) =1-E(f ; D) \end{aligned} $
      
  - 对于数据分布$\mathcal{D}$和概率密度函数$p(·)$，
    
      $\begin{aligned}
      \operatorname{acc}(f ; \mathcal{D})=\int _ {\boldsymbol{x} \sim \mathcal{D}} \mathbb{I}(f(\boldsymbol{x})=y) p(\boldsymbol{x}) \mathrm{d} \boldsymbol{x}
      =1-E(f ; \mathcal{D})
    \end{aligned}$
    
  - 查准率，查全率（两者互相矛盾）与F1

    - **查准率**$P=\frac{T P}{T P+F P}$（在预测结果为正例的情况下预测正确的概率）

    - **查全率**$R=\frac{T P}{T P+F N}$(在真实情况为正例的情况下预测正确的概率)

    - T/F判断是否正确

      P/N预测结果是否是正例

      以下为**混淆矩阵**：

    - | 真实情况\预测结果 | 正例         | 反例         |
      | ----------------- | ------------ | ------------ |
      | 正例              | TP（真正例） | FN（假反例） |
      | 反例              | FP（假正例） | TN（真反例） |

      T/F判断是否正确

      P/N预测结果是否是正例

    - 矛盾度量

      - 查准率高，查全率低
      - 比如：为了高查全率，将所有西瓜都选上，所有的好瓜也都必然被选上了，但这样查准率就低
      - 查准率高，只挑选最有把握的瓜，但可能就会漏掉不少好瓜

    - P-R曲线与**平衡点**

      - 若一个学习器的P-R曲线被另一个学习器的曲线完全“包住”，则后者的性能优与前者（比如A优于C）
      - 如果两个学习器的P-R曲线发生交叉，则具体情况具体分析。此时，可以比较P-R曲线下面积的大小，这表征了学习器在查准率和查全率上取得相对较好的比例

    <img src="/assets/MLpics/T2.png" alt="图片2" style="zoom:60%;" />

    - ​	平衡点（BEP）

      - 综合考虑查准率，查全率的性能度量
      - 查准率=查全率的取值
      - 平衡点越高，学习器性能越好

    - **F1**度量（调和平均）

      - BEP较为简化

      - 与算术平均和几何平均相比，调和平均更重视较小值

      - $F 1=\frac{2 \times P \times R}{P+R}=\frac{2 \times T P}{\text { 样例总数 }+T P-T N}$

      - 表达出对查准率/查全率的不同偏好：F1度量的一般形式（加权调和平均）

        $F _ {\beta}=\frac{\left(1+\beta^{2}\right) \times P \times R}{\left(\beta^{2} \times P\right)+R}$

        - $\beta>0$度量了查全率对查准率的相对重要性
          - $\beta>1$对查全率有更大影响
          - $\beta=1$退化为标准的F1
          - $\beta<1$对查准率有更大影响

    - 多个二分类混淆矩阵综合考察

      - 方法1:先在各混淆矩阵上分别计算出查准率和查全率，记为(P1,R1),(P2,R2),...,(Pn,Rn)，再计算平均值

        - 宏查准率

          $\operatorname{macro}-P=\frac{1}{n} \sum _ {i=1}^{n} P _ {i}$

          - 宏查全率

          $\operatorname{macro}-R=\frac{1}{n} \sum _ {i=1}^{n} R _ {i}$

          - 宏F1

          $\operatorname{macro}-F 1=\frac{2 \times \operatorname{macro}-P \times \operatorname{macro}-R}{\operatorname{macro}-P+\operatorname{macro}-R}$

      - 对各混淆矩阵的对应元素进行平均，得到TP，FP，
        TN，FN的平均值，再计算出

        - 微查准率

        $\operatorname{micro}-P=\frac{\overline{T P}}{\overline{T P}+\overline{F P}}$

        - 微查全率

        $\operatorname{micro}-R=\frac{\overline{T P}}{\overline{T P}+\overline{F N}}$

        - 微F1

        $\operatorname{micro}-F 1=\frac{2 \times \operatorname{micro}-P \times \operatorname{micro}-R}{\operatorname{micro}-P+\operatorname{micro}-R}$

- **ROC与AUC**

  - 很多学习器是为样本产生一个实值或概率预测，然后将这个预测值与一个分类**阈值（threshold）**比较，大于阈值为正类，否则为反。

  - 这个实值预测结果的好坏，直接决定了学习器的泛化能力。根据这个实值或概率预测结果，将测试样本进行排序。

  - 分类过程=以某个截断点将样本分为两部分，前一部分判定为正例，后部分为反例

    - 更重视查准率：选择排序靠前的位置进行截断
    - 更重视查全率：选择靠后的位置进行截断

  - ROC：研究学习器泛化性能的工具

    - 根据预测结果对样例进行排序，按此排序逐个把样本作为正例进行预测，每次计算出两个量的值，分别以他们为横纵坐标作图

      - 真正确率（纵轴）

      $\mathrm{TPR}=\frac{T P}{T P+F N}$

      - 假正例率（横轴）

      $\mathrm{FPR}=\frac{F P}{T N+F P}$

      - <img src="/assets/MLpics/T3.png" style="zoom:50%;" />
      - 曲线说明
        - 有限个坐标对时，无法产生光滑曲线
        - 先把分类阈值设为最大，所有样例均为反例，此时坐标为（0，0）
        - 依次将每个样例划分为正例
        - 在统计预测结果时，预测值=分类阈值的样本也算作预测为正例

    - AUC：如果两个ROC曲线发生交叉，则难以一般性断言两者谁优谁劣，此时比较ROC曲线的面积=AUC

      $\mathrm{AUC}=\frac{1}{2} \sum _ {i=1}^{m-1}\left(x _ {i+1}-x _ {i}\right) \cdot\left(y _ {i}+y _ {i+1}\right)$

      - 🎃书案例：

        假设有7个预测结果（不做详细说明），画出ROC曲线，将增加面积看成梯形来计算，（梯形和长方形的面积公式相同）

        **梯形面积**$=\frac{1}{2} \left(x _ {i+1}-x _ {i}\right) \cdot\left(y _ {i}+y _ {i+1}\right)$

        <img src="/assets/MLpics/T4.png" style="zoom:50%;" />

      - 给$m^{+}$个正例和$m^{-}$个反例，令$D^{+}$和$D^{-}$分别表示正、反例集合，则loss为

        $\ell _ {r a n k}=\frac{1}{m^{+} m^{-}} \sum _ {x^{+} \in D^{+}} \sum _ {x^{-} \in D^{-}}\left(\mathbb{I}\left(f\left(x^{+}\right)<f\left(x^{-}\right)\right)+\frac{1}{2} \mathbb{I}\left(f\left(x^{+}\right)=f\left(x^{-}\right)\right)\right)$

      - 如果我们对上面的式子继续做变形，可以得到：
      
      $$\begin{aligned} \ell _ {rank} &=\frac{1}{m^{+} m^{-}} \sum _ {x^{+} \in D^{+}} \sum _ {x^{-} \in D^{-}}\left(\mathbb{I}\left(f\left(x^{+}\right)<f\left(x^{-}\right)\right)+\frac{1}{2} \mathbb{I}\left(f\left(x^{+}\right)=f\left(x^{-}\right)\right)\right)  \\  &=\frac{1}{m^{+} m^{-}} \sum _ {x^{+} \in D^{+}}\left[\sum _ {x^{-} \in D^{-}} \mathbb{I}\left(f\left(x^{+}\right)<f\left(x^{-}\right)\right)+\frac{1}{2} \cdot \sum _ {x^{-} \in D^{-}} \mathbb{I}\left(f\left(x^{+}\right)=f\left(x^{-}\right)\right)\right]  \\ &=\sum _ {x^{+} \in D^{+}}\left[\frac{1}{m^{+}} \cdot \frac{1}{m^{-}} \sum _ {x^{-} \in D^{-}} \mathbb{I}\left(f\left(x^{+}\right)<f\left(x^{-}\right)\right)+\frac{1}{2} \cdot \frac{1}{m^{+}} \cdot \frac{1}{m^{-}} \sum _ {x^{-} \in D^{-}} \mathbb{I}\left(f\left(x^{+}\right)=f\left(x^{-}\right)\right)\right]  \\ &=\sum _ {x^{+} \in D^{+}} \frac{1}{2} \cdot \frac{1}{m^{+}} \cdot\left[\frac{2}{m^{-}} \sum _ {x^{-} \in D^{-}} \mathbb{I}\left(f\left(x^{+}\right)<f\left(x^{-}\right)\right)+\frac{1}{m^{-}} \sum _ {x^{-} \in D^{-}} \mathbb{I}\left(f\left(x^{+}\right)=f\left(x^{-}\right)\right)\right]
        \end{aligned}$$
  
      - 因为新增正例就新增一条蓝色/绿色线段，所以$\sum _ {x^{+} \in D^{+}}$是在遍历所有蓝色和绿色线段

        - 后面那一项是在求绿色线段/蓝色线段与y轴围成的面积

          $\frac{1}{2} \cdot \frac{1}{m^{+}} \cdot\left[\frac{2}{m^{-}} \sum _ {x^{-} \in D^{-}} \mathbb{I}\left(f\left(\boldsymbol{x}^{+}\right)<f\left(\boldsymbol{x}^{-}\right)\right)+\frac{1}{m^{-}} \sum _ {\boldsymbol{x}^{-} \in D^{-}} \mathbb{I}\left(f\left(\boldsymbol{x}^{+}\right)=f\left(\boldsymbol{x}^{-}\right)\right)\right]$

          - $\frac{1}{m^{+}} $为梯形的高

          - 梯形的上底，每增加一个假正例时x坐标就新增一个单位，实则为目前预测值$x^{+}$大的假正例个数乘$\frac{1}{m^{-}}$

            $\frac{1}{m^{-}} \sum _ {\boldsymbol{x}^{-} \in D^{-}} \mathbb{I}\left(f\left(\boldsymbol{x}^{+}\right)<f\left(\boldsymbol{x}^{-}\right)\right)$

          - 梯形的下底，每增加一个假正例时x坐标就新增一个单位，实则为目前预测值$x^{+}$大/等的假正例个数乘$\frac{1}{m^{-}}$

            $\frac{1}{m^{-}}\left(\sum _ {\boldsymbol{x}^{-} \in D^{-}} \mathbb{I}\left(f\left(\boldsymbol{x}^{+}\right)<f\left(\boldsymbol{x}^{-}\right)\right)+\sum _ {\boldsymbol{x}^{-} \in D^{-}} \mathbb{I}\left(f\left(\boldsymbol{x}^{+}\right)=f\left(\boldsymbol{x}^{-}\right)\right)\right)$
    
          - $\mathrm{AUC}\approx1-\ell _ {\text {rank}}$

  - **代价敏感错误率**与**代价曲线**

    - 情况：错误的把患者检测为健康（必须避免发生）/把健康的人检测为患者（增加了一次进一步检查的麻烦）

    - **代价矩阵**$cost _ {ij}$代表将第i类样本预测为第j类样本的代价
  
      | 真实类别\预测类别 | 0           | 1           |
      | ----------------- | ----------- | ----------- |
    | 0                 | 0           | $cost _ {01}$ |
      | 1                 | $cost _ {10}$ | 0           |

    - 之前的性能度量大多隐式的假设了均等代价，并没有考虑不同错误会造成不同的后果

    - 在非均等代价下，我们希望最小化“总体代价”而非“错误次数”
  
    - 将第0类作为正类，第1类作为反类，$D^{+}\text{与}D^{-}$分别代表D的正例子集和反例子集，则代价敏感错误率
      
    $$\begin{aligned}
      E(f ; D ; \cos t)=& \frac{1}{m}\left(\sum _ {\boldsymbol{x} _ {i} \in D^{+}} \mathbb{I}\left(f\left(\boldsymbol{x} _ {i}\right) \neq y _ {i}\right) \times \operatorname{cost} _ {01}
+\sum _ {\boldsymbol{x} _ {i} \in D^{-}} \mathbb{I}\left(f\left(\boldsymbol{x} _ {i}\right) \neq y _ {i}\right) \times \operatorname{cost} _ {10}\right)
      \end{aligned}$$
  
    - 在非均等代价下，ROC曲线不能直接反映出学习器的期望总体代价，但“代价曲线”可达到该目的

      - 横轴：取值为[0,1]的正例概率代价

        $P(+) \cos t=\frac{p \times \cos t _ {01}}{p \times \cos t _ {01}+(1-p) \times \cos t _ {10}}$

      - 纵轴：取值为[0,1]的归一化代价

        $\operatorname{cost} _ {n o r m}=\frac{\operatorname{FNR} \times p \times \cos t _ {01}+\mathrm{FPR} \times(1-p) \times \operatorname{cost} _ {10}}{p \times \operatorname{cost} _ {01}+(1-p) \times \operatorname{cost} _ {10}}$

        FPR是假正例率，FNR=1-TPR是假反例率
  
      - ROC曲线上的每一点对应了代价平面上的一条线段，设ROC曲线上点的坐标为（TPR，FPR），则可相应计算出FNR，然后绘制一条从（0，FPR）到（1，FNR）的线段
    
      - 线段下的面积代表该条件下的期望总体代价

## 比较检验

- **假设检验**

  - 假设：对学习器泛化错误率分布的某种判断或猜想

    - 现实生活中我们并不知道学习器的泛化错误率，只能知道其测试错误率
    - 泛化错误率与测试错误率不一定相同，但二者接近的可能性较大

  - 泛化错误率为$\epsilon$的学习器在一个样本上犯错的概率是$\epsilon$

  - 测试错误率$\hat{\epsilon}$意味着在m个测试样本中恰有$\hat{\epsilon}\times m$被错误分类

    - 假设测试样本是从样本分布中独立采样获得，泛化错误率为$\epsilon$的学习器恰好将其中$m'$个样本误分类的概率$\epsilon^{m^{\prime}}(1-\epsilon)^{m-m^{\prime}}$

    - 恰好将$\hat{\epsilon}\times m$个样本误分类的概率/泛化错误率为$\epsilon$的学习器被测试得测试错误率为$\hat{\epsilon}$的概率$$ P(\hat{\epsilon} ; \epsilon)=\left(\begin{array}{c} m  \\ \hat{\epsilon} \times m  \end{array}\right) \epsilon^{\hat{\epsilon} \times m}(1-\epsilon)^{m-\hat{\epsilon} \times m}$$

    - 给定$\hat{\epsilon}$，则

      ​	<img src="/assets/MLpics/T5.png" style="zoom:50%;" />

      - $\partial P(\hat{\epsilon} ; \epsilon) / \partial \epsilon=0$
      - $P(\hat{\epsilon} ; \epsilon)$在$\hat{\epsilon}=\epsilon$时最大，如图，若$\epsilon=0.3$则10个样本中测得3个被误分类的概率最大
      - $ \vert \hat{\epsilon}-\epsilon \vert $增大时减小
      - 符合二项分布，注意$$\left(\begin{array}{c}
        m  \\ 
        \hat{\epsilon} \times m
        \end{array}\right) $$

    - **二项检验**

      - 对$\epsilon \leq \epsilon _ {0}$进行假设检验（单侧）

      - 求解最大错误率$\bar{\epsilon}$

      - $ 1-\alpha$为概率学习中的置信度，如0.95，0.90

      - 计算事件最小发生频率到最大发生频率的概率和现$\epsilon _ {0}$的情况做对比，从而得到检验结果

      - $\frac{\bar{C}}{m}$为事件最大发生频率$$\bar{C}=\min C \quad \text { s.t. } \sum _ {i=C+1}^{m}\left(\begin{array}{c}
        m  \\ 
        i
        \end{array}\right) p _ {0}^{i}\left(1-p _ {0}\right)^{m-i}<\alpha$$
        $$\frac{\bar{C}}{m}=\min \frac{C}{m} \quad \text { s.t. } \quad \sum _ {i=C+1}^{m}\left(\begin{array}{c}
        m  \\ 
        i
        \end{array}\right) p _ {0}^{i}\left(1-p _ {0}\right)^{m-i}<\alpha$$

        将$\frac{\bar{C}}{m}, \frac{C}{m}, p _ {0}$等价替换为$\bar{\epsilon}, \epsilon, \epsilon _ {0}$
        $$\bar{\epsilon}=\min \epsilon \quad \text { s.t. } \quad \sum _ {i=\epsilon \times m+1}^{m}\left(\begin{array}{c}
        m  \\ 
        i
        \end{array}\right) \epsilon _ {0}^{i}\left(1-\epsilon _ {0}\right)^{m-i}<\alpha$$

      - 若在$\alpha$的显著度下，$\epsilon \leq \epsilon _ {0}$不能被拒绝，那么就能以$1-\alpha$的置信度认为，学习器的泛化错误率不大于$\epsilon _ {0}$

    - **t检验**（大学概率学习内容）

      - 经过多次重复留出法/交叉验证法进行多次训练/测试，会得到多个测试错误率

      - 注意误分类样本数服从二项分布，大量样本之后，根据**中心极限定理**，所以大量二项分布（极限）服从正态

        <img src="/assets/MLpics/T13.png" style="zoom:50%;" />

      - 假设我们得到了k个测试错误率$\hat\epsilon _ {1}, \hat{\epsilon} _ {2}, \ldots, \hat{\epsilon} _ {k}$，则

        - $\mu=\frac{1}{k} \sum _ {i=1}^{k} \hat{\epsilon} _ {i}$
        - $\sigma^{2}=\frac{1}{k-1} \sum _ {i=1}^{k}\left(\hat{\epsilon} _ {i}-\mu\right)^{2}$
        - 则$\tau _ {t}=\frac{\sqrt{k}\left(\mu-\epsilon _ {0}\right)}{\sigma}$

      - 大学概率补充知识：

        <img src="/assets/MLpics/T6.png" style="zoom:50%;" />

        <img src="/assets/MLpics/T8.png" style="zoom:50%;" />

        - t假设检验<img src="/assets/MLpics/T7.png" style="zoom:50%;" />

        - 得到服从自由度为k-1的t分布

          <img src="/assets/MLpics/T9.png" style="zoom:50%;" />

        - 对假设$\mu=\epsilon$和显著度$\alpha$，我们可以计算出当测试错误率均值为$\epsilon _ {0}$时，在$1-\alpha$概率内能观测到的最大错误率

        - 假设检验（两侧）

        - $ \vert \mu-\epsilon _ {0} \vert $在$[t _ {-\alpha/2},t _ {\alpha/2}]$内，为接受域

- **交叉验证t检验**

  - 对两个学习器A和B，使用k折交叉验证法得到的测试错误率分别为$\epsilon _ {1}^{A},\epsilon _ {2}^{A}...\epsilon _ {k}^{A}$和$\epsilon _ {1}^{B},\epsilon _ {2}^{B}...\epsilon _ {k}^{B}$，其中$\epsilon _ {i}^{A}$和$\epsilon _ {i}^{B}$是在相同的第i折训练/测试集上得到的结果

  - 如果两个学习器的性能相同，则它们使用相同的训练/测试集得到的测试错误率应相同

  - 先对每对结果求差$\Delta i=\epsilon _ {i}^{A}-\epsilon _ {i}^{B}$

    - 若两个学习器性能相同，则差值均值应该为0

    - 应该根据差值$\Delta 1,\Delta 2,...\Delta k$来对学习器A与B性能相同这个假设做t检验

    - 计算差值的均值$\mu$和方差$\sigma^{2}$

    - $\tau _ {t}= \vert \frac{\sqrt{k}\mu}{\sigma} \vert $

    - 小于$t _ {\alpha/2}(k-1)$，为接受域，两个学习器的性能没有显著差别

    - 大学概率补充知识

      <img src="/assets/MLpics/T10.png" style="zoom:50%;" />

  - 假设检验的重要前提：测试错误率均为泛化错误率的独立采样

    - 但通常情况下由于样本有限，所以在使用交叉验证时，不同轮次的训练集会有一定程度的重叠
    - 过高估计假设成立的概率
    - 采用5x2交叉验证
      - 做5次2折交叉验证
      - 在每次2折交叉验证之前随机将数据打乱，使得5次交叉验证中的数据划分不重复
      - 对两个学习器A，B，第i次2折交叉验证将产生两对错误测试率，对其分别求差，得到第1折上的差值$\Delta _ {i}^{1}$和第2折上的差值$\Delta _ {i}^{2}$
      - 为缓解测试错误率的非独立性，仅计算第1次2折交叉验证的两个结果的平均值$\mu=0.5(\Delta _ {1}^{1}+\Delta _ {1}^{2})$
      - 对每次2折实验的结果都计算其方差$\sigma _ {i}^{2}=(\Delta _ {i}^{1}-\frac{\Delta _ {1}^{1}+\Delta _ {1}^{2}}{2})^{2}+(\Delta _ {i}^{2}-\frac{\Delta _ {1}^{1}+\Delta _ {1}^{2}}{2})^{2}$
      - $\tau _ {t}=\frac{\mu}{\sqrt{0.2}\sum _ {i=1}^{5}\sigma _ {i}^{2}}$服从自由度为5的t分布

- **McNemar检验**

  - **列联表（contingency table）**是观测数据按两个或更多属性（定性变量）分类时所列出的频数表。它是由两个以上的变量进行交叉分类的频数分布表。

  - <img src="/assets/MLpics/T11.png" style="zoom:50%;" />

  - 若假设为两学习器性能相同，则应有$e _ {01}=e _ {10}$（实则为概率$p _ {e _ {01}}=p _ {e _ {10}}$）

  - 根据ALLEN L. EDWARDS的论文

    “NOTE ON THE "CORRECTION FOR CONTINUITY" IN TESTING THE SIGNIFICANCE OF THE DIFFERENCE BETWEEN CORRELATED PROPORTIONS ”

    <img src="/assets/MLpics/T15.png" style="zoom:40%;" />

  - 通过卡方分布进行评估，可将上表的$e _ {01}$与$e _ {10}$两个频率中较小的一个加上0.5、较大的一个减去0.5来进行**连续性校正**。

  - 那么变量$ \vert e _ {01}-e _ {10} \vert $应该服从正态分布，均值为1，方差为$e _ {01}+e _ {10}$

  - 变量$\tau _ {\chi}^{2}=\frac{( \vert e _ {01}-e _ {10} \vert -1)^{2}}{e _ {01}+e _ {10}}$

  - 服从自由度为1的**$\chi^{2}$分布**

  - 大学概率补充：

    <img src="/assets/MLpics/T12.png" style="zoom:50%;" />

- **Friedman检验**与Nemenyi后续检验

  - 当有多个算法参与比较时

    - 在每个数据集上分别列出两两比较的结果

      - 假定我们用$D _ {1},D _ {2},D _ {3},D _ {4}$四个数据集对ABC进行比较

      - 使用留出法/交叉验证法得到每个算法在每个数据集上的测试结果

      - 在每个数据集上根据测试性能由好到坏排序，并赋予序值

      - 若算法测试性能相同，则平分序值

        <img src="/assets/MLpics/T14.png" style="zoom:50%;" />

    - 使用基于算法排序的Friedman检验

      - 判断这些算法是否性能都相同（根据平均序值）

      - 我们在N个数据集上比较k个算法

        - $r _ {i}$表示第i个算法的平均序值

        - 暂时不考虑平分序值的情况

        - $r _ {i}$服从正态分布

          - 均值：$sum=k(k+1)/2$ 所以$\mu=sum/k=(k+1)/2$
        
          - 方差  
            
            $$ \begin{aligned} \delta^2 &=[(1-(k+1)/2)^{2}+(2-(k+1)/2)^{2}+...+(k-(k+1)/2)^{2}]/k \\ &=[(2-k-1)^{2}/4+(4-k-1)^{2}/4+...+(2k-k-1)^{2}/4]/k  \\ &= (4-2\times2(k+1)+(k+1)^{2}+...+(4k^{2}-2k\times2(k+1)+(k+1)^{2})/4k  \\ &=[(4+16+...+4k^{2})-2(k+1)(2+4+...+2k)+k(k+1)^{2}]/4k  \\ &=[4(1+2^{2}+...+k^{2})-4(k+1)(1+2+...+k)+k(k+1)^{2}]/4k \end{aligned} $$
      
            - 根据平方和公式（注意该图中n对应上式中k）<img src="/assets/MLpics/T16.png" style="zoom:40%;" />
      $$ \begin{aligned} \delta^2 &=[4k(k+1)(2k+1)/6-4(k+1)(1+2+...+k)+k(k+1)^{2}]/4k  \\ &=(k+1)[8k^{2}/6+4k/6-(k^{2}+k)]/4  \\ &=(k+1)[k^{2}/3-k/3]/4k \\ &=(k^{2}-1)/12 \end{aligned} $$
            
    - 所以其均值和方差分别为$(k+1)/2$和$(k^{2}-1)/12$，则
        
      $$ \begin{aligned} \tau _ {\chi}^{2}&=\frac{k-1}{k}\frac{12N}{k^{2}-1}\sum _ {i=1}^{k}(r _ {i}-\frac{k+1}{2})^{2}  \\ &=\frac{12N}{k(k+1)}(\sum _ {i=1}^{k}r _ {i}^{2}-\sum _ {i=1}^{k}r _ {i}(k+1)+\frac{k(k+1)^{2}}{4}) \\ &=\frac{12N}{k(k+1)}(\sum _ {i=1}^{k}r _ {i}^{2}-\frac{k(k+1)^{2}}{4}) \end{aligned}$$
        
  - 在k和N都较大的情况下，其服从自由度为k-1的$\chi^{2}$分布
        
  - 现在通常使用**F检验**
        
      - $\tau _ {F}=\frac{(N-1)\tau _ {\chi^{2}}}{N(k-1)-\tau _ {\chi^{2}}}$
        
      - 补充知识，**F分布**
        
        <img src="/assets/MLpics/T17.png" style="zoom:50%;" />
        
      - $\tau _ {F}$服从自由度为$k-1$和$(k-1)(N-1)$的F分布
    
- Nemenyi后续检验算法
    
  - 若所有算法的性能相同的假设被拒绝，需要进行“后续检验”来进一步区分各算法
    
  - Nemenyi检验计算出平均序值差别的临界值域
    
  - $CD=q _ {\alpha}\sqrt{\frac{k(k+1)}{6N}}$
    
    <img src="/assets/MLpics/T18.png" style="zoom:50%;" />
    
  - 若两个算法的平均序值之差超出了临界值域CD，则拒绝两个算法性能相同的假设
    
- 举个例子
    
  <img src="/assets/MLpics/T14.png" style="zoom:50%;" />
    
  - 先计算出$\tau _ {F}=24.429$，它大于$\alpha=0.05$时F的检验临界值，因此它拒绝“所有算法性能相同”的假设
    
      - 使用Nemenyi后续检验，$k=3$时$q _ {0.05}=2.344$，临界值域$CD=1.657$，所以算法A与B的差距，以及算法B与C的差距均未超过临界值域，而算法A和C的差距超过临界值域，所以认为A与C的性能显著不同
    
      - Friedman检验图如下图所示
    
        <img src="/assets/MLpics/T19.png" style="zoom:50%;" />
    
        - 两个横线段（临界值域）有交叠，则说明这两个算法没有显著差别

## 偏差与方差

- 偏差-方差分解

  - 是解释学习算法泛化性能的一种重要工具

  - 对学习算法的期望泛化错误率进行拆解

    - 算法在不同训练集（可能来自同一分布）上学得的结果可能不同

    - 对测试样本$x$，令$y _ {D}$为$x$在数据集中的标记，$y$为$x$的真实标记，$f(x;D)$为训练集D上学得模型$f$在$x$上的预测输出

    - 以回归任务为例

      - 学习算法期望

      <img src="/assets/MLpics/T20.png" style="zoom:40%;" />

      - **偏差**
        - 期望预测与真实结果的偏离程度
        - 学习算法本身的拟合能力
      - **方差**
        - 同样大小的训练集变动所导致的
        - 学习性能的变化数据扰动所造成的影响

    - 噪声

      - 在当前任务上任何学习算法所能达到的期望泛化误差的下界
      - 计算数据集中标记和真实标记的差别，属于学习该问题本身存在的差距
      - 学习问题本身的难度
      - 假定噪声期望为0，也就是<img src="/assets/MLpics/T21.png" style="zoom:40%;" />

    - 通过推导，可对算法的期望泛化误差进行分解

      <img src="/assets/MLpics/T22.png" style="zoom:50%;" />

    - 公式推导详细说明
      - 3-4: 
        
        $$ \begin{aligned} \mathbb{E} _ {D}[2(f(x;D)-\mathop{f}\limits_{}^-(x))\mathop{f}\limits_{}^-(x)] &= \mathbb{E} _ {D}[2(f(x;D)\mathop{f}\limits_{}^-(x)-\mathop{f}\limits_{}^-(x)^2)] \\ &= 2\mathop{f}\limits_{}^-(x)\mathbb{E} _ {D}(f(x;D))-2\mathop{f}\limits _ {}^-(x)^2 \\ &= 2\mathop{f}\limits _ {}^-(x)^2-2\mathop{f}\limits _ {}^-(x)^2=0\end{aligned} $$
        
        - $ \mathop{f}\limits _ {}^-(x)$为常量，并且$\mathbb{E} _ {D}(f(x;D))=\mathop{f}\limits _ {}^-(x) $，所以
    
          $$ \begin{aligned} \mathbb{E} _ {D}[2(f(x;D)-\mathop{f}\limits_{}^-(x))y_{D}] &= \mathbb{E} _ {D}[2(f(x;D)\mathop{f}y _ {D}-\mathop{f}\limits _ {}^-(x)y_{D})] \\ &= 2\mathbb{E} _ {D}(y _ {D})\mathbb{E} _ {D}(f(x;D))-2\mathop{f}\limits_{}^-(x)\mathbb{E} _ {D}(y _ {D}) \\ &= 2\mathop{f}\limits_{}^-(x)\mathbb{E} _ {D}(y _ {D})-2\mathop{f}\limits _ {}^-(x)\mathbb{E} _ {D}(y _ {D})=0 \end{aligned} $$
      
      - 6-7: $ \mathbb{E} _ {D}(2(\mathop{f}\limits _ {}^-(x)-y)(y-y _ {D}))=2(\mathop{f}\limits _ {}^-(x)-y)\mathbb{E} _ {D}(y-y _ {D}) $
        - $\mathbb{E} _ {D}(y-y _ {D})=0$
      - 原式=0
      
  - 最终结果$E(f;D)=var(x)+bias^2(x)+\epsilon^2$
  
      - 泛化误差可以分解为偏差、方差与噪声之和
    - 泛化性能是由学习算法的能力、数据的充分性和学习任务本身的难度所决定的
  
  - 偏差-方差窘境（bias-variance dilemma)
  
    - <img src="/assets/MLpics/T23.png" style="zoom:50%;" />
  - 在训练不足时，学习器的拟合能力不强，偏差大，方差小，此时偏差主导泛化错误率
    - 随着训练逐渐加深，拟合能力逐渐增强，方差逐渐主导泛化错误率
    - 当训练充足时，学习器拟合能力超强，方差大，偏差小，训练数据的微小变化都会导致学习器发生显著变化
      - 导致过拟合：训练数据自身的非全局的特性被学习到了，并且学习器使用这些无用的“特性”进行后续预测
