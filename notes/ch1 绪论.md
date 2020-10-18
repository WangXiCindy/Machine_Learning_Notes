---
Typora-root-url: ../assets/MLpics
---
# 绪论

## 基本术语

- 数据

  - 数据集：记录的集合

  - 示例：数据集中每条记录--关于一个事件/对象的描述

  - 属性/特征：反映事件或对象在某方面的表现或性质的事项

  - 属性/样本/输入空间：属性张成的空间

  - 特征向量：属性空间中的每一个点对应一个坐标向量---一个示例

    公式说明$$ D=\left\{ x _ {1},x _ {2},...,x _ {m} \right\} $$

    其中D代表数据集，xi代表示例$$x _ {i}=(x _ {i1};x _ {i2};...x _ {in})$$

- 学习/训练：从数据中学得模型的过程

  - 训练数据

  - 训练样本

  - 训练集

  - 假设：学得模型对应了关于数据的某种潜在的规律

  - 真相：潜在规律

  - 标签：一个示例的具体结果说明

  - 标记：结果信息（当然也存在标记空间）

  - 样例：拥有了标记信息的示例

    样例：$(x _ {i},y _ {i})$ $y _ {i}$是标记

  - 预测（监督学习）

    - 分类：预测的是离散值
      - 二分类(样本空间为$\{-1,+1\}$或$\{0,1\}$)
      - 多分类（样本空间的模>2）
    - 回归：预测的是连续值

  - 测试：学得模型后进行预测的过程

    - 测试样本：被预测的样本

  - 聚类（非监督学习）：有助于帮助我们发掘数据内在规律

  - 泛化：从样本很小空间得到的模型可以很好的适用于整个样本空间

    - 归纳：特殊到一般的泛化
    - 演绎：一般到特殊的特化

- 假设空间

  - 归纳学习

    - 狭义：从训练数据中学得概念，亦称概念学习/概念形成

      - 布尔概念学习：0/1

      案例：假设判断是否是好瓜有三种属性：

      色泽（3）	根蒂（2）	敲声（2）

      假设空间规模大小就是4x3x3+1=37

      （之所以为4x3x3是要考虑可能啥都行的情况，+1是因为要考虑可能根本没有好瓜的情况）

    - 广义：从样例中学习

  - 版本空间：可能有多个假设和训练集一致，这个假设集合就被称为这个训练集对应的版本空间

- 归纳偏好：在面临多个假设的时候无法判断哪一个更好，这时算法偏好就会起到作用

  - 引导算法

    - **奥卡姆剃刀**（Occam's razor）：若有多个假设与观察一致，则选最简单的那个（but简单的诠释也往往不同）

  - $\mathfrak{L} _ {a}$代表一种学习算法，对于不同的学习算法$\mathfrak{L} _ {a}$公和$\mathfrak{L} _ {b}$它们在不同的情况下会展现出不同的优势和缺陷

  - $\mathfrak{L} _ {a}$在训练集之外的所有样本上的误差为：

    $E _ {\text {ote}}\left(\mathfrak{L} _ {a} \mid X, f\right)=\sum _ {h} \sum _ {\boldsymbol{x} \in \mathcal{X}-X} P(\boldsymbol{x}) \mathbb{I}(h(\boldsymbol{x}) \neq f(\boldsymbol{x})) P\left(h \mid X, \mathfrak{L} _ {a}\right)$	（1.1）
  
    - $\mathbb{I}(·)$是指示函数，内部为真取1，否则取0
    - X为训练数据
    - $\mathcal{X}$为样本空间（离散）
    - $\mathcal{H}$为假设空间（离散）
    - $P\left(h \mid X, \mathfrak{L} _ {a}\right)$代表算法基于训练数据X产生假设h的概率
    - $f$我们希望学习的真实目标函数
    - 个人公式理解：
      - $P(x)$表示在训练数据以外（样本空间之内）该x出现的概率。
    - 在某一种假设和训练数据条件下，$\mathbb{I}(h(\boldsymbol{x}) \neq f(\boldsymbol{x})) P\left(h \mid X, \mathfrak{L} _ {a}\right)$为基于该训练数据关于x的预测结果和实际结果相同的概率
  
  - 考虑到二分类问题，且真实目标函数可以是任何函数，对所有可能的f按均匀分布对误差求和
    
    $$\begin{aligned}
    \sum _ {f} E _ {\text {ote}}\left(\mathfrak{L} _ {a} \mid X, f\right) &=\sum _ {f} \sum _ {h} \sum _ {\boldsymbol{x} \in \mathcal{X}-X} P(\boldsymbol{x}) \mathbb{I}(h(\boldsymbol{x}) \neq f(\boldsymbol{x})) P\left(h \mid X, \mathfrak{L} _ {a}\right)  \\ 
    &=\sum _ {\boldsymbol{x} \in \mathcal{X}-X} P(\boldsymbol{x}) \sum _ {h} P\left(h \mid X, \mathfrak{L} _ {a}\right) \sum _ {f} \mathbb{I}(h(\boldsymbol{x}) \neq f(\boldsymbol{x})) \qquad   \qquad   \qquad          (1.2) \\ 
    &=\sum _ {\boldsymbol{x} \in \mathcal{X}-X} P(\boldsymbol{x}) \sum _ {h} P\left(h \mid X, \mathfrak{L} _ {a}\right) \frac{1}{2} 2^{ \vert \mathcal{X} \vert }  \\ 
  &=\frac{1}{2} 2^{ \vert \mathcal{X} \vert } \sum _ {\boldsymbol{x} \in \mathcal{X}-X} P(\boldsymbol{x}) \sum _ {h} P\left(h \mid X, \mathfrak{L} _ {a}\right)
    \end{aligned}$$
  
  南瓜书具体说明：
  
    - 第1步到第2步
      - $\sum _ {i}^{m} \sum _ {j}^{n} \sum _ {k}^{o} a _ {i} b _ {j} c _ {k}=\sum _ {i}^{m} a _ {i} \cdot \sum _ {j}^{n} b _ {j} \cdot \sum _ {k}^{o} c _ {k}$
    - 第2步到第3步
      - 对于f的假设是任何能将样本映射到0，1的函数并且服从均匀分布。不止一个f且f出现的概率相等
      - 举个栗子：样本空间只有两个样本时($ \vert \mathcal{X} \vert =2$)，f的可能性$2^{ \vert \mathcal{X} \vert }$有四种，比如：$f _ {1}(x _ {1})=0,f _ {1}(x _ {2})=0;$
      - 所以通过$\mathfrak{L}  _ {a}$学习出的模型h(x)对 *每个样本* 无论是预测值为0还是1必然有一半的f与其预测值相等，例如$h(x _ {1})=1$则必有两个$f _ {n}(x _ {1})=1$
      - 所以说$\sum _ {f} \mathbb{I}(h(\boldsymbol{x}) \neq f(\boldsymbol{x}))=\frac{1}{2}2^{ \vert \mathcal{X} \vert }$
  - 最终结果：总误差和学习算法无关（NFL**“没有免费的午餐”** 定理） 
  
- NFL定理的bug前提：所有问题出现的机会相同/所有问题同等重要，但现实并非如此。
  
    - 一般来说我们只需要在某个具体的应用任务上找到一个解决方案
    - 瓜瓜🍉栗子：我们对好瓜会有一种评判标准，比如卖瓜的时候会喜欢买色泽青绿，根蒂蜷缩，敲声浊响的瓜，那么这种好瓜会更为常见，而根蒂硬挺，敲声清脆的好瓜罕见甚至不存在
