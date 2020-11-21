---
Typora-root-url: ../../assets/MLpics
---


## 习题

- 7.1 试使用极大似然法估算西瓜数据集3.0中前3个属性的类条件概率

  <img src="/T54.png" style="zoom:50%;" />

  - $P(色泽=青绿 \vert 好瓜)=p_1$

  - $P(色泽=乌黑 \vert 好瓜)=p_2$

  - $P(色泽=浅白 \vert 好瓜)=p_3=1-p_1-p_2$

  - 以$[p_1,p_2]$为参数求解，（概率的乘方要见3.0数据集好瓜所属色泽个数）

    $$\begin{aligned} P(D_{好瓜} \vert \theta_{好瓜})&=P(D_{好瓜} \vert [p_1,p_2])\\ &=\prod_{x \in D_{好瓜}}P(x \vert [p_1,p_2]) \\ &= p_1^3p_2^4(1-p_1-p_2)\end{aligned}$$

  - 似然函数：

    - $L(\xi)=\xi_1^3 \xi_2^4(1-\xi_1-\xi_2)$
    - $L(\xi_1)'=\xi_1^2\xi_2^4(3-4\xi_1-3\xi_2)$
    - $L(\xi_2)'=\xi_1^3\xi_2^3(4-4\xi_1-5\xi_2)$
    - $L(\xi_1)'=L(\xi_2)'=0$
    - $\xi_1=\frac{3}{8},\xi_2=\frac{1}{2},\xi_3=\frac{1}{8}$

- 7.2 试证明:条件独立性假设不成立时，朴素贝叶斯分类器仍有可能产生最优贝叶斯分类器。

  - 思路：作出朴素贝叶斯分类器和最优贝叶斯分类器的决策边界，在两个分类器决策边界之间以外的区域都是相通的。

- 7.3 试编程实现拉普拉斯修正的朴素贝叶斯分类器，并以西瓜数据集3.0为训练集，对p.151“测1”样本进行判别。

  - 属性值对应表

    | 属性 | 文字值 | 数字值 |
    | ---- | ------ | ------ |
    | 色泽 | 青绿   | 0      |
    | 色泽 | 乌黑   | 1      |
    | 色泽 | 浅白   | 2      |
    | 根蒂 | 蜷缩   | 0      |
    | 根蒂 | 稍蜷   | 1      |
    | 根蒂 | 硬挺   | 2      |
    | 敲声 | 浊响   | 0      |
    | 敲声 | 沉闷   | 1      |
    | 敲声 | 清脆   | 2      |
    | 纹理 | 清晰   | 0      |
    | 纹理 | 稍糊   | 1      |
    | 纹理 | 模糊   | 2      |
    | 脐部 | 凹陷   | 0      |
    | 脐部 | 稍凹   | 1      |
    | 脐部 | 平坦   | 2      |
    | 触感 | 硬滑   | 0      |
    | 触感 | 软粘   | 1      |

  - 代码

    ```python
    #条件概率计算
    def calContProb(data,feature):
        if(feature=="密度" or feature=="含糖率"):
            return calContProbCon(data,feature)
        else:
            return calContProbDis(data,feature)
            
            
    def calContProbCon(data,feature):
        
        prob=np.zeros((len(data['好瓜'].unique()),len(data[feature].unique())))
        
        for i in data['好瓜'].unique():
            for j in data[feature].unique():
                mean=data[data['好瓜']==i][feature].mean()
                std=data[data['好瓜']==i][feature].std()
                
                pos=continus[feature,str(j)]
                prob[i,pos]=exp(-((j-mean)*(j-mean))/(2*std*std))/(sqrt(2*math.pi)*std)
                prob[i,pos]=round(prob[i,pos],3)#保留三位小数
    
        return prob
        
    def calContProbDis(data,feature):
        
        
        ni=len(data[feature].unique())
        prob=np.zeros((len(data['好瓜'].unique()),len(data[feature].unique())))
        
        for i in data['好瓜'].unique():
            nc=len(data[data['好瓜']==i])+ni
            for j in data[feature].unique():
                #使用拉普拉斯修正
                prob[i,j]=(len(data[data['好瓜']==i][data[feature]==j])+1)/nc
                prob[i,j]=round(prob[i,j],3)#保留三位小数
                
        return prob
    
    #训练贝叶斯分类器
    def trainBayes(data,attribute):
        prob={}
        for i in attribute:
            if i!='好瓜':
                prob[i]=calContProb(pd.DataFrame(data, columns=['好瓜', i]), i)
        return prob
    ```

  - 结果：

    ```python
    判断为好瓜
    {1: 0.051175667069966325, 0: 0.00013825795281941037}
    ```

  - 注意：书上的P152条件概率计算有一定错误，注意分别

- 7.4 实践中使用式（7.5）决定分类类别时，若数据的维数非常高，则概率连乘的结果通常会非常接近于0从而导致下溢。试述防止下溢的可能方案。

  - 使用log，从而使乘法变成加法

- 7.5 试证明：二分类任务中两类数据满足高斯分布且方差相同时，线性判别分析产生贝叶斯最有分类器。

  - 线性判别分析

    - 根据式3.39，投影界面：$w=S_w^{-1}(\mu_0-\mu_1)=(\sum_0+\sum_1)^{-1}(\mu_0-\mu_1)$
    - 在两类数据方差相同时$w=\frac{1}{2}\sum^{-1}(\mu_1-\mu_0)$
    - 投影界面中点：$\frac{1}{2}(\mu_1+\mu_0)^Tw=\frac{1}{4}(\mu_1+\mu_0)^T\sum^{-1}(\mu_1-\mu_0)$
    - 决策边界：$g(x)=x^T\sum^{-1}(\mu_1-\mu_0)-\frac{1}{2}(\mu_1+\mu_0)^T \sum^{-1}(\mu_1-\mu_0)$

  - 贝叶斯最优分类器

    - $h^*(x)=arg_{c \in \mathcal{Y}}max\ P(c \mid x)$

    - $h^*(x)=arg_{c \in \mathcal{Y}}max\ P(x \mid c)P(c)$

    - 满足高斯分布（正态分布）

      $$\begin{aligned} h^*(x) &=arg_{c \in \mathcal{Y}}max \ \ log(f(x \mid c)P(c)) \\&=arg_{c \in \mathcal{Y}}max \ \ log(\frac{1}{(2 \pi)^{\frac{1}{2}}\vert \sum \vert^{\frac{1}{2}}}exp(-\frac{1}{2}(x-\mu_c)^T \sum^{-1}(x-\mu_c)))+log(P(c)) \\ &=arg_{c \in \mathcal{Y}}max-\frac{1}{2}(x-\mu_c)^T \sum^{-1}(x-\mu_c)+log(P(c)) \\&= arg_{c \in \mathcal{Y}}max\ x^T \sum^{-1}\mu_c-\frac{1}{2}\mu_c^T \sum^{-1} \mu_c+log(P(c))\end{aligned}$$

    - 贝叶斯决策边界为

      $$\begin{aligned} 
      g(x) &= x^{T} \Sigma^{-1} \mu_{1}-x^{T} \Sigma^{-1} \mu_{0}-(\frac{1}{2} \mu_{1}^{T} \Sigma^{-1} \mu_{1}-\frac{1}{2} \mu_{0}^{T} \Sigma^{-1} \mu_{0})+\log (\frac{P(1)}{P(0)}) \\
      &= x^{T} \Sigma^{-1}(\mu_{1}-\mu_{0})-\frac{1}{2}(\mu_{1}+\mu_{0})^{T} \Sigma^{-1}(\mu_{1}-\mu_{0})+\log (\frac{P(1)}{P(0)})
      \end{aligned}$$

- 7.6 试编程实现 AODE 分类器，并以西瓜数据集 3.0 为训练集，对 p.151的"测1" 样本进行判别。

  - 处理途中数据对应表

  - 代码

    ```python
    def calContProb(data,feature,attribute):
        if(feature!="密度" and feature!="含糖率"):
            return calContProbDis(data,feature,attribute)
        else:
            return
        
    def calContProbDis(data,feature,attribute):
        
        ni=len(data[feature].unique())
        prob=np.zeros((len(data['好瓜'].unique())))
        
        for i in data['好瓜'].unique():
            nc=len(data)+ni
            prob[i]=(len(data[data['好瓜']==i][data[feature]==0])+1)/nc
            for k in attribute:
                if (k!="密度" and k!="含糖率"):
                    prob[i]*=calContProbDis_xij(data,i,0,feature,k)
                else:
                    break
        return prob
    
    # 不同于拉普拉斯法，注意涉及xj,直接求乘积即可
    # 只需要针对xi=0和xj=0（也就是测试1计算即可）
    
    def calContProbDis_xij(data,c,xi,feature1,feature2):
        
        xj=0
        nc=len(data[data['好瓜']==c][data[feature1]==xi])+len(data[feature2].unique())
        prob=(len(data[data['好瓜']==c][data[feature1]==xi][data[feature2]==xj])+1)/nc
        return prob
      
    def trainAODE(data,attribute):
        prob={}
        for i in attribute: 
            if i!='好瓜':
                prob[i]=calContProb(data, i, attribute)        
        return prob
    ```

  - 结果

    ```python
    判断为好瓜
    {1: 0.09695613977659626, 0: 0.003268716304243502}
    ```

- 7.7 给定 d 个二值属性的二分类任务，假设对于任何先验概率项的估算至少需要30个样例，则在朴素贝叶斯分类器式中估算先验概率项需要60个样例。试估计在AODE式中估算先验概率项所需的样例数。（分别考虑最好和最坏情况）

  - 最好情况：每个类的属性都一致，需要30\*2=60个样例
  - 最坏情况：每个类属性都不一致，需要30\*2\*d=60d个样例

- 7.8 考虑图7.3，证明：在同父结构中，若$x_1$的取值未知，则$x_3 \perp x_4$不成立。在顺序结构中，$y \perp z \vert x$成立，但$y \perp z$不成立。

  - 同父结构：
    - $x_1$未知时，$p(x_1,x_3,x_4)=p(x_1)p(x_3 \vert x_1)p(x_4 \vert x_1)$
    - $p(x_3,x_4)=\sum_{x_1}p(x_1,x_3,x_4)$
    - 由于$x_1$取值未知，所以不成立
  - 顺序结构（注意图中箭头方向决定公式）：
    - x未知时，$p(x,y,z)=p(z)p(y \vert x)p(x \vert z)$
    - $p(y,z \vert x)=\frac{p(x,y,z)}{p(x)}=p(z \vert x)p(y \vert x)$
    - $p(y,z)=\sum_xp(x,y,z)$
    - 虽然$y \perp z \vert x$成立，但是无法得出$p(y,z)=p(y)p(z)$的结论

