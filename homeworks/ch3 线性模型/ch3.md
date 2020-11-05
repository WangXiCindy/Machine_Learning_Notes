---
Typora-root-url: ../../assets/MLpics
---

## 习题

- 3.1 试析在什么情形下式（3.2）中不必考虑偏置项b

  - 式3.2：$f(x)=w^Tx+b$
  - 根据$b=\frac{1}{m} \sum _ {i=1}^{m}\left(y _ {i}-w x _ {i}\right)$，所有标志值之和等于所有加权属性值之和时，偏置项b等于0
  - 根据$y_i'=y_i-y_0=w(x_i-x_0)$当训练的样本集为原样本集中每个样本与任意一个样本之差时，不用考虑b

- 3.2 试证明，对于参数w，对率回归的目标函数（3.18）是非凸的，但其对数似然函数（3.27）是凸的

  - 凸函数就是一个定义在某个向量空间的凸子集C（区间）上的实值函数

    ![](/T33.png)

    - 定义

      <img src="/T34.png" style="zoom:70%;" />

      - 如果一个多元函数是凸函数，他的Hessian矩阵是半正定的

        - 多元函数的Hessian矩阵就类似一元函数的二阶导。
        - 多元函数Hessian矩阵半正定就相当于一元函数二阶导非负

      - **Hessian（海森）矩阵**

        - $H_{ij}=\frac{\partial^2f(x)}{\partial x_i \partial y_j}$

        ![](/T35.png)

      - 半正定：对任意不为0的是实列向量X，都有$X^TAX>=0$

      - 泰勒展开

        <img src="/assets/MLpics/T36.png" style="zoom:80%;" />

        ![](/assets/MLpics/T37.png)

      - Hessian矩阵和凸函数的关系

        - 直接在x0点处二阶展开形式$$\begin{equation}
          f(\mathbf{x})=f\left(\mathbf{x} _{0}\right)+\nabla f\left(\mathbf{x} _{0}\right)\left(\mathbf{x}-\mathbf{x} _{0}\right)+\frac{1}{2}\left(\mathbf{x}-\mathbf{x} _{0}\right)^{T} \mathbf{H}\left(\mathbf{x}-\mathbf{x} _{0}\right)
          \end{equation}$$
        - H为海森矩阵，$H=\nabla^2 f(x_0)$，也就是把梯度向量推广为二阶形式
          - 因为梯度向量本身也是雅可比行列式$\frac{\partial\left(u_{1}, u_{2}, \cdots, u_{n}\right)}{\partial\left(x_{1}, x_{2}, \cdots, x_{n}\right)}$的特例
        - 通过凸函数图像$f(x) \ge f(x_0)+\nabla f(x_0)(x-x_0)$对于任意的x和x0都成立
        - $\frac{1}{2}(x-x_0)^TH(x-x_0) \ge0$必须对于任意的x和x0都成立（凸函数的二阶条件$\nabla^2 f(x) \ge0$)
          - 也就是说对于任意$\Delta x	$，$\Delta x^TH\Delta x \ge0$恒成立
          - 正是H半正定的充要条件

  - 式3.18： $y=\frac{1}{1+e^{-(w^Tx+b)}}$      $y \in (0,1)$

    - 则对上式一阶求导，可得：

      $$\begin{aligned}\frac{dy}{dw}&=-(1+e^{-(w^Tx+b)})^{-2}\times e^{-(w^Tx+b)}(-x) \\&=\frac{xe^{-(w^Tx+b)}}{(1+e^{-(w^Tx+b)})^2} \\ &=\frac{x(1+e^{-(w^Tx+b)}-1)}{(1+e^{-(w^Tx+b)})^2} \\&= \frac{x}{1+e^{-(w^Tx+b)}}-\frac{x}{(1+e^{-(w^Tx+b)})^2} \\&=x(y-y^2)\end{aligned}$$

    - 对上式二阶求导，可得：
      
      $$\begin{aligned}\frac{d}{dw}(\frac{dy}{dw})&=x[y'-2yy']\\ &=x^Tx[y-y^2-2y^2+2y^3]\\ &=x^Tx[y-3y^2+2y^3]\end{aligned}$$
      
      - $x^Tx$半正定，$x^Tx \ge0$
      - 如果$y-y^2>0$则$y(1-y)<0$，$y>1$
      - 如果$y-3y^2+2y^3>0$则$y(1-2y)(1-y)>0$，则$(1-2y)>0$，$2y<1$，矛盾，所以非凸

  - 式3.27：$\ell(\beta)=\sum _ {i=1}^{m}(-y _ {i} \boldsymbol{\beta}^{\mathrm{T}} \hat{\boldsymbol{x}} _ {i}+\ln (1+e^{\boldsymbol{\beta}^{\mathrm{T}} \hat{\boldsymbol{x}} _ {i}}))$

    - 则：$\frac{d\ell(\beta)}{d \beta}=\sum _ {i=1}^{m}(-y _ {i}  \hat{x} _ {i}+\frac{e^{\beta^T \hat{x} _ i} \hat{x} _ i}{(1+e^{\beta^T \hat{x} _ i}))}$
    - 二阶求导：$\frac{d^2\ell(\beta)}{(d \beta)^2}=\sum _ {i=1}^{m}\frac{\hat{x} _ i^2e^{\beta^T \hat{x} _ i}}{(1+e^{\beta^T \hat{x} _ i})^2}$
      - 上式必大于等于0，所以为凸函数

- 3.3 编程实现对率回归，并给出西瓜数据集$3.0\alpha$的结果

  ```python
  #本部分为训练部分，使用误差反传
  #看起来更接近二次函数，so添加二次项
  def fit_double(X,y,eta=0.01,n_iters=500000,eps=1e-8):
      # 注意是wx+b，要多一行
      beta=np.ones((len(X),1))
  
      # 按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。
      data=np.c_[beta,X]
      
      # 计算密度的平方
      x2=np.square(data[:,[1]])
      # 添加密度平方项
      data=np.c_[data,x2]
      
      weights=np.ones((4,1))
      i_iters=0
      
      while i_iters<n_iters :
          y_sig=sigmoid(data.dot(weight))
          m=y_sig-y  #计算误差值
          weights=weights-data.transpose().dot(m)*eta   #误差反传更新参数
          i_iters+=1
          
      #打印最后的误差值
      print(np.abs(m).sum())
      
      return weights,data
  ```

  结果如下

  <img src="/T38.png" style="zoom:50%;" />

- 3.4 选择两个UCI数据集，比较10折交叉验证法和留一法所估计出的对率回归的错误率

  - 留一法

    ```python
    #留一法
    def leave_one():
        X,y=ReadExcel('./data.xlsx')
        total=X.shape[0]
        sum=0
        for k in range(total):
            test_index=k #测试集下标
            
            test_X=X[k]
            test_y=y[k]       
            
            train_X=np.delete(X,test_index,axis=0)
            train_y=np.delete(y,test_index,axis=0)
            
            weights,data=fit(train_X,train_y)
            
            y_pre=pre(test_X,weights,1)#代表使用留一法
            sum+=accuracy(y_pre,test_y,1)
            
        print('''LeaveOneOut's Accuracy: ''', sum / total)
    ```

  - 10折交叉验证法

  ```python
  # 10折交叉验证
  def cross_val():
      X,y=ReadExcel('./data.xlsx')
      total=X.shape[0]
      sum=0
      num_split=int(total/10)
      # 把样本分成10等分，依次抽取一个做测试集
      for k in range(10):
          test_index=range(k*num_split,(k+1)*num_split) #测试集下标
          
          test_X=X[test_index]
          test_y=y[test_index]
          
          train_X=np.delete(X,test_index,axis=0)
          train_y=np.delete(y,test_index,axis=0)
          
          weights,data=fit(train_X,train_y)
          
          y_pre=pre(test_X,weights,0)#代表使用非留一法
          sum+=accuracy(y_pre,test_y,0)
          
      print('''10-foldCrossValidation's Accuracy: ''', sum / total)
  ```

  留一法准确率高，运算时间长，准确率在Iris数据集上差距不大

- 3.5

  ```python
  def LDA(X0,X1):
      # X0 8行2列，X1 9行2列
      # mean0 1行2列
      mean0=np.mean(X0,axis=0,keepdims=True)# 仍然保持为矩阵而非vector
      mean1=np.mean(X1,axis=0,keepdims=True)
  
      # 由于最终结果是2x2的矩阵，所以不能按照公式写法
      Sw=(X0-mean0).T.dot(X0-mean0)+(X1-mean1).T.dot(X1-mean1)
  
      Sw=np.array(Sw,dtype='float')# 为了求逆
      omega=np.linalg.inv(Sw).dot((mean0-mean1).T)
  
      return omega
  ```

  结果如下图：

  <img src="/T42.png" style="zoom:50%;" />

- 3.6 线性判别分析仅在线性可分数据上能获得理想结果，试设计一个改进方法，使其能较好地用于非线性可分数据

  使用Kernal function

- 3.9 使用OvR和MvM将多分类任务分解为二分类任务求解时，试述为何无需专门针对类别不平衡性进行处理

  - 因为对于OvR和MvM来说，对于每个类进行了相同的处理，拆解出的二分类任务中类别不平衡的影响会相互抵消
  - 以ECOC编码为例，每个生成的二分类器会将所有样本分成较为均衡的二类，使类别不平衡的影响减小。
  - 但是拆解后仍然可能出现明显的类别不平衡现象，比如出现了一个非常大的类和一群小类。
