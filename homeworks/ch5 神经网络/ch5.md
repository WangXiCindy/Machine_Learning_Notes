---
Typora-root-url: ../../assets/MLpics
---

## 习题

- 5.1 试述将线性函数$f(x)=w^Tx$用作神经元激活函数的缺陷

  - 神经网络中必须要有非线性的激活函数，如果全部用线性函数做激活函数，无论就是线性回归而非神经网络。

- 5.2 试述使用图5.2(b)激活函数的神经元与对率回归的联系

  <img src="/T65.png" style="zoom:50%;" />

  - 相同点：两者都是将连续值映射到{0,1}上
  - 不同点：激活函数不一定要使用sigmoid，只要是非线性的可导函数都可以使用。

- 5.3 对于图5.7中的$v_{ih}$，试推导出BP算法的更新公式

  见附录1

- 5.4 试述式（5.6）中学习率的取值对神经网络训练的影响

  - 式5.6 $\Delta w_{hj}=- \eta \frac{\delta E_k}{\delta w_{hj}}$
  - 学习率过大，可能会出现扰动现象，在最小值附近来回波动
  - 学习率过小，会导致下降过慢，迭代次数很多

- 5.5 试编程实现标准BP算法和累积BP算法，在西瓜数据集3.0上分别用这两个算法训练一个单隐层网络，并进行比较

  - ```python
    # 标准BP算法
    # hideNum为隐层神经元个数
    def BP(X,Y,hideNum=5,eta=0.01,epoch=1000000):
    
        # 权值及偏置初始化
        V = np.random.rand(X.shape[1],hideNum)
        V_b = np.random.rand(1,hideNum)
        W = np.random.rand(hideNum,Y.shape[1])
        W_b = np.random.rand(1,Y.shape[1])
    
        trainNum=0
        
        while trainNum<epoch:
            # 标准BP每次处理一个样本
            for k in range(X.shape[0]):
                B_h=sigmoid(X[k,:].dot(V)-V_b) # 输入层-隐层 注意是减去阈值
                Y_=sigmoid(B_h.dot(W)-W_b)     # 隐层-输出层 注意是减去阈值
                loss=sum((Y[k]-Y_)**2)*0.5      # 算均方误差
                
                # 计算梯度并更新参数
                g=Y_*(1-Y_)*(Y[k]-Y_)
                e=B_h*(1-B_h)*g.dot(W.T)
                
                # 参数更新
                W+=eta*B_h.T.dot(g)
                W_b-=eta*g
                V+=eta*X[k].reshape(1,X[k].size).T.dot(e)
                V_b-=eta*e
                trainNum+=1
                
        print("标准BP")
        print("总训练次数：",trainNum)
        print("最终损失：",loss)
    ```

  - ```python
    # 累积BP算法
    def BPAcc(X,Y,hideNum=5,eta=0.01,epoch=1000000):
        
        # 权值及偏置初始化
        V = np.random.rand(X.shape[1],hideNum)
        V_b = np.random.rand(1,hideNum)
        W = np.random.rand(hideNum,Y.shape[1])
        W_b = np.random.rand(1,Y.shape[1])
        
        trainNum=0
        
        while trainNum<epoch:
            # 累积BP直接处理所有样本
            B_h=sigmoid(X.dot(V)-V_b)   # 输入层-隐层 注意是减去阈值
            Y_=sigmoid(B_h.dot(W)-W_b)  # 隐层-输出层 注意是减去阈值
            loss=0.5*sum((Y-Y_)**2)/X.shape[0]     # 算均方误差
            
            # 计算梯度并更新参数
            g=Y_*(1-Y_)*(Y-Y_)
            e=B_h*(1-B_h)*g.dot(W.T)
                
            # 参数更新
            W+=eta*B_h.T.dot(g)
            W_b-=eta*g.sum(axis=0)
            V+=eta*X.T.dot(e)
            V_b-=eta*e.sum(axis=0)
            trainNum+=1
            
        print("累积BP")
        print("总训练次数：",trainNum)
        print("最终损失：",loss)
    ```

  - 结果

    <img src="/T82.png" style="zoom:50%;" />

    - 累积BP性能更好

- 5.6 试设计一个BP改进算法，能通过动态调整学习率显著提升收敛速度，编程实现该算法，并选择两个UCI数据集与标准BP算法进行实验比较

  - BP优化算法，其中自适应调节学习率相关领域现在比较火热
  - 推荐blog[深度学习 --- BP算法详解（BP算法的优化）](https://blog.csdn.net/weixin_42398658/article/details/83958133)

- 5.7 根据式（5.18）和（5.19），试构造一个能解决疑惑问题的单层RBF神经网络

  - ```python
    class RBF():
        # 权值及偏置初始化
        # 注意是单层RBF
        def __init__(self):
            self.hideNum=4
            self.epoch=10000
        
            self.w = np.random.rand(self.hideNum,1)
            self.beta = np.random.rand(self.hideNum,1)
            self.c=np.random.rand(self.hideNum,2)   #中心
            
        def forward(self,X):
            self.X=X
            self.dist=np.sum((X-self.c)**2,axis=1,keepdims=True)
            # 高斯径向基
            self.rho=np.exp(-self.beta*self.dist)# 注意径向基为激活函数，相当于BP的sigmoid
            self.y=self.w.T.dot(self.rho)
            # w第一位代表w_b,所以y第一位代表预测值
            return self.y[0, 0]
            
            
        # 梯度下降
        # 通过y回退
        def grad(self,y):
            grad_y=self.y-y
            grad_w=grad_y*self.rho
            grad_rho=grad_y*self.w
            grad_beta=-grad_rho*self.rho*self.dist
            grad_c=grad_rho*self.rho*2*self.beta*(self.X-self.c)
            self.grads = [grad_w, grad_beta, grad_c]
            
        # 参数更新
        def update(self,eta=0.01):
            self.w-=eta*self.grads[0]
            self.beta-=eta*self.grads[1]
            self.c-=eta*self.grads[2]
        
        def loss(self,X,y):
            y_=self.forward(X)
            loss=0.5*(y_-y)**2
            return loss
        
        def train(self,X,y):
            losses=[]
            for e in range(self.epoch):
                loss=0
                for i in range(len(X)):
                    self.forward(X[i])
                    self.grad(y[i])
                    self.update()
                    loss+=self.loss(X[i],y[i])
                    
                losses.append(loss)
            return losses
    ```

  - 结果

    <img src="/T83.png" style="zoom:50%;" />

